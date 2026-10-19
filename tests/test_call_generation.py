"""
Testing for the generation of calls in the Devon Air Ambulance Discrete Event Simulation (DES)
Model, focusing on days where no calls are generated.

Days with no calls happen when demand is reduced (e.g. a demand multiplier of 0.9 turns a day
with 1 call into a day with 0), so the model must be able to skip over them.

These tests run the model directly rather than through runSim, so that no results are written to
the data folder. The model is run one event at a time, so that a model that stops advancing
through time fails the test rather than hanging.

Planned tests are listed below with a [].
Implemented tests are listed below with a [x].

## Days with no calls

[x] The model completes a run when the first day has no calls
[x] The model completes a run when a day with no calls follows a day with calls
[x] The model completes a run with several days with no calls in a row
[x] The model completes a run when no day has any calls, generating no patients
[x] The model completes a run with no calls on the first day when starting part way through a day
[x] The number of calls is decided exactly once for every day of the simulation
[x] Resources are still checked for servicing on days with no calls
[x] The model completes a run with reduced demand using the historic call numbers
"""

from datetime import datetime

import pandas as pd
import pytest

from air_ambulance_des.des_hems import DES_HEMS

# A day with calls takes a few hundred simulation events, so a model taking far more than this
# has stopped advancing through time
MAX_EVENTS_PER_DAY = 5000

MIDNIGHT = datetime(2023, 1, 1, 0, 0)


def make_model(sim_days, sim_start_date=MIDNIGHT, demand_increase_percent=1.0):
    return DES_HEMS(
        run_number=0,
        sim_duration=sim_days * 24 * 60,
        warm_up_duration=0,
        sim_start_date=sim_start_date,
        amb_data=False,
        random_seed=42,
        demand_increase_percent=demand_increase_percent,
        activity_duration_multiplier=1.0,
        print_debug_messages=False,
    )


def set_daily_calls(model, calls_per_day):
    """
    Replace the sampled number of calls each day with the given numbers, returning a list that
    records each number used.
    """
    remaining = iter(calls_per_day)
    used = []

    def next_day_calls(quarter, quarter_or_season):
        calls = next(remaining)
        used.append(calls)
        return calls

    model.utils.inc_per_day_samples = next_day_calls

    return used


def run_model(model):
    """
    Run the model in the same way as model.run(), without writing results to the data folder,
    failing if the model stops advancing through time. Returns the arrivals.
    """
    model.env.process(model.generate_calls())

    sim_end = model.sim_duration + model.warm_up_duration
    max_events = MAX_EVENTS_PER_DAY * (sim_end / (24 * 60) + 1)
    events = 0

    while model.env.peek() < sim_end:
        model.env.step()
        events += 1

        if events > max_events:
            pytest.fail(
                f"[FAIL - CALL GENERATION - STUCK] Model stopped advancing at {model.env.now} minutes "
                f"({model.sim_start_date + pd.Timedelta(minutes=model.env.now)})"
            )

    # The call generator should still be waiting to continue at the end of the simulation,
    # rather than having stopped early
    assert model.env.peek() >= sim_end, (
        f"[FAIL - CALL GENERATION - ENDED EARLY] Model ran out of events at {model.env.now} minutes"
    )

    results = pd.DataFrame(model.results_list)

    if results.empty:
        return results

    return results[results["time_type"] == "arrival"]


def arrivals_per_day(arrivals, sim_start_date, sim_days):
    """Return the number of arrivals on each day of the simulation, including days with none."""
    days = pd.date_range(pd.Timestamp(sim_start_date).normalize(), periods=sim_days + 1).date

    if arrivals.empty:
        return pd.Series(0, index=days)

    return (
        pd.to_datetime(arrivals["timestamp_dt"])
        .dt.date.value_counts()
        .reindex(days, fill_value=0)
    )


##############################################################################
# Begin tests                                                                #
##############################################################################


##################################
# Days with no calls             #
##################################
@pytest.mark.calls
@pytest.mark.parametrize(
    "calls_per_day",
    [
        pytest.param([0, 3, 3], id="first_day"),
        pytest.param([3, 0, 3], id="after_day_with_calls"),
        pytest.param([2, 0, 0, 0, 2], id="several_in_a_row"),
        pytest.param([3, 3, 0], id="last_day"),
    ],
)
def test_days_with_no_calls_skipped(calls_per_day):
    sim_days = len(calls_per_day)
    model = make_model(sim_days)
    used = set_daily_calls(model, calls_per_day)

    arrivals = run_model(model)

    assert used == calls_per_day, (
        f"[FAIL - CALL GENERATION - DAYS] Calls decided for {len(used)} days, expected {sim_days}"
    )

    per_day = arrivals_per_day(arrivals, MIDNIGHT, sim_days)

    for day, (calls, arrived) in enumerate(zip(calls_per_day, per_day)):
        if calls == 0:
            assert arrived == 0, (
                f"[FAIL - CALL GENERATION - NO CALLS] {arrived} arrivals on day {day + 1}, which should have no calls"
            )
        else:
            assert 0 < arrived <= calls, (
                f"[FAIL - CALL GENERATION - CALLS] {arrived} arrivals on day {day + 1}, expected between 1 and {calls}"
            )


@pytest.mark.calls
def test_no_calls_on_any_day():
    sim_days = 5
    model = make_model(sim_days, demand_increase_percent=0)

    arrivals = run_model(model)

    assert arrivals.empty, (
        f"[FAIL - CALL GENERATION - NO DEMAND] {len(arrivals)} arrivals when demand is 0"
    )


@pytest.mark.calls
def test_no_calls_on_first_day_starting_part_way_through_day():
    # Starting at 8am, so the simulation covers part of a third day
    sim_start_date = datetime(2023, 1, 1, 8, 0)
    calls_per_day = [0, 3, 0]

    model = make_model(2, sim_start_date=sim_start_date)
    used = set_daily_calls(model, calls_per_day)

    arrivals = run_model(model)

    assert used == calls_per_day, (
        f"[FAIL - CALL GENERATION - DAYS] Calls decided for {len(used)} days, expected 3"
    )

    per_day = arrivals_per_day(arrivals, sim_start_date, 2)

    assert list(per_day == 0) == [True, False, True], (
        f"[FAIL - CALL GENERATION - PART DAY] Arrivals per day were {list(per_day)}, expected only day 2 to have arrivals"
    )

    # The first arrival can't be before midnight on day 2, so time must have skipped to then
    assert pd.to_datetime(arrivals["timestamp_dt"]).min() >= pd.Timestamp(2023, 1, 2), (
        "[FAIL - CALL GENERATION - PART DAY] Arrival before the first day with calls"
    )


@pytest.mark.calls
def test_servicing_checked_on_days_with_no_calls():
    calls_per_day = [3, 0, 0, 3]
    model = make_model(len(calls_per_day))
    set_daily_calls(model, calls_per_day)

    original_check = model.hems_resources.daily_servicing_check
    dates_checked = []

    def recording_check(current_dt, hour, month):
        dates_checked.append(current_dt.date())
        return original_check(current_dt, hour, month)

    model.hems_resources.daily_servicing_check = recording_check

    run_model(model)

    expected_dates = list(pd.date_range(MIDNIGHT, periods=len(calls_per_day)).date)

    assert dates_checked == expected_dates, (
        f"[FAIL - CALL GENERATION - SERVICING] Servicing checked on {dates_checked}, expected {expected_dates}"
    )


@pytest.mark.calls
def test_reduced_demand_with_historic_calls():
    # With demand reduced to 20%, any day with fewer than 5 calls historically has no calls
    demand_increase_percent = 0.2
    sim_days = 60

    model = make_model(sim_days, demand_increase_percent=demand_increase_percent)

    sample_calls = model.utils.inc_per_day_samples
    sampled = []

    def recording_sample(quarter, quarter_or_season):
        calls = sample_calls(quarter, quarter_or_season)
        sampled.append(calls)
        return calls

    model.utils.inc_per_day_samples = recording_sample

    arrivals = run_model(model)

    calls_per_day = [int(calls * demand_increase_percent) for calls in sampled]

    # Check this seed produces both kinds of day, so the test covers moving between them
    assert 0 in calls_per_day and any(calls_per_day), (
        f"[FAIL - CALL GENERATION - REDUCED DEMAND] Expected a mix of days with and without calls, got {calls_per_day}"
    )

    assert len(calls_per_day) == sim_days, (
        f"[FAIL - CALL GENERATION - REDUCED DEMAND] Calls decided for {len(calls_per_day)} days, expected {sim_days}"
    )

    per_day = arrivals_per_day(arrivals, MIDNIGHT, sim_days)

    days_with_unexpected_arrivals = [
        day + 1
        for day, (calls, arrived) in enumerate(zip(calls_per_day, per_day))
        if calls == 0 and arrived > 0
    ]

    assert not days_with_unexpected_arrivals, (
        f"[FAIL - CALL GENERATION - REDUCED DEMAND] Arrivals on days that should have no calls: {days_with_unexpected_arrivals}"
    )
