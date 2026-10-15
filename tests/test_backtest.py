"""
Backtesting for the Devon Air Ambulance Discrete Event Simulation (DES) Model.

These run a fixed scenario with a fixed seed and check that the results are identical to a stored
baseline, so that any change to the model's results is noticed. They don't check whether the
results are realistic - see test_against_reality.py for comparisons with historic data.

The scenario uses the DEFAULT parameter csvs in actual_data (the parameters the web app starts
with), and the fitted distributions in distribution_data. It is run in the same way as the web
app, through parallelProcessJoblib and collateRunResults. actual_data and data are restored
afterwards.

## When a backtest fails

The results of the scenario have changed since the baseline was stored. If the change is
expected (e.g. a bug fix, or refitted distributions), check the new results (saved to
tests/test_outputs/BACKTEST_CURRENT_<metric>.csv) and then update the baseline with

    pytest -m backtest --update-backtest-baseline

and commit the updated files in tests/backtest_baseline alongside the change that caused them.
If the change isn't expected, something has changed the model's results unintentionally.

test_inputs_match_baseline shows whether any of the model's input files have changed since the
baseline was stored, which helps tell input changes apart from code changes.

## Results compared

Counts and times are totals across all runs, rather than averages, so they can be compared
exactly.

Planned tests are listed below with a [].
Implemented tests are listed below with a [x].

[x] Scenario settings match those used to create the baseline
[x] Input files match those used to create the baseline
[x] Calls by month
[x] Calls by hour of day
[x] Calls by day of week
[x] Jobs and total time on jobs by month and callsign
[x] Missed calls (no HEMS resource available) by month
[x] Missed calls (no HEMS resource available) by hour of day
[x] Number, median and mean duration of each job stage by month and vehicle type
"""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from air_ambulance_des.des_parallel_process import (
    collateRunResults,
    parallelProcessJoblib,
    removeExistingResults,
)

BASELINE_FOLDER = Path("tests/backtest_baseline")
METADATA_PATH = BASELINE_FOLDER / "metadata.json"
CURRENT_RESULTS_PATH = "tests/test_outputs/BACKTEST_CURRENT_{metric}.csv"

# The fixed scenario. Changing any of these means the baseline needs updating.
SCENARIO = {
    "total_runs": 5,
    "sim_duration_days": 730,
    "warm_up_hours": 0,
    "sim_start_date": "2023-01-01 08:00:00",
    "master_seed": 42,
    "demand_increase_percent": 1.0,
    "activity_duration_multiplier": 1.0,
    "amb_data": False,
}

# Files read by the model, whose changes would be expected to change the results
INPUT_FILE_PATTERNS = ["actual_data/*_DEFAULT.csv", "distribution_data/*"]
INPUT_FILES_IGNORED = ["distribution_data/README.md"]

JOB_STAGES = [
    "time_allocation",
    "time_mobile",
    "time_to_scene",
    "time_on_scene",
    "time_to_hospital",
    "time_to_clear",
]

DAYS_OF_WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Results are deterministic for a given seed, so this only allows for floating point differences
# between machines
RELATIVE_TOLERANCE = 1e-6


##############################################################################
# Running the scenario                                                       #
##############################################################################
def backup_folder(folder):
    backup = os.path.join(tempfile.mkdtemp(), "backup")
    shutil.copytree(folder, backup)
    return backup


def restore_folder(backup, folder):
    """Restore a folder to exactly match its backup, removing any files created since."""
    shutil.rmtree(folder)
    shutil.copytree(backup, folder)
    shutil.rmtree(os.path.dirname(backup), ignore_errors=True)


def use_default_parameters():
    """Overwrite each parameter csv in actual_data with its DEFAULT counterpart."""
    for default_path in Path("actual_data").glob("*_DEFAULT.csv"):
        shutil.copyfile(
            default_path, default_path.with_name(default_path.name.replace("_DEFAULT", ""))
        )


def run_scenario():
    """Run the fixed scenario, returning the combined event log of every run."""
    actual_data_backup = backup_folder("actual_data")
    data_backup = backup_folder("data")

    try:
        use_default_parameters()
        removeExistingResults(remove_run_results_csv=True)

        parallelProcessJoblib(
            total_runs=SCENARIO["total_runs"],
            sim_duration=SCENARIO["sim_duration_days"] * 24 * 60,
            warm_up_time=SCENARIO["warm_up_hours"] * 60,
            sim_start_date=datetime.fromisoformat(SCENARIO["sim_start_date"]),
            amb_data=SCENARIO["amb_data"],
            demand_increase_percent=SCENARIO["demand_increase_percent"],
            activity_duration_multiplier=SCENARIO["activity_duration_multiplier"],
            master_seed=SCENARIO["master_seed"],
        )
        collateRunResults()

        return pd.read_csv("data/run_results.csv")

    finally:
        restore_folder(actual_data_backup, "actual_data")
        restore_folder(data_backup, "data")


def input_file_hashes():
    """Return a fingerprint of each input file, ignoring differences in line endings."""
    paths = sorted(
        str(path).replace("\\", "/")
        for pattern in INPUT_FILE_PATTERNS
        for path in Path(".").glob(pattern)
        if path.is_file()
    )

    return {
        path: hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in paths
        if path not in INPUT_FILES_IGNORED
    }


##############################################################################
# Calculating the results compared                                           #
##############################################################################
def calls_by(arrivals, column, all_values):
    return (
        arrivals[column]
        .value_counts()
        .reindex(all_values, fill_value=0)
        .rename_axis(column)
        .reset_index(name="calls")
    )


def calculate_metrics(event_log):
    """
    Calculate each result compared from the event log. Returns a dictionary of
    (dataframe, key columns) tuples, keyed by metric name.
    """
    event_log = event_log.copy()
    event_log["timestamp_dt"] = pd.to_datetime(event_log["timestamp_dt"])

    patient = ["run_number", "P_ID"]

    arrivals = event_log[event_log["time_type"] == "arrival"][
        patient + ["timestamp_dt"]
    ].copy()
    arrivals["month"] = arrivals["timestamp_dt"].dt.strftime("%Y-%m")
    arrivals["hour"] = arrivals["timestamp_dt"].dt.hour
    arrivals["day_of_week"] = arrivals["timestamp_dt"].dt.strftime("%a")
    months = sorted(arrivals["month"].unique())

    # Jobs and time on jobs
    job_starts = event_log[event_log["event_type"] == "resource_use"][
        patient + ["time_type", "timestamp"]
    ].rename(columns={"time_type": "callsign", "timestamp": "start"})
    job_ends = event_log[event_log["event_type"] == "resource_use_end"][
        patient + ["timestamp"]
    ].rename(columns={"timestamp": "end"})
    jobs = job_starts.merge(job_ends, on=patient, how="left").merge(
        arrivals[patient + ["month"]], on=patient, how="left"
    )
    jobs["minutes_on_job"] = jobs["end"] - jobs["start"]

    jobs_by_month_and_callsign = (
        jobs.groupby(["month", "callsign"])
        .agg(
            jobs=("P_ID", "count"),
            # Jobs still in progress at the end of the simulation have no end time
            completed_jobs=("minutes_on_job", "count"),
            minutes_on_completed_jobs=("minutes_on_job", "sum"),
        )
        .reset_index()
    )

    # Missed calls
    missed = event_log[
        (event_log["event_type"] == "resource_request_outcome")
        & (event_log["time_type"] == "No Resource Available")
    ][patient].merge(arrivals, on=patient, how="left")

    # Job stage durations
    stages = event_log[event_log["event_type"].isin(JOB_STAGES)][
        patient + ["event_type", "time_type", "vehicle_type"]
    ].rename(columns={"event_type": "stage", "time_type": "minutes"})
    stages["minutes"] = stages["minutes"].astype(float)
    stages = stages.merge(arrivals[patient + ["month"]], on=patient, how="left")

    job_stage_durations = (
        stages.groupby(["month", "vehicle_type", "stage"])["minutes"]
        .agg(jobs="count", median_minutes="median", mean_minutes="mean")
        .reset_index()
    )

    return {
        "calls_by_month": (calls_by(arrivals, "month", months), ["month"]),
        "calls_by_hour": (calls_by(arrivals, "hour", range(24)), ["hour"]),
        "calls_by_day_of_week": (
            calls_by(arrivals, "day_of_week", DAYS_OF_WEEK),
            ["day_of_week"],
        ),
        "jobs_by_month_and_callsign": (
            jobs_by_month_and_callsign,
            ["month", "callsign"],
        ),
        "missed_calls_by_month": (
            calls_by(missed, "month", months).rename(columns={"calls": "missed_calls"}),
            ["month"],
        ),
        "missed_calls_by_hour": (
            calls_by(missed, "hour", range(24)).rename(columns={"calls": "missed_calls"}),
            ["hour"],
        ),
        "job_stage_durations_by_month": (
            job_stage_durations,
            ["month", "vehicle_type", "stage"],
        ),
    }


METRICS = [
    "calls_by_month",
    "calls_by_hour",
    "calls_by_day_of_week",
    "jobs_by_month_and_callsign",
    "missed_calls_by_month",
    "missed_calls_by_hour",
    "job_stage_durations_by_month",
]


##############################################################################
# Comparing with the baseline                                                #
##############################################################################
def find_differences(baseline, current, keys):
    """Return a dataframe listing every value that differs between the baseline and current results."""
    merged = baseline.merge(
        current, on=keys, how="outer", suffixes=("_baseline", "_current"), indicator=True
    )

    differences = []

    for _, row in merged[merged["_merge"] != "both"].iterrows():
        differences.append(
            {
                **{key: row[key] for key in keys},
                "column": "(whole row)",
                "baseline": "present" if row["_merge"] == "left_only" else "missing",
                "current": "present" if row["_merge"] == "right_only" else "missing",
            }
        )

    both = merged[merged["_merge"] == "both"]
    value_columns = [column for column in baseline.columns if column not in keys]

    for column in value_columns:
        old = both[f"{column}_baseline"].astype(float)
        new = both[f"{column}_current"].astype(float)
        changed = ~np.isclose(old, new, rtol=RELATIVE_TOLERANCE, atol=1e-9, equal_nan=True)

        for _, row in both[changed].iterrows():
            differences.append(
                {
                    **{key: row[key] for key in keys},
                    "column": column,
                    "baseline": row[f"{column}_baseline"],
                    "current": row[f"{column}_current"],
                }
            )

    return pd.DataFrame(differences)


@pytest.fixture(scope="module")
def backtest_results(request):
    """Run the scenario once, and update the baseline instead if requested."""
    metrics = calculate_metrics(run_scenario())

    if request.config.getoption("--update-backtest-baseline"):
        BASELINE_FOLDER.mkdir(exist_ok=True)

        for metric, (results, _) in metrics.items():
            results.to_csv(BASELINE_FOLDER / f"{metric}.csv", index=False)

        METADATA_PATH.write_text(
            json.dumps(
                {"scenario": SCENARIO, "input_files": input_file_hashes()}, indent=4
            )
            + "\n"
        )

        pytest.skip(f"Backtest baseline updated in {BASELINE_FOLDER}")

    if not METADATA_PATH.exists():
        pytest.fail(
            f"[FAIL - BACKTEST] No baseline found in {BASELINE_FOLDER}. "
            "Create one with: pytest -m backtest --update-backtest-baseline"
        )

    return metrics


##############################################################################
# Begin tests                                                                #
##############################################################################
@pytest.mark.backtest
def test_scenario_matches_baseline(backtest_results):
    baseline_scenario = json.loads(METADATA_PATH.read_text())["scenario"]

    assert baseline_scenario == SCENARIO, (
        f"[FAIL - BACKTEST - SCENARIO] The scenario has changed since the baseline was stored.\n"
        f"Baseline: {baseline_scenario}\nCurrent: {SCENARIO}\n"
        "If this is intended, update the baseline with: pytest -m backtest --update-backtest-baseline"
    )


@pytest.mark.backtest
def test_inputs_match_baseline(backtest_results):
    baseline_hashes = json.loads(METADATA_PATH.read_text())["input_files"]
    current_hashes = input_file_hashes()

    changed = sorted(
        path
        for path in set(baseline_hashes) | set(current_hashes)
        if baseline_hashes.get(path) != current_hashes.get(path)
    )

    assert not changed, (
        "[FAIL - BACKTEST - INPUTS] These input files have been added, removed or changed since "
        f"the baseline was stored: {changed}\n"
        "If this is intended, check the results and update the baseline with: "
        "pytest -m backtest --update-backtest-baseline"
    )


@pytest.mark.backtest
@pytest.mark.parametrize("metric", METRICS)
def test_results_match_baseline(backtest_results, metric):
    current, keys = backtest_results[metric]
    current.to_csv(CURRENT_RESULTS_PATH.format(metric=metric), index=False)

    baseline = pd.read_csv(BASELINE_FOLDER / f"{metric}.csv")
    # Read the current results the same way as the baseline, so the types match
    current = pd.read_csv(CURRENT_RESULTS_PATH.format(metric=metric))

    differences = find_differences(baseline, current, keys)

    assert differences.empty, (
        f"[FAIL - BACKTEST - RESULTS] {len(differences)} values in {metric} differ from the baseline "
        f"(showing up to 20):\n{differences.head(20).to_string(index=False)}\n\n"
        f"Current results saved to {CURRENT_RESULTS_PATH.format(metric=metric)}. If this change is "
        "expected, update the baseline with: pytest -m backtest --update-backtest-baseline"
    )
