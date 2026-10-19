"""
Testing for the school holiday dates that HEMS resource servicing is scheduled around in the
Devon Air Ambulance Discrete Event Simulation (DES) Model.

These check that school holiday dates are read correctly whether they are stored in UK
(dd/mm/yyyy) or ISO (yyyy-mm-dd) format, and that the school holidays in the actual_data csvs,
the parameter templates and the predefined scenarios are plausible.

Planned tests are listed below with a [].
Implemented tests are listed below with a [x].

## Reading dates

[x] UK format dates are read day first, including ambiguous dates such as 03/04/2026
[x] ISO format dates are read, with or without a time
[x] Dates that are already datetimes (e.g. read from Excel) are left unchanged
[x] Values that aren't valid dates in either format raise an error
[x] A school holidays table where a holiday ends before it starts raises an error
[x] A school holidays csv is read into timestamps

## School holiday data

[x] Every school holiday file can be read, and no holiday ends before it starts
[x] No school holiday lasts longer than 7 weeks
[x] The average school holiday lasts between 1 and 3 weeks
[] School holidays cover the whole simulation period set in additional_parameters
"""

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from air_ambulance_des.school_holidays import (
    load_school_holidays,
    parse_holiday_date,
    parse_school_holidays,
)
from helpers import read_parameter_file

SCHOOL_HOLIDAY_CSV_PATHS = [
    "actual_data/school_holidays.csv",
    "actual_data/school_holidays_DEFAULT.csv",
]

# Ignore lock files created while a file is open in Excel
PARAMETER_FILE_PATHS = sorted(
    str(f)
    for f in list(Path("app/inputs").glob("*.xlsx")) + list(Path("scenarios").glob("*.xlsx"))
    if not f.name.startswith("~$")
)

MAX_HOLIDAY_WEEKS = 7
MIN_MEAN_HOLIDAY_WEEKS = 1
MAX_MEAN_HOLIDAY_WEEKS = 3


def read_all_school_holidays():
    """Return the parsed school holidays from every csv and parameter file, keyed by source."""
    school_holidays = {
        path: load_school_holidays(path) for path in SCHOOL_HOLIDAY_CSV_PATHS
    }

    for path in PARAMETER_FILE_PATHS:
        school_holidays[path] = parse_school_holidays(
            read_parameter_file(path)["school_holidays"]
        )

    return school_holidays


def holiday_durations_in_weeks(school_holidays):
    """Return the length of each holiday in weeks, counting both the start and end date."""
    return ((school_holidays["end_date"] - school_holidays["start_date"]).dt.days + 1) / 7


##############################################################################
# Begin tests                                                                #
##############################################################################


##################################
# Reading dates                  #
##################################
@pytest.mark.holidays
@pytest.mark.parametrize(
    "value, expected",
    [
        ("26/10/2026", pd.Timestamp(2026, 10, 26)),
        # Ambiguous dates should be read as UK format, not US format
        ("03/04/2026", pd.Timestamp(2026, 4, 3)),
        ("01/11/2026", pd.Timestamp(2026, 11, 1)),
        (" 01/11/2026 ", pd.Timestamp(2026, 11, 1)),
    ],
)
def test_uk_dates_read_day_first(value, expected):
    assert parse_holiday_date(value) == expected, (
        f"[FAIL - SCHOOL HOLIDAYS - UK DATES] '{value}' read as {parse_holiday_date(value)}, expected {expected}"
    )


@pytest.mark.holidays
@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-04-03", pd.Timestamp(2026, 4, 3)),
        ("2026-11-01", pd.Timestamp(2026, 11, 1)),
        ("2026-11-01 00:00:00", pd.Timestamp(2026, 11, 1)),
    ],
)
def test_iso_dates_read(value, expected):
    assert parse_holiday_date(value) == expected, (
        f"[FAIL - SCHOOL HOLIDAYS - ISO DATES] '{value}' read as {parse_holiday_date(value)}, expected {expected}"
    )


@pytest.mark.holidays
@pytest.mark.parametrize(
    "value",
    [
        date(2026, 4, 3),
        datetime(2026, 4, 3),
        pd.Timestamp(2026, 4, 3),
    ],
)
def test_datetimes_unchanged(value):
    assert parse_holiday_date(value) == pd.Timestamp(2026, 4, 3), (
        f"[FAIL - SCHOOL HOLIDAYS - DATETIMES] {value!r} read as {parse_holiday_date(value)}"
    )


@pytest.mark.holidays
@pytest.mark.parametrize(
    "value",
    [
        "31/02/2026",  # Day doesn't exist
        "13/13/2026",
        "not a date",
        "",
        float("nan"),
    ],
)
def test_invalid_dates_raise_error(value):
    with pytest.raises(ValueError):
        parse_holiday_date(value)


@pytest.mark.holidays
def test_holiday_ending_before_start_raises_error():
    school_holidays = pd.DataFrame(
        {
            "year": [2026, 2026],
            "start_date": ["27/07/2026", "26/10/2026"],
            "end_date": ["02/09/2026", "01/10/2026"],
        }
    )

    with pytest.raises(ValueError, match="end before they start"):
        parse_school_holidays(school_holidays)


@pytest.mark.holidays
def test_csv_read_into_timestamps(tmp_path):
    csv_path = tmp_path / "school_holidays.csv"

    pd.DataFrame(
        {
            "year": [2026, 2026],
            "start_date": ["03/04/2026", "2026-10-26"],
            "end_date": ["19/04/2026", "2026-11-01"],
        }
    ).to_csv(csv_path, index=False)

    school_holidays = load_school_holidays(csv_path)

    assert list(school_holidays["start_date"]) == [
        pd.Timestamp(2026, 4, 3),
        pd.Timestamp(2026, 10, 26),
    ], "[FAIL - SCHOOL HOLIDAYS - CSV] Start dates not read correctly"

    assert list(school_holidays["end_date"]) == [
        pd.Timestamp(2026, 4, 19),
        pd.Timestamp(2026, 11, 1),
    ], "[FAIL - SCHOOL HOLIDAYS - CSV] End dates not read correctly"


##################################
# School holiday data            #
##################################
@pytest.mark.holidays
def test_school_holiday_files_readable():
    # Reading each file checks that its dates are valid and no holiday ends before it starts
    school_holidays = read_all_school_holidays()

    assert len(school_holidays) == len(SCHOOL_HOLIDAY_CSV_PATHS) + len(
        PARAMETER_FILE_PATHS
    ), "[FAIL - SCHOOL HOLIDAYS - FILES] Not every school holiday file was read"


@pytest.mark.holidays
def test_no_holiday_longer_than_max():
    for path, school_holidays in read_all_school_holidays().items():
        durations = holiday_durations_in_weeks(school_holidays)
        too_long = school_holidays[durations > MAX_HOLIDAY_WEEKS]

        assert too_long.empty, (
            f"[FAIL - SCHOOL HOLIDAYS - MAX DURATION] {path} has holidays longer than {MAX_HOLIDAY_WEEKS} weeks:\n"
            f"{too_long[['start_date', 'end_date']].to_string()}"
        )


@pytest.mark.holidays
def test_mean_holiday_duration_plausible():
    for path, school_holidays in read_all_school_holidays().items():
        mean_weeks = holiday_durations_in_weeks(school_holidays).mean()

        assert MIN_MEAN_HOLIDAY_WEEKS <= mean_weeks <= MAX_MEAN_HOLIDAY_WEEKS, (
            f"[FAIL - SCHOOL HOLIDAYS - MEAN DURATION] Holidays in {path} last {mean_weeks:.1f} weeks on average, "
            f"expected between {MIN_MEAN_HOLIDAY_WEEKS} and {MAX_MEAN_HOLIDAY_WEEKS} weeks"
        )
