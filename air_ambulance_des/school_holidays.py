"""
Functions for reading the school holiday dates that HEMS resource servicing is scheduled around.

School holiday dates may be stored in UK format (dd/mm/yyyy), as in the csvs in the actual_data
folder, or in ISO format (yyyy-mm-dd), as written by app/setup.py when a parameter template or
scenario is imported. UK format is tried first, so ambiguous dates such as 03/04/2026 are read
as 3rd April rather than 4th March.
"""

from datetime import date, datetime

import pandas as pd

UK_DATE_FORMAT = "%d/%m/%Y"


def parse_holiday_date(value) -> pd.Timestamp:
    """
    Convert a single school holiday date to a timestamp.

    Parameters
    ----------
    value : str, datetime.date, datetime.datetime or pandas.Timestamp
        The date to convert. Strings may be in UK (dd/mm/yyyy) or ISO (yyyy-mm-dd, optionally
        followed by a time) format.

    Returns
    -------
    pandas.Timestamp

    Raises
    ------
    ValueError
        If the value is not a date, or is a string in neither UK nor ISO format
        (e.g. a date that doesn't exist, such as 31/02/2026).
    """
    # Dates read from Excel are already datetimes, so don't need parsing
    if isinstance(value, date):
        return pd.Timestamp(value)

    text = str(value).strip()

    try:
        return pd.Timestamp(datetime.strptime(text, UK_DATE_FORMAT))
    except ValueError:
        pass

    try:
        parsed = pd.to_datetime(text, format="ISO8601")
    except ValueError:
        parsed = pd.NaT

    # Blank and missing values are converted to NaT rather than raising an error
    if pd.isna(parsed):
        raise ValueError(
            f"Could not read school holiday date '{value}'. "
            "Dates should be in dd/mm/yyyy or yyyy-mm-dd format."
        )

    return parsed


def parse_school_holidays(school_holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the start and end dates of a table of school holidays to timestamps, checking that
    no holiday ends before it starts.

    Parameters
    ----------
    school_holidays : pandas.DataFrame
        One row per holiday, with 'start_date' and 'end_date' columns.

    Returns
    -------
    pandas.DataFrame
        A copy of the table with 'start_date' and 'end_date' converted to timestamps.

    Raises
    ------
    ValueError
        If any date can't be read, or any holiday ends before it starts.
    """
    school_holidays = school_holidays.copy()

    for column in ["start_date", "end_date"]:
        school_holidays[column] = pd.to_datetime(
            school_holidays[column].apply(parse_holiday_date)
        )

    ends_before_start = school_holidays[
        school_holidays["end_date"] < school_holidays["start_date"]
    ]

    if not ends_before_start.empty:
        raise ValueError(
            "These school holidays end before they start:\n"
            f"{ends_before_start[['start_date', 'end_date']].to_string()}"
        )

    return school_holidays


def load_school_holidays(path: str) -> pd.DataFrame:
    """
    Read a school holidays csv, converting its start and end dates to timestamps.

    See parse_school_holidays for the checks made on the dates.
    """
    return parse_school_holidays(pd.read_csv(path))
