"""
Testing for the parameter files that can be imported into the Devon Air Ambulance Discrete Event
Simulation (DES) Model web app.

These check that the blank parameter template and every predefined scenario in the scenarios
folder contain everything the setup page (app/setup.py) reads when a file is uploaded or a
scenario is selected, so that importing them will not cause an error.

Planned tests are listed below with a [].
Implemented tests are listed below with a [x].

## File contents

[x] Each file contains all of the sheets read by the app
[x] Each sheet contains the columns read by the app
[x] Scenario details contain a name, description and notes
[x] Additional parameters contain every parameter read by the app
[x] Additional parameters can be converted into the formats used by the app

## Scenarios

[x] At least one scenario is provided
[x] Each scenario has a unique name
[x] Each scenario differs from the blank template in its parameters

## Importing in the web app

[] Selecting a predefined scenario overwrites the csvs in actual_data
[] Uploading a completed template overwrites the csvs in actual_data
[] Resetting to defaults after an import restores the DEFAULT csvs

"""

import pandas as pd
import pytest
from pathlib import Path

from helpers import read_parameter_file

TEMPLATE_PATH = "app/inputs/parameter_template.xlsx"

# Ignore lock files created while a file is open in Excel
SCENARIO_PATHS = sorted(
    str(f) for f in Path("scenarios").glob("*.xlsx") if not f.name.startswith("~$")
)

PARAMETER_FILE_PATHS = [TEMPLATE_PATH] + SCENARIO_PATHS

# Sheets read by app/setup.py, and the columns it relies on in each
REQUIRED_COLUMNS = {
    "scenario_details": ["what", "value"],
    "callsign_registration_lookup": [
        "registration",
        "callsign",
        "model",
        "vehicle_type",
        "callsign_group",
    ],
    "HEMS_ROTA": [
        "callsign",
        "category",
        "vehicle_type",
        "callsign_group",
        "summer_start",
        "winter_start",
        "summer_end",
        "winter_end",
    ],
    "service_schedules_by_model": [],
    "rota_start_end_months": [],
    "service_history": ["registration", "last_service"],
    "upper_allowable_time_bounds": [],
    "school_holidays": ["start_date", "end_date"],
    "additional_parameters": ["parameter", "value"],
}

REQUIRED_SCENARIO_DETAILS = ["scenario_name", "scenario_description", "scenario_notes"]

REQUIRED_ADDITIONAL_PARAMETERS = [
    "number_of_runs",
    "simulation_duration_days",
    "simulation_warm_up_duration_hours",
    "simulation_start_date",
    "simulation_start_time",
    "master_random_seed",
    "activity_duration_multiplier",
]

# Sheets holding model parameters (rather than descriptive text), used to check that a
# scenario is not just a copy of the template
PARAMETER_SHEETS = [sheet for sheet in REQUIRED_COLUMNS if sheet != "scenario_details"]

##############################################################################
# Begin tests                                                                #
##############################################################################


##################################
# File contents                  #
##################################
@pytest.mark.scenarios
def test_required_sheets_present():
    for path in PARAMETER_FILE_PATHS:
        parameter_file = read_parameter_file(path)

        missing_sheets = [
            sheet for sheet in REQUIRED_COLUMNS if sheet not in parameter_file
        ]

        assert not missing_sheets, (
            f"[FAIL - SCENARIO IMPORTING - SHEETS] {path} is missing sheets: {missing_sheets}"
        )


@pytest.mark.scenarios
def test_required_columns_present():
    for path in PARAMETER_FILE_PATHS:
        parameter_file = read_parameter_file(path)

        for sheet, columns in REQUIRED_COLUMNS.items():
            missing_columns = [
                col for col in columns if col not in parameter_file[sheet].columns
            ]

            assert not missing_columns, (
                f"[FAIL - SCENARIO IMPORTING - COLUMNS] {path} is missing columns in sheet {sheet}: {missing_columns}"
            )


@pytest.mark.scenarios
def test_scenario_details_complete():
    for path in PARAMETER_FILE_PATHS:
        scenario_details = read_parameter_file(path)["scenario_details"]

        for detail in REQUIRED_SCENARIO_DETAILS:
            value = scenario_details[scenario_details["what"] == detail]["value"]

            assert len(value) == 1 and pd.notna(value.values[0]), (
                f"[FAIL - SCENARIO IMPORTING - SCENARIO DETAILS] {path} has no value for {detail}"
            )


@pytest.mark.scenarios
def test_additional_parameters_complete():
    for path in PARAMETER_FILE_PATHS:
        additional_parameters = read_parameter_file(path)["additional_parameters"]

        for parameter in REQUIRED_ADDITIONAL_PARAMETERS:
            value = additional_parameters[
                additional_parameters["parameter"] == parameter
            ]["value"]

            assert len(value) == 1 and pd.notna(value.values[0]), (
                f"[FAIL - SCENARIO IMPORTING - ADDITIONAL PARAMETERS] {path} has no value for {parameter}"
            )


@pytest.mark.scenarios
def test_additional_parameters_usable():
    # Repeats the conversions applied in app/setup.py when a file is imported
    for path in PARAMETER_FILE_PATHS:
        additional_parameters = (
            read_parameter_file(path)["additional_parameters"]
            .set_index("parameter")["value"]
        )

        try:
            sim_start_date = additional_parameters["simulation_start_date"].strftime(
                "%Y-%m-%d"
            )

            pd.to_datetime(
                f"{sim_start_date} {additional_parameters['simulation_start_time']}"
            )

            float(additional_parameters["activity_duration_multiplier"])

        except (AttributeError, ValueError) as e:
            pytest.fail(
                f"[FAIL - SCENARIO IMPORTING - ADDITIONAL PARAMETERS] {path} has a start date, start time or activity duration multiplier in an unusable format: {e}"
            )

        for parameter in [
            "number_of_runs",
            "simulation_duration_days",
            "simulation_warm_up_duration_hours",
            "master_random_seed",
        ]:
            value = float(additional_parameters[parameter])

            assert value >= 0 and value == int(value), (
                f"[FAIL - SCENARIO IMPORTING - ADDITIONAL PARAMETERS] {path} has a value for {parameter} that is not a whole number of 0 or more: {value}"
            )


##################################
# Scenarios                      #
##################################
@pytest.mark.scenarios
def test_scenarios_provided():
    assert len(SCENARIO_PATHS) > 0, (
        "[FAIL - SCENARIO IMPORTING - SCENARIOS] No .xlsx files found in the scenarios folder"
    )


@pytest.mark.scenarios
def test_scenario_names_unique():
    scenario_names = []

    for path in PARAMETER_FILE_PATHS:
        scenario_details = read_parameter_file(path)["scenario_details"]

        scenario_names.append(
            scenario_details[scenario_details["what"] == "scenario_name"][
                "value"
            ].values[0]
        )

    duplicate_names = {name for name in scenario_names if scenario_names.count(name) > 1}

    assert not duplicate_names, (
        f"[FAIL - SCENARIO IMPORTING - SCENARIOS] Scenario names used by more than one file: {duplicate_names}"
    )


@pytest.mark.scenarios
def test_scenarios_differ_from_template():
    template = read_parameter_file(TEMPLATE_PATH)

    for path in SCENARIO_PATHS:
        scenario = read_parameter_file(path)

        assert not all(
            template[sheet].equals(scenario[sheet]) for sheet in PARAMETER_SHEETS
        ), (
            f"[FAIL - SCENARIO IMPORTING - SCENARIOS] {path} has the same parameters as the blank template"
        )
