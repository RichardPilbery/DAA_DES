"""
Testing for the parameter files that can be imported into the Devon Air Ambulance Discrete Event
Simulation (DES) Model web app.

These check that the blank parameter template and every predefined scenario in the scenarios
folder contain everything the setup page (app/setup.py) reads when a file is uploaded or a
scenario is selected, so that importing them will not cause an error.

They also run the setup page headlessly using Streamlit's AppTest to check that importing a file
in the web app overwrites the input data used by the model. These tests overwrite the csvs in
actual_data while running, and restore them afterwards.

Planned tests are listed below with a [].
Implemented tests are listed below with a [x].

## File contents

[x] Each file contains all of the sheets read by the app
[x] Each sheet contains the columns read by the app
[x] Scenario details contain a name, description and notes
[x] Additional parameters contain every parameter read by the app
[x] Additional parameters can be converted into the formats used by the app
[] Every callsign in HEMS_ROTA appears in callsign_registration_lookup, with the same vehicle
   type and callsign group
[] Every model in callsign_registration_lookup has a servicing schedule in
   service_schedules_by_model
[] Every registration in callsign_registration_lookup has an entry in service_history
[] Rota categories, vehicle types and models only use values listed in the allowable_values sheet
[] Rota start and end hours are whole numbers between 0 and 23
[] Summer and winter rota start and end months are valid months
[] School holidays cover the whole simulation period set in additional_parameters
[] The warm-up period is shorter than the simulation duration
[] The blank template contains the same parameters as the DEFAULT csvs in actual_data
   (ignoring row order and number formatting), so the two can't drift apart

## Scenarios

[x] At least one scenario is provided
[x] Each scenario has a unique name
[x] Each scenario differs from the blank template in its parameters
[] The model completes a short run using each scenario's parameters
[] Each scenario changes model results in the way its description implies
   (e.g. 24 hour running results in more calls being answered overnight)

## Importing in the web app

[x] Selecting a predefined scenario overwrites the csvs in actual_data
[x] Uploading a completed template overwrites the csvs in actual_data
[x] Resetting to defaults after an import restores the DEFAULT csvs
[] Choosing to set parameters from the Excel template shows the file uploader, not the manual
   rota builder
[] The download template button provides the parameter template file
[] Importing a file sets the simulation parameters in session state (number of runs, duration,
   warm-up, start date and time, seed and activity duration multiplier) to the values in the file
[] Importing a file updates the summer start and end months shown when switching to setting
   parameters manually
[] The list of predefined scenarios contains every file in the scenarios folder and nothing else
   (e.g. Excel lock files)

"""

import pandas as pd
import pytest
from io import StringIO
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

from helpers import read_parameter_file, backup_actual_data, restore_actual_data

# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys

# AppTest doesn't add the page's folder to the path in the way 'streamlit run' does
sys.path.append(str(Path(__file__).resolve().parent.parent / "app"))

TEMPLATE_PATH = "app/inputs/parameter_template.xlsx"

SETUP_PAGE_PATH = "app/setup.py"
SETUP_PAGE_TIMEOUT = 60  # seconds

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

# csvs in actual_data that app/setup.py overwrites when a file is imported, each of which has
# a DEFAULT counterpart
IMPORTED_CSVS = [
    "callsign_registration_lookup",
    "HEMS_ROTA",
    "service_schedules_by_model",
    "rota_start_end_months",
    "service_history",
    "upper_allowable_time_bounds",
    "school_holidays",
]

# csvs that are written straight from their sheet without their values being reformatted,
# so can be compared directly with the sheet
DIRECTLY_IMPORTED_CSVS = [
    "HEMS_ROTA",
    "service_schedules_by_model",
    "rota_start_end_months",
    "upper_allowable_time_bounds",
]


def get_csvs_not_matching_file(path):
    """Return the names of any csvs in actual_data that don't match the parameter file."""
    parameter_file = read_parameter_file(path)

    # app/setup.py drops these columns before writing the callsign lookup csv
    parameter_file["callsign_registration_lookup"] = parameter_file[
        "callsign_registration_lookup"
    ].drop(columns=["vehicle_type", "callsign_group"])

    mismatched_csvs = []

    for name in DIRECTLY_IMPORTED_CSVS + ["callsign_registration_lookup"]:
        # Write the sheet to csv and back so it is read in the same way as the csv
        expected = pd.read_csv(StringIO(parameter_file[name].to_csv(index=False)))
        actual = pd.read_csv(f"actual_data/{name}.csv")

        if not expected.equals(actual):
            mismatched_csvs.append(name)

    return mismatched_csvs

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


##################################
# Importing in the web app       #
##################################
@pytest.mark.scenarios
def test_selecting_scenario_overwrites_csvs():
    backup_folder = backup_actual_data()

    try:
        for path in SCENARIO_PATHS:
            at = AppTest.from_file(SETUP_PAGE_PATH, default_timeout=SETUP_PAGE_TIMEOUT)
            at.run()

            setup_type = at.radio[0]
            setup_type.set_value(
                next(option for option in setup_type.options if "scenario" in option)
            ).run()

            scenario_list = next(r for r in at.radio if r.label == "Select a scenario")
            scenario_list.set_value(Path(path).name).run()

            assert not at.exception, (
                f"[FAIL - SCENARIO IMPORTING - WEB APP] Error when selecting {path}: {at.exception}"
            )

            mismatched_csvs = get_csvs_not_matching_file(path)

            assert not mismatched_csvs, (
                f"[FAIL - SCENARIO IMPORTING - WEB APP] Selecting {path} did not overwrite: {mismatched_csvs}"
            )

    finally:
        restore_actual_data(backup_folder)


@pytest.mark.scenarios
def test_uploading_template_overwrites_csvs():
    backup_folder = backup_actual_data()

    try:
        # AppTest can't upload files, so instead the file uploader returns the path to each
        # scenario as though it had been uploaded
        for path in SCENARIO_PATHS:
            with patch("streamlit.file_uploader", return_value=path):
                at = AppTest.from_file(
                    SETUP_PAGE_PATH, default_timeout=SETUP_PAGE_TIMEOUT
                )
                at.run()

                setup_type = at.radio[0]
                setup_type.set_value(
                    next(
                        option
                        for option in setup_type.options
                        if "Excel Template" in option
                    )
                ).run()

            assert not at.exception, (
                f"[FAIL - SCENARIO IMPORTING - WEB APP] Error when uploading {path}: {at.exception}"
            )

            mismatched_csvs = get_csvs_not_matching_file(path)

            assert not mismatched_csvs, (
                f"[FAIL - SCENARIO IMPORTING - WEB APP] Uploading {path} did not overwrite: {mismatched_csvs}"
            )

    finally:
        restore_actual_data(backup_folder)


@pytest.mark.scenarios
def test_reset_after_import_restores_defaults():
    backup_folder = backup_actual_data()

    try:
        for path in SCENARIO_PATHS:
            at = AppTest.from_file(SETUP_PAGE_PATH, default_timeout=SETUP_PAGE_TIMEOUT)
            at.run()

            setup_type = at.radio[0]
            setup_type.set_value(
                next(option for option in setup_type.options if "scenario" in option)
            ).run()

            scenario_list = next(r for r in at.radio if r.label == "Select a scenario")
            scenario_list.set_value(Path(path).name).run()

            # The reset button is the first button on the page
            at.button[0].click().run()

            assert not at.exception, (
                f"[FAIL - SCENARIO IMPORTING - RESET] Error when resetting after selecting {path}: {at.exception}"
            )

            not_reset_csvs = [
                name
                for name in IMPORTED_CSVS
                if not pd.read_csv(f"actual_data/{name}.csv").equals(
                    pd.read_csv(f"actual_data/{name}_DEFAULT.csv")
                )
            ]

            assert not not_reset_csvs, (
                f"[FAIL - SCENARIO IMPORTING - RESET] Resetting after selecting {path} did not restore: {not_reset_csvs}"
            )

            scenario_list = next(r for r in at.radio if r.label == "Select a scenario")

            assert scenario_list.value == "None", (
                f"[FAIL - SCENARIO IMPORTING - RESET] Resetting after selecting {path} did not clear the selected scenario"
            )

    finally:
        restore_actual_data(backup_folder)
