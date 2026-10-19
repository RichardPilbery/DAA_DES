import streamlit as st
from datetime import datetime, date
import pandas as pd
import platform
# Note: following
# Dmitri's approach here to avoid issues with session state setting
# https://discuss.streamlit.io/t/mini-tutorial-initializing-widget-values-and-getting-them-to-stick-without-double-presses/31391/6

DEFAULT_INPUTS = {
    "num_helicopters": 2,
    "num_cars": 1,
    "demand_adjust_type": "Overall Demand Adjustment",
    # "overall_demand_mult": 100,
    "overall_demand_mult": 100,
    "spring_demand_mult": 100,
    "summer_demand_mult": 100,
    "autumn_demand_mult": 100,
    "winter_demand_mult": 100,
    "amb_data": False,
    "sim_duration_input": 365,
    "warm_up_duration": 0,
    "number_of_runs_input": 10,
    "create_animation_input": False,
    # "sim_start_date_input": date.today().strftime('%Y-%m-%d'),
    "sim_start_date_input": "2023-01-01",
    "sim_start_time_input": "08:00",
    "scenario_1_set": False,
    "scenario_2_set": False,
    "rota_initialised": False,
    # "activity_duration_multiplier": 1.0
    "activity_duration_multiplier": 1.0,
    "master_seed": 42,
    "debugging_messages_to_log": False,
    "summer_start_month_index": 3,
    "summer_end_month_index": 9,
}

# Adjust some parameters depending on whether it is running
# locally on a users computer (in which case parallel processing
# can be invoked) or on the Streamlit community cloud platform

# This check is a way to guess whether it's running on
# Streamlit community cloud
if platform.processor() == "":
    ADDITIONAL_INPUTS = {"sim_duration_input": 365 * 2, "number_of_runs_input": 5}

else:
    ADDITIONAL_INPUTS = {"sim_duration_input": 365 * 2, "number_of_runs_input": 12}

DEFAULT_INPUTS.update(ADDITIONAL_INPUTS)


# def setup_state():
#     for session_state_key, session_state_default_value in DEFAULT_INPUTS.items():
#         if session_state_key in st.session_state:
#             st.session_state[session_state_key] = st.session_state[session_state_key]
#         # else:
#         #     st.session_state[session_state_key] = session_state_default_value


def setup_state():
    if "rota_initialised" not in st.session_state:
        # Set the rota back to defaults, overwriting any changes made
        # the last time the app was run
        base_rota = pd.read_csv("actual_data/HEMS_ROTA_DEFAULT.csv")
        base_rota.to_csv("actual_data/HEMS_ROTA.csv", index=False)
        st.session_state.rota_initialised = True

    for session_state_key, session_state_default_value in DEFAULT_INPUTS.items():
        if session_state_key not in st.session_state:
            st.session_state[session_state_key] = session_state_default_value


def reset_to_defaults(reset_session_state=True, reset_csvs=True, notify=True):
    if reset_session_state:
        for session_state_key, session_state_default_value in DEFAULT_INPUTS.items():
            st.session_state[session_state_key] = session_state_default_value

    # Overwrite csvs in data folder with their DEFAULT counterparts
    # TODO: This could be refactored to cycle through each csv in this folder automatically.
    # All files in this folder should have a DEFAULT counterpart.
    if reset_csvs:
        callsign_registration_lookup = pd.read_csv(
            "actual_data/callsign_registration_lookup_DEFAULT.csv"
        )
        callsign_registration_lookup.to_csv(
            "actual_data/callsign_registration_lookup.csv", index=False
        )

        base_rota = pd.read_csv("actual_data/HEMS_ROTA_DEFAULT.csv")
        base_rota.to_csv("actual_data/HEMS_ROTA.csv", index=False)

        rota_start_end_months = pd.read_csv(
            "actual_data/rota_start_end_months_DEFAULT.csv"
        )
        rota_start_end_months.to_csv(
            "actual_data/rota_start_end_months.csv", index=False
        )

        models = pd.read_csv("actual_data/service_schedules_by_model_DEFAULT.csv")
        models.to_csv("actual_data/service_schedules_by_model.csv", index=False)

        school_holidays = pd.read_csv("actual_data/school_holidays_DEFAULT.csv")
        school_holidays.to_csv("actual_data/school_holidays.csv", index=False)

        service_history = pd.read_csv("actual_data/service_history_DEFAULT.csv")
        service_history.to_csv("actual_data/service_history.csv", index=False)

        service_schedules_by_model = pd.read_csv(
            "actual_data/service_schedules_by_model_DEFAULT.csv"
        )
        service_schedules_by_model.to_csv(
            "actual_data/service_schedules_by_model.csv", index=False
        )

        upper_allowable_time_bounds = pd.read_csv(
            "actual_data/upper_allowable_time_bounds_DEFAULT.csv"
        )
        upper_allowable_time_bounds.to_csv(
            "actual_data/upper_allowable_time_bounds.csv", index=False
        )

    if notify:
        st.toast(
            "All parameters have been reset to the default values",
            icon=":material/history:",
            duration="long",
        )


# TODO: Implement action
def set_scenario_1_params():
    st.toast(
        "Scenario 1 has been set up to use the current parameters",
        icon=":material/looks_one:",
    )

    # TODO: Implement action

    st.session_state.scenario_1_set = True


# TODO: Implement action
def set_scenario_2_params():
    st.toast(
        "Scenario 2 has been set up to use the current parameters",
        icon=":material/looks_two:",
    )

    # TODO: Implement action

    st.session_state.scenario_2_set = True
