import streamlit as st
from datetime import datetime

# Note: following
# Dmitri's approach here to avoid issues with session state setting
# https://discuss.streamlit.io/t/mini-tutorial-initializing-widget-values-and-getting-them-to-stick-without-double-presses/31391/6

DEFAULT_INPUTS = {
    'num_helicopters': 2,
    "num_cars": 1,
    "demand_adjust_type": "Overall Demand Adjustment",
    "overall_demand_mult": 100,
    "spring_demand_mult": 100,
    "summer_demand_mult": 100,
    "autumn_demand_mult": 100,
    "winter_demand_mult": 100,
    "amb_data": False,
    "sim_duration_input": 365,
    "warm_up_duration": 0,
    "number_of_runs_input": 20,
    "create_animation_input": False,
    "sim_start_date_input": datetime.today().strftime('%Y-%m-%d'),
    "sim_start_time_input": "08:00",

    "scenario_1_set": False,
    "scenario_2_set": False
}

# def setup_state():
#     for session_state_key, session_state_default_value in DEFAULT_INPUTS.items():
#         if session_state_key in st.session_state:
#             st.session_state[session_state_key] = st.session_state[session_state_key]
#         # else:
#         #     st.session_state[session_state_key] = session_state_default_value

def setup_state():
    for session_state_key, session_state_default_value in DEFAULT_INPUTS.items():
        if session_state_key not in st.session_state:
            st.session_state[session_state_key] = session_state_default_value


def reset_to_defaults():
    for session_state_key, session_state_default_value in DEFAULT_INPUTS.items():
            st.session_state[session_state_key] = session_state_default_value
    st.toast("All parameters have been reset to the default values",
             icon=":material/history:")

# TODO: Implement action
def set_scenario_1_params():
    st.toast("Scenario 1 has been set up to use the current parameters",
             icon=":material/looks_one:")

    # TODO: Implement action

    st.session_state.scenario_1_set = True

# TODO: Implement action
def set_scenario_2_params():
    st.toast("Scenario 2 has been set up to use the current parameters",
             icon=":material/looks_two:")

    # TODO: Implement action

    st.session_state.scenario_2_set = True
