import streamlit as st

default_inputs = {
    'num_helicopters': 2,
    "num_cars": 1,
    "demand_adjust_type": "Overall Demand Adjustment",
    "overall_demand_mult":100,
    "spring_demand_mult":100,
    "summer_demand_mult":100,
    "autumn_demand_mult":100,
    "winter_demand_mult":100,
    "amb_data": False,
    "sim_duration_input": 7,
    "warm_up_duration": 0,
    "number_of_runs_input": 5,
    "create_animation_input": False,
}

for session_state_key, session_state_default_value in default_inputs.items():
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = session_state_default_value

if "visited_setup_page" not in st.session_state:
    st.session_state["visited_setup_page"] = False

pg = st.navigation(

    {

        "Model Setup": [
            st.Page("setup.py", title="Choose Model Parameters")
        ],

        "Model Outputs": [

            st.Page("model.py", title="Run Simulation"),
            ],
        "Model Information": [
            st.Page("what_is.py", title="Introduction to Simulation"),
            st.Page("info.py", title="Model Information"),
            ]

    }


     )

pg.run()
