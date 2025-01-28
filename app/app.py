import streamlit as st
from _state_control import setup_state

setup_state()

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
