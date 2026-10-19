import streamlit as st
from _state_control import setup_state, reset_to_defaults
from _app_utils import get_quarto
import platform

# Ensure state parameters (necessary for passing parameters between separate
# pages) are defined
setup_state()

if "visited_setup_page" not in st.session_state:
    st.session_state["visited_setup_page"] = False

if "first_load" not in st.session_state:
    reset_to_defaults(reset_session_state=False, reset_csvs=True, notify=False)
    st.session_state["first_load"] = False

if platform.processor() == "":
    get_quarto("daa_des")  # This name must match the repository name on GitHub

pg = st.navigation(
    {
        "Model Setup": [
            st.Page("welcome.py", title="Welcome"),
            st.Page("setup.py", title="Choose Model Parameters"),
        ],
        "Model Outputs": [
            st.Page("model.py", title="Run Simulation"),
            # st.Page("compare_scenarios.py", title="Scenario Comparison")
        ],
        "Additional Information": [
            # st.Page("what_is.py", title="Introduction to Simulation"),
            st.Page("glossary.py", title="Glossary of Terms"),
            st.Page("info.py", title="Model Details"),
            st.Page("acknowledgements.py", title="Acknowledgements"),
        ],
    }
)

pg.run()
