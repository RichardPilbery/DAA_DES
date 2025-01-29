import streamlit as st
from _app_utils import create_logic_diagram
import streamlit as st

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("Model Information")

with col2:
    st.image("app/assets/daa-logo.svg", width=300)


tab_model_logic, tab_demand_data,tab_activity_durations, tab_stand_downs = st.tabs(["Summary of Model Logic", "Demand Data", "Activity Durations", "Stand-downs"])

with tab_model_logic:
    st.image(create_logic_diagram())

with tab_demand_data:
    st.header("How has the data for demand been calculated?")

    st.write("Coming Soon!")

    st.subheader("Isn't this an underestimate of the true demand?")

    st.write("Coming Soon!")



with tab_activity_durations:
    st.header("How are the durations of calls calculated?")

    st.write("Coming Soon!")

with tab_stand_downs:
    st.header("How are stand-downs calculated?")

    st.write("Coming Soon!")
