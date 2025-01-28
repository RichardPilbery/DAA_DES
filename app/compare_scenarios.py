import streamlit as st

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("Compare Scenarios")

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

col_scenario_1, col_scenario_blank, col_scenario_2 = st.columns([0.45, 0.1, 0.45])

with col_scenario_1:
    st.header("Scenario 1")

    st.write("Coming Soon!")

with col_scenario_2:
    st.header("Scenario 2")

    st.write("Coming Soon!")
