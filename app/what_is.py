import streamlit as st

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("An Introduction to Discrete Event Simulation Modelling")

with col2:
    st.image("app/assets/daa-logo.svg", width=300)


tab_intro, tab_benefits, tab_limitations = st.tabs(
    "An Introduction to Simulation Modelling",
    "Benefits of Simulation Modelling",
    "Limitations of Simulation Modelling"
)

with tab_intro:
    st.write("Coming Soon!")

with tab_benefits:
    st.write("Coming Soon!")

with tab_limitations:
    st.write("Coming Soon!")
