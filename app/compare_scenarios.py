import streamlit as st

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


st.title("Compare Scenarios")

st.write("Coming Soon!")
