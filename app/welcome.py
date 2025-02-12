import streamlit as st
from _app_utils import get_text, get_text_sheet

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

text_df=get_text_sheet("welcome")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title(get_text("page_title", text_df))

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

st.write(get_text("page_description", text_df))
