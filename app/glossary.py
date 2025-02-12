import streamlit as st
import pandas as pd

from _app_utils import get_text, get_text_sheet

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

text_df=get_text_sheet("glossary")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title(get_text("page_title", text_df))

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

st.caption(get_text("page_description", text_df))

st.dataframe(
    pd.read_csv("app/glossary.csv"),
    use_container_width=True,
    hide_index=True,
)
