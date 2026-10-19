import streamlit as st

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

from _app_utils import get_text, get_text_sheet

text_df = get_text_sheet("what_is")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title(get_text("page_title", text_df))

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

st.caption(get_text("page_description", text_df))

tab_1, tab_2, tab_3 = st.tabs(
    [
        get_text("tab_1_name", text_df),
        get_text("tab_2_name", text_df),
        get_text("tab_3_name", text_df),
    ]
)

with tab_1:
    st.write(get_text("tab_1_content", text_df))

with tab_2:
    st.write(get_text("tab_2_content", text_df))

with tab_3:
    st.write(get_text("tab_3_content", text_df))
