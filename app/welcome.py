import streamlit as st

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

from _app_utils import get_text, get_text_sheet, COLORSCHEME
from streamlit_extras.stylable_container import stylable_container

text_df = get_text_sheet("welcome")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title(get_text("page_title", text_df))

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

st.write(get_text("page_description", text_df))

col_nav_1, col_nav_2 = st.columns(2)

with col_nav_1:
    with stylable_container(
        css_styles=f"""
                button {{
                        background-color: {COLORSCHEME["teal"]};
                        color: white;
                    }}
                    """,
        key="green_buttons",
    ):
        if st.button(
            "Want to run a simulation of the current service?\n\nHead straight to the simulation page.",
            icon=":material/play_circle:",
        ):
            st.switch_page("model.py")


with col_nav_2:
    with stylable_container(
        css_styles=f"""
                button {{
                        background-color: {COLORSCHEME["teal"]};
                        color: white;
                        border-color: {COLORSCHEME["navy"]};
                    }}
                    """,
        key="green_buttons",
    ):
        if st.button(
            "Want to change some parameters first?\n\nHead to the setup page.",
            icon=":material/display_settings:",
        ):
            st.switch_page("setup.py")
