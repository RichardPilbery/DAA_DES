import streamlit as st

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# from _app_utils import create_logic_diagram
from _app_utils import get_text, get_text_sheet

text_df = get_text_sheet("info")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title(get_text("page_title", text_df))

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

st.caption(get_text("page_description", text_df))

(
    tab_model_logic,
    tab_demand_data,
    tab_activity_durations,
    tab_stand_downs,
    tab_resource_allocation,
) = st.tabs(
    [
        get_text("tab_1_name", text_df),
        get_text("tab_2_name", text_df),
        get_text("tab_3_name", text_df),
        get_text("tab_4_name", text_df),
        get_text("tab_5_name", text_df),
    ]
)

with tab_model_logic:
    st.caption(get_text("tab_1_content", text_df))
    # st.image(create_logic_diagram())
    st.image(
        "https://raw.githubusercontent.com/RichardPilbery/DAA_DES/refs/heads/main/reference/daa_des_model_logic.png",
        width=1200,
        use_container_width=True,
    )


with tab_demand_data:
    st.markdown(get_text("tab_2_content", text_df))

with tab_activity_durations:
    st.markdown(get_text("tab_3_content", text_df))

with tab_stand_downs:
    st.markdown(get_text("tab_4_content", text_df))

with tab_resource_allocation:
    st.markdown(get_text("tab_5_content", text_df))
