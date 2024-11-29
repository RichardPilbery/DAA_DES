import streamlit as st

pg = st.navigation(
    [st.Page("app_model.py", title="Run Simulation"),
     st.Page("app_background.py", title="Model Information")]
     )

pg.run()
