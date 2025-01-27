import streamlit as st

pg = st.navigation(

    {

        "Model Setup": [
            st.Page("setup.py", title="Choose Model Parameters")
        ],

        "Model Outputs": [

            st.Page("model.py", title="Run Simulation"),
            ],
        "Model Information": [

            st.Page("info.py", title="Model Information"),
            st.Page("what_is.py", title="Introduction to Simulation")
            ]

    }


     )

pg.run()
