import pandas as pd
import numpy as np
import plotly.express as px
import re
import streamlit as st

# Temporarily placed in same folder as this as relative imports not behaving with Streamlit
from des_parallel_process import runSim
from utils import Utils

st.set_page_config(layout="wide")

st.title("Air Ambulance Simulation")

with st.sidebar:
    st.subheader("Model Inputs")

    st.markdown("#### Simulation Run Settings")
    sim_duration_input =  st.slider("Simulation Duration (days)", 1, 20, 3)

    warm_up_duration =  st.slider("Warm-up Duration (hours)", 1, 24*10, 36)
    st.markdown(f"The simulation will not start recording metrics until {(warm_up_duration / 24):.2f} days have elapsed")

    number_of_runs_input = st.slider("Number of Runs", 1, 20, 3)

    sim_start_date_input = st.date_input("Enter the Simulation Start Date")

button_run_pressed = st.button("Run simulation")

if button_run_pressed:
    progress_text = "Simulation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    with st.spinner('Simulating the system...'):

        results = []

        for run in range(number_of_runs_input):
            run_results = runSim(
                    run = run,
                    total_runs = number_of_runs_input,
                    sim_duration =sim_duration_input * 24 * 60 ,
                    warm_up_time = warm_up_duration * 60,
                    sim_start_date = sim_start_date_input
                )

            results.append(
                run_results
                )

            my_bar.progress((run+1)/number_of_runs_input, text=progress_text)


        results_all_runs = pd.concat(results)

        tab1, tab2, tab3 = st.tabs(
            ["Simple Exploration", "Full Dataset", "Temp"]
        )

        with tab1:
            st.plotly_chart(
                px.scatter(
                    results_all_runs,
                    x="timestamp",
                    y="run_number",
                    color="time_type")
            )

        with tab2:

            st.write(results_all_runs)
