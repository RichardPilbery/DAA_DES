import pandas as pd
import numpy as np
import plotly.express as px
import re
from datetime import datetime
import streamlit as st

# Temporarily placed in same folder as this as relative imports not behaving with Streamlit
from des_parallel_process import runSim
from utils import Utils

st.set_page_config(layout="wide")

st.title("Air Ambulance Simulation")

with st.sidebar:
    st.subheader("Model Inputs")

    st.markdown("#### Simulation Run Settings")
    sim_duration_input =  st.slider("Simulation Duration (days)", 1, 30, 10)

    warm_up_duration =  st.slider("Warm-up Duration (hours)", 0, 24*10, 0)
    st.markdown(f"The simulation will not start recording metrics until {(warm_up_duration / 24):.2f} days have elapsed")

    number_of_runs_input = st.slider("Number of Runs", 1, 100, 5)

    sim_start_date_input = st.date_input("Enter the Simulation Start Date", value=datetime.strptime("2024-08-01 07:00:00", '%Y-%m-%d %H:%M:%S'))

    sim_start_time_input = st.time_input("Enter the Simulations Start Time", value=datetime.strptime("2024-08-01 07:00:00", '%Y-%m-%d %H:%M:%S'))



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
                    sim_start_date = datetime.combine(sim_start_date_input, sim_start_time_input)
                )

            results.append(
                run_results
                )

            my_bar.progress((run+1)/number_of_runs_input, text=progress_text)


        results_all_runs = pd.concat(results)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Summary Visualisations", "Animation",
             "Full Event Dataset", "Debugging Visualisations"]
        )

        with tab1:
            t1_col1, t1_col2 = st.columns(2)
            with t1_col1:
                st.subheader("Unmet Demand")
                st.write("Placeholder")

            with t1_col2:
                st.subheader("Helicopter Utilisation")
                st.write("Placeholder")

        with tab2:
            st.write("Placeholder")

        with tab3:
            st.subheader("Observed Event Types")

            st.write(
                pd.DataFrame(
                    results_all_runs[["run_number", "time_type"]].value_counts()).reset_index()
                    .pivot(index="run_number", columns="time_type", values="count")
                    )

            st.subheader("Observed Callsigns")

            st.write(
                pd.DataFrame(
                    results_all_runs[["run_number", "callsign"]].value_counts()).reset_index()
                    .pivot(index="run_number", columns="callsign", values="count")
                    )


            st.subheader("Full Event Log")

            st.write(results_all_runs)

        with tab4:
            st.subheader("Event Overview")

            st.plotly_chart(
                px.scatter(
                    results_all_runs,
                    x="timestamp_dt",
                    y="run_number",
                    facet_row="time_type",
                    color="time_type"),
                    use_container_width=True
            )

            @st.fragment
            def patient_viz():
                st.subheader("Per-patient journey exploration")

                patient_filter = st.selectbox("Select a patient", results_all_runs.index.unique())

                st.plotly_chart(
                    px.scatter(
                        results_all_runs[results_all_runs.index==patient_filter],
                        x="timestamp_dt",
                        y="run_number",
                        color="time_type"),
                        use_container_width=True
                )



            patient_viz()
