import streamlit as st
import platform

# Data processing imports
import pandas as pd
import numpy as np
from datetime import datetime
import re

# Plotting
import plotly.express as px
from vidigi.animation import animate_activity_log
from vidigi.prep import reshape_for_animations

# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Simulation imports
from des_parallel_process import runSim, parallelProcessJoblib
from utils import Utils

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

col1, col2 = st.columns([0.7, 0.3])

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

with col1:
    st.title("Devon Air Ambulance Simulation")



# Inputs to the model are currently contained within a collapsible sidebar
# We may wish to move these elsewhere when
with st.sidebar:
    st.subheader("Model Inputs")

    st.markdown("#### Simulation Run Settings")
    sim_duration_input =  st.slider("Simulation Duration (days)", 1, 30, 10)

    warm_up_duration =  st.slider("Warm-up Duration (hours)", 0, 24*10, 0)
    st.markdown(f"The simulation will not start recording metrics until {(warm_up_duration / 24):.2f} days have elapsed")

    number_of_runs_input = st.slider("Number of Runs", 1, 100, 5)

    sim_start_date_input = st.date_input(
        "Enter the Simulation Start Date",
        value=datetime.strptime("2024-08-01 07:00:00", '%Y-%m-%d %H:%M:%S')
        )

    sim_start_time_input = st.time_input(
        "Enter the Simulations Start Time",
        value=datetime.strptime("2024-08-01 07:00:00", '%Y-%m-%d %H:%M:%S')
        )

    create_animation_input = st.toggle("Create Animation", value=False)

button_run_pressed = st.button("Run simulation")

if button_run_pressed:
    progress_text = "Simulation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    with st.spinner('Simulating the system...'):

        # If running on community cloud, parallelisation will not work
        # so run instead using the runSim function sequentially
        if platform.processor() == '':
            print("Running sequentially")
            results = []

            for run in range(number_of_runs_input):
                run_results = runSim(
                        run = run,
                        total_runs = number_of_runs_input,
                        sim_duration = sim_duration_input * 24 * 60,
                        warm_up_time = warm_up_duration * 60,
                        sim_start_date = datetime.combine(sim_start_date_input, sim_start_time_input)
                    )

                results.append(
                    run_results
                    )

                my_bar.progress((run+1)/number_of_runs_input, text=progress_text)

            # Turn into a single dataframe when all runs complete
            results_all_runs = pd.concat(results)

        # If running locally, use parallel processing function to speed up execution significantly
        else:
            print("Running in parallel")
            parallelProcessJoblib(
                        total_runs = number_of_runs_input,
                        sim_duration = sim_duration_input * 24 * 60,
                        warm_up_time = warm_up_duration * 60,
                        sim_start_date = datetime.combine(sim_start_date_input, sim_start_time_input)
            )

            results_all_runs = pd.read_csv("data/run_results.csv")


        tab1, tab2, tab3, tab4 = st.tabs(
            [
             "Summary Visualisations",
             "Full Event Dataset",
             "Debugging Visualisations",
             "Animation"
             ]
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

        with tab3:
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


                tab_list =  st.tabs([f"Run {i+1}" for i in range(number_of_runs_input)])

                for idx, tab in enumerate(tab_list):
                    st.plotly_chart(
                        px.scatter(
                            results_all_runs[
                                (results_all_runs.index==patient_filter) &
                                (results_all_runs.run_number==idx)],
                            x="timestamp_dt",
                            y="time_type",
                            color="time_type"),
                            use_container_width=True
                    )

            patient_viz()

        with tab4:
            if create_animation_input:
                event_position_df = pd.DataFrame([

                    {'event': 'AMB call start',
                    'x':  160, 'y': 100, 'label': "Ambulance Call Start"},

                    {'event': 'AMB arrival at hospital',
                    'x':  360, 'y': 100, 'label': "Ambulance Arrive at Hospital"},

                    {'event': 'AMB clear',
                    'x':  660, 'y': 100, 'label': "Ambulance Clear"},

                    {'event': 'HEMS call start',
                    'x':  160, 'y': 600, 'label': "HEMS Call Start"},

                    {'event': 'HEMS arrival at hospital',
                    'x':  360, 'y': 600, 'label': "HEMS Arrive at Hospital"},

                    {'event': 'HEMS to AMB handover',
                    'x':  360, 'y': 300, 'label': "HEMS to AMB handover"},

                    {'event': 'HEMS clear',
                    'x':  660, 'y': 600, 'label': "HEMS Clear"},

                    ]
                )

                event_log = results_all_runs.reset_index().rename(
                                columns = {"timestamp":"time",
                                "P_ID": "patient",
                                "time_type": "event",
                                "callsign": "pathway"}
                                )

                event_log['pathway'] = event_log['pathway'].fillna('Shared')

                st.plotly_chart(
                    animate_activity_log(
                            event_log = event_log[event_log["run_number"]==1],
                            event_position_df=event_position_df,
                            setup_mode=True,
                            debug_mode=True,
                            every_x_time_units=1,
                            limit_duration=60*24*1,
                            time_display_units="dhm"
                    )
                )
            else:
                st.write("Animation turned off - toggle on in sidebar and rerun model if required")
