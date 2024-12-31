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
from des_parallel_process import runSim, parallelProcessJoblib, collateRunResults
from utils import Utils

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("Devon Air Ambulance Simulation")

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

# Inputs to the model are currently contained within a collapsible sidebar
# We may wish to move these elsewhere when
with st.sidebar:
    st.subheader("Model Inputs")

    st.markdown("#### Simulation Run Settings")

    amb_data = st.toggle("Model ambulance service data", value=False)

    sim_duration_input =  st.slider("Simulation Duration (days)", 1, 30, 7)

    warm_up_duration =  st.slider("Warm-up Duration (hours)", 0, 24*10, 0)
    st.markdown(f"The simulation will not start recording metrics until {(warm_up_duration / 24):.2f} days have elapsed")

    number_of_runs_input = st.slider("Number of Runs", 1, 30, 5)

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
                        sim_start_date = datetime.combine(sim_start_date_input, sim_start_time_input),
                        amb_data=amb_data
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
                        sim_start_date = datetime.combine(sim_start_date_input, sim_start_time_input),
                        amb_data = amb_data
            )
            collateRunResults()
            results_all_runs = pd.read_csv("data/run_results.csv")


        tab_names = [
             "Summary Visualisations",
             "Full Event Dataset",
             "Debugging Visualisations"
             ]

        if create_animation_input:
            tab_names.append("Animation")
            tab1, tab2, tab3, tab4 = st.tabs(
                tab_names
            )
        else:
            tab1, tab2, tab3 = st.tabs(
                tab_names
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

            event_counts_df =  (pd.DataFrame(
                    results_all_runs[["run_number", "time_type"]].value_counts()).reset_index()
                    .pivot(index="run_number", columns="time_type", values="count")
            )
            st.write(
               event_counts_df
                    )

            st.subheader("Observed Callsigns")

            st.write(
                pd.DataFrame(
                    results_all_runs[["run_number", "callsign_group"]].value_counts()).reset_index()
                    .pivot(index="run_number", columns="callsign_group", values="count")
                    )


            st.subheader("Full Event Log")

            st.write(results_all_runs)

        with tab3:
            st.subheader("Event Overview")

            tab3a, tab3b = st.tabs(["By Event", "By Run"])

            with tab3a:
                fig = px.scatter(
                        results_all_runs,
                        x="timestamp_dt",
                        y="run_number",
                        facet_row="time_type",
                        color="time_type",
                        height=800,
                        title="Events Over Time - By Run")
                fig.update_traces(marker=dict(size=3, opacity=0.5))
                st.plotly_chart(
                    fig,
                        use_container_width=True
                    )

            with tab3b:
                st.plotly_chart(
                    px.line(
                        results_all_runs[results_all_runs["time_type"]=="arrival"],
                        x="timestamp_dt",
                        y="P_ID",
                        color="run_number",
                        height=800,
                        title="Cumulative Arrivals Per Run"),
                        use_container_width=True
                    )

            st.subheader("Event Counts")
            st.write("Period: {sim_duration_input} days")

            # st.write(event_counts_df.reset_index(drop=False).melt(id_vars="run_number"))

            event_counts_long = event_counts_df.reset_index(drop=False).melt(id_vars="run_number")

            st.plotly_chart(
                    px.bar(
                        event_counts_long[event_counts_long["time_type"].isin(["arrival", "AMB call start", "HEMS call start"])],
                        x="run_number",
                        y="value",
                        facet_col="time_type",
                        height=600
                )
            )

            hems_events = ["arrival", "HEMS call start", "HEMS allocated to call", "HEMS mobile", "HEMS stood down en route", "HEMS on scene", "HEMS patient treated (not conveyed)", "HEMS leaving scene", "HEMS arrived destination", "HEMS clear"]

            st.plotly_chart(
                    px.funnel(
                        event_counts_long[event_counts_long["time_type"].isin(hems_events)],
                        facet_col="run_number",
                        x="value",
                        y="time_type",
                        category_orders={"time_type": hems_events[::-1]}

                )
            )

            amb_events = ["arrival", "AMB call start", "AMB clear"]

            st.plotly_chart(
                    px.funnel(
                        event_counts_long[event_counts_long["time_type"].isin(amb_events)],
                        facet_col="run_number",
                        x="value",
                        y="time_type",
                        category_orders={"time_type": amb_events[::-1]},

                )
            )


            @st.fragment
            def patient_viz():
                st.subheader("Per-patient journey exploration")

                patient_filter = st.selectbox("Select a patient", results_all_runs.P_ID.unique())

                tab_list =  st.tabs([f"Run {i+1}" for i in range(number_of_runs_input)])

                for idx, tab in enumerate(tab_list):
                    tab.plotly_chart(
                        px.scatter(
                            results_all_runs[
                                (results_all_runs.P_ID==patient_filter) &
                                (results_all_runs.run_number==idx+1)],
                            x="timestamp_dt",
                            y="time_type",
                            color="time_type"),
                            use_container_width=True
                    )

            patient_viz()

        if create_animation_input:
            with tab4:
                event_position_df = pd.DataFrame([
                
                    {'event': 'HEMS call start',
                    'x':  10, 'y': 600, 'label': "HEMS Call Start"},

                    {'event': 'HEMS allocated to call',
                    'x':  180, 'y': 550, 'label': "HEMS Allocated"},

                    {'event': 'HEMS mobile',
                    'x':  300, 'y': 500, 'label': "HEMS Mobile"},

                    {'event': 'HEMS on scene',
                    'x':  400, 'y': 450, 'label': "HEMS On Scene"},

                    {'event': "HEMS stood down en route",
                    'x':  400, 'y': 425, 'label': "HEMS Stood Down"},

                    {'event': 'HEMS leaving scene',
                    'x':  530, 'y': 400, 'label': "HEMS Leaving Scene"},

                    {'event': 'HEMS arrived destination',
                    'x':  700, 'y': 350, 'label': "HEMS Arrived Destination"},

                    {'event': 'HEMS clear',
                    'x':  900, 'y': 300, 'label': "HEMS Clear"},

                    # {'event': 'AMB call start',
                    # 'x':  160, 'y': 100, 'label': "Ambulance Call Start"},

                    # {'event': 'AMB arrival at hospital',
                    # 'x':  360, 'y': 100, 'label': "Ambulance Arrive at Hospital"},

                    # {'event': 'AMB clear',
                    # 'x':  660, 'y': 100, 'label': "Ambulance Clear"},

                    # {'event': 'HEMS to AMB handover',
                    # 'x':  360, 'y': 300, 'label': "HEMS to AMB handover"},

                    ]
                )

                event_log = results_all_runs.reset_index().rename(
                                columns = {"timestamp":"time",
                                "P_ID": "patient",
                                "time_type": "event",
                                "callsign_group": "pathway"}
                                )

                event_log['pathway'] = event_log['pathway'].fillna('Shared')

                #print(event_log.head(50))

                st.plotly_chart(
                    animate_activity_log(
                            event_log = event_log[event_log["run_number"]==1],
                            event_position_df=event_position_df,
                            setup_mode=True,
                            debug_mode=True,
                            every_x_time_units=10,
                            limit_duration=60*24*1,
                            time_display_units="dhm"
                    )
                )
