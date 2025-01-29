import streamlit as st
import platform

# Data processing imports
import pandas as pd
import numpy as np
from datetime import datetime
import re

# Plotting
import plotly.express as px
from vidigi.animation import animate_activity_log, generate_animation
from vidigi.prep import reshape_for_animations, generate_animation_df

# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Simulation imports
from des_parallel_process import runSim, parallelProcessJoblib, collateRunResults
from utils import Utils

from _state_control import setup_state
from _app_utils import iconMetricContainer, file_download_confirm

from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

setup_state()

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("Run a Simulation")

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

with st.sidebar:
    with stylable_container(css_styles="""
hr {
    border-color: #a6093d;
    background-color: #a6093d;
    color: #a6093d;
    height: 1px;
  }
""", key="hr"):
        st.divider()
    if 'number_of_runs_input' in st.session_state:
        st.subheader("Model Input Summary")

        with stylable_container(
            css_styles="""
                    button {
                            background-color: green;
                            color: white;
                        }
                        """,
            key="green_buttons"
            ):
            if st.button("Want to change the parameters? Click here to go to the parameter page", type="primary"):
                st.switch_page("setup.py")

        st.write(f"Number of Helicopters: {st.session_state.num_helicopters}")
        st.write(f"Number of **Extra** (non-backup) Cars: {st.session_state.num_cars}")

        if st.session_state.demand_adjust_type == "Overall Demand Adjustment":
            if st.session_state.overall_demand_mult == 100:
                st.write(f"Demand is based on historically observed demand with no adjustments")
            elif st.session_state.overall_demand_mult < 100:
                st.write(f"Modelled demand is {100-st.session_state.overall_demand_mult}% less than historically observed demand")
            elif st.session_state.overall_demand_mult > 100:
                st.write(f"Modelled demand is {st.session_state.overall_demand_mult-100}% more than historically observed demand")

        # TODO: Add this in if we decide seasonal demand adjustment is a thing that's wanted
        elif st.session_state.demand_adjust_type == "Per Season Demand Adjustment":
            pass

        elif st.session_state.demand_adjust_type == "Per AMPDS Code Demand Adjustment":
            pass

        else:
            st.error("TELL A DEVELOPER: Check Conditional Code for demand modifier in model.py")


        with stylable_container(css_styles="""
hr {
    border-color: #a6093d;
    background-color: #a6093d;
    color: #a6093d;
    height: 1px;
  }
""", key="hr"):
            st.divider()

        st.write(f"The model will run {st.session_state.number_of_runs_input} replications of {st.session_state.sim_duration_input} days, starting from {st.session_state.sim_start_date_input}")

        if st.session_state.create_animation_input:
            st.write("An animated output will be created.")
            st.info("Turn off this option if the model is running very slowly!")
        else:
            st.write("No animated output will be created.")

        if st.session_state.amb_data:
            st.write("SWAST Ambulance Activity will be modelled")
        else:
            st.write("SWAST Ambulance Activity will not be modelled")

if not st.session_state["visited_setup_page"]:
    st.warning("You haven't set up any parameters - default parameters will be used!")

    if st.button("Click here to go to the parameter page, or continue to use the default model parameters",
                 type="primary"):
            st.switch_page("setup.py")

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

            for run in range(st.session_state.number_of_runs_input):
                run_results = runSim(
                        run = run,
                        total_runs = st.session_state.number_of_runs_input,
                        sim_duration = float(st.session_state.sim_duration_input * 24 * 60),
                        warm_up_time = float(st.session_state.warm_up_duration * 60),
                        sim_start_date = datetime.combine(
                            datetime.strptime(st.session_state.sim_start_date_input, '%Y-%m-%d').date(),
                            datetime.strptime(st.session_state.sim_start_time_input, '%H:%M').time(),
                            ),
                        amb_data=st.session_state.amb_data
                    )

                results.append(
                    run_results
                    )

                my_bar.progress((run+1)/st.session_state.number_of_runs_input, text=progress_text)

            # Turn into a single dataframe when all runs complete
            results_all_runs = pd.concat(results)

        # If running locally, use parallel processing function to speed up execution significantly
        else:
            print("Running in parallel")
            parallelProcessJoblib(
                        total_runs = st.session_state.number_of_runs_input,
                        sim_duration = float(st.session_state.sim_duration_input * 24 * 60),
                        warm_up_time = float(st.session_state.warm_up_duration * 60),
                        sim_start_date = datetime.combine(
                            datetime.strptime(st.session_state.sim_start_date_input, '%Y-%m-%d').date(),
                            datetime.strptime(st.session_state.sim_start_time_input, '%H:%M').time(),
                            ),
                        amb_data = st.session_state.amb_data
            )
            collateRunResults()
            results_all_runs = pd.read_csv("data/run_results.csv")


        tab_names = [
            "Simulation Results Summary",
            "Visualisations",
            "Debugging Visualisations",
            ]

        if st.session_state.create_animation_input:
            tab_names.append("Animation")
            tab1, tab2, tab3, tab4 = st.tabs(
                tab_names
            )
        else:
            tab1, tab2, tab3 = st.tabs(
                tab_names
            )

        with tab1:
            @st.fragment
            def download_button_quarto():
                # st.download_button(
                st.button(
                    "Click here to download these results as a file",
                    on_click=file_download_confirm,
                    icon=":material/download:"
                    )

            download_button_quarto()

            st.info(f"All Metrics are averaged across {st.session_state.number_of_runs_input} simulation runs")

            t1_col1, t1_col2 = st.columns(2)

            with t1_col1:
                with iconMetricContainer(key="nonattend_metric", icon_unicode="e61f", family="outline"):
                    st.metric("Number of Calls DAAT Resource Couldn't Attend",
                            "47 of 1203 (3.9%)",
                            border=True)
                    st.caption("""
These are the 'missed' calls where no DAAT resource was available.
This could be due to
- no resource being on shift
- all resources being tasked to other jobs at the time of the call
""")

            with t1_col2:
                with iconMetricContainer(key="helo_util", icon_unicode="f60c", type="symbols"):
                    st.metric("Overall Helicopter Utilisation",
                            "78%",
                            border=True)

                    st.caption("""
This is how much of the available time (where a helicopter is on shift and able to fly) the helicopter
was in use for.

Time where the helicopter was unable to fly due to weather conditions is not counted as available time here.
For reference, the helicopter was unable to fly for 5.3% of on-shift hours on average (range 3.4% to 7.6%)
                """)


            t1_col3, t1_col4 = st.columns(2)

            with t1_col3:
                with iconMetricContainer(key="preferred_response_metric", icon_unicode="e838", family="outline"):
                    st.metric("Preferred Resource Allocated",
                            "907 of 1203 (75.4%)",
                            border=True)
                    st.caption("""
This is the percentage of time where the 'preferred' resource was available at the time of the call
for response.
""")


        with tab2:
            tab_2_1, tab_2_2 = st.tabs(["Summary Graphs", "Per-Run Breakdowns"])
            with tab_2_1:
                st.header("Summary Graphs")

                tab_2_1_col_1, tab_2_1_col_2 = st.columns(2)

                with tab_2_1_col_1:
                    st.subheader("Utilisation by Callsign")

                    heli_util_df_dummy = pd.DataFrame([
                        {"Vehicle": "H70/CC70",
                        "Average Utilisation": 89},
                        {"Vehicle": "H71/CC71",
                        "Average Utilisation": 61},
                        {"Vehicle": "CC72",
                         "Average Utilisation": 71}
                        ]
                    )

                    util_fig_simple = px.bar(heli_util_df_dummy,
                            x="Average Utilisation",
                            y="Vehicle",
                            orientation="h",
                            height=300
                            ).update_xaxes(ticksuffix = "%", range=[0, 105])

                    # Add optimum range
                    util_fig_simple.add_vrect(x0=65, x1=85,
                                            fillcolor="#5DFDA0", opacity=0.25,  line_width=0)
                    # Add extreme range (above)
                    util_fig_simple.add_vrect(x0=85, x1=100,
                                            fillcolor="#D45E5E", opacity=0.25, line_width=0)
                    # Add suboptimum range (below)
                    util_fig_simple.add_vrect(x0=40, x1=65,
                                            fillcolor="#FDD049", opacity=0.25, line_width=0)
                    # Add extreme range (below)
                    util_fig_simple.add_vrect(x0=0, x1=40,
                                            fillcolor="#D45E5E", opacity=0.25, line_width=0)

                    st.plotly_chart(
                        util_fig_simple
                    )

                with tab_2_1_col_2:
                    st.subheader("Utilisation Split")

                    # heli_util_df_dummy = pd.DataFrame([
                    #     {"Helicopter": "H70",
                    #     "Average Utilisation": 89},
                    #     {"Helicopter": "H71",
                    #     "Average Utilisation": 61},
                    #     ]
                    # )

                    # st.plotly_chart(
                    #     px.bar(heli_util_df_dummy,
                    #         x="Average Utilisation",
                    #         y="Helicopter",
                    #         orientation="h",
                    #         height=300
                    #         ).update_layout(xaxis=dict(ticksuffix = "%", ))
                    # )

            with tab_2_2:
                st.header("Per-run Breakdowns")
                st.write("Placeholder")

        with tab3:
            tab_3_1, tab_3_2, tab_3_3, tab_3_4, tab_3_5, tab_3_6 = st.tabs([
                "Comparisons with Real-World Data", "Counts", "Logs", "Debug Events", "Debug Resources", "Test Results"
                ])
            with tab_3_1:
                st.write("Placeholder")

            with tab_3_2:
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

            with tab_3_3:
                st.subheader("Full Event Log")

                st.write(results_all_runs)

            with tab_3_4:
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
                st.write(f"Period: {st.session_state.sim_duration_input} days")

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

                    tab_list =  st.tabs([f"Run {i+1}" for i in range(st.session_state.number_of_runs_input)])

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

            with tab_3_5:
                st.subheader("Resource Use")

                resource_use_events_only = results_all_runs[results_all_runs["event_type"].str.contains("resource_use")]
                with st.expander("Click here to see the timings of resource use"):
                    st.dataframe(resource_use_events_only)
                    st.dataframe(resource_use_events_only[["P_ID", "time_type", "timestamp_dt", "event_type"]].melt(id_vars=["P_ID", "time_type", "event_type"], value_vars="timestamp_dt"))

                st.plotly_chart(
                    px.scatter(
                    resource_use_events_only[["P_ID", "time_type", "timestamp_dt", "event_type"]].melt(id_vars=["P_ID", "time_type", "event_type"], value_vars="timestamp_dt"),
                    x="value",
                    y="time_type",
                    color="event_type"
                    )
                )

            with tab_3_6:
                st.write("Placeholder")


        if st.session_state.create_animation_input:
            with tab4:

                st.error("Warning - this is not yet working as intended")
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
                                # "time_type": "event",
                                "callsign_group": "pathway"}
                                )

                event_log['pathway'] = event_log['pathway'].fillna('Shared')
                event_log['resource_id'] = 1

                #print(event_log.head(50))
                event_log['callsign'] = event_log['vehicle_type'].str[0].str.upper() + event_log['pathway'].astype(str)

                event_log['event'] = event_log.apply(lambda row:
                    row['time_type'] if row['time_type'] in ['arrival', 'depart'] else row['callsign'],
                    axis=1)


                with st.expander("See final event log"):
                    st.dataframe(event_log[event_log["run_number"]==1])

                event_position_df = pd.DataFrame([
                    {'event': 'arrival',
                    'x':  50, 'y': 400,
                    'label': "Arrival" },

                    {'event': 'H70',
                    'x':  150, 'y': 275,
                    'resource':'n_H70',
                    'label': "H70 Attending"},

                    {'event': 'CC70',
                    'x':  150, 'y': 175,
                    'resource':'n_CC70',
                    'label': "CC70 Attending"},

                {'event': 'H71',
                    'x':  325, 'y': 275,
                    'resource':'n_H71',
                    'label': "H71 Attending"},

                    {'event': 'CC71',
                    'x':  325, 'y': 175,
                    'resource':'n_CC71',
                    'label': "CC71 Attending"},

                    {'event': 'CC72',
                    'x':  475, 'y': 175,
                    'resource':'n_CC72',
                    'label': "CC72 Attending"},

                    {'event': 'exit',
                    'x':  270, 'y': 70,
                    'label': "Exit"}

                ])

                class g():
                    n_H70 = 1
                    n_CC70 = 1
                    n_H71 = 1
                    n_CC71 = 1
                    n_CC72 = 1

                full_patient_df = reshape_for_animations(
                    event_log[event_log["run_number"]==1],
                    every_x_time_units=5,
                    limit_duration=60*24*sim_duration_input,
                    debug_mode=True
                )

                with st.expander("See step 1 animation dataframe"):
                    st.dataframe(full_patient_df)

                full_patient_df_with_position = generate_animation_df(full_patient_df=full_patient_df, event_position_df=event_position_df)

                with st.expander("See step 2 animation dataframe"):
                    st.dataframe(full_patient_df_with_position)

                st.plotly_chart(
                    generate_animation(
                        full_patient_df_plus_pos=full_patient_df_with_position,
                        event_position_df=event_position_df,
                        scenario=g(),
                        plotly_height=750,
                        # start_date=datetime.combine(sim_start_date_input, sim_start_time_input),
                        # time_display_units="dhm"

                                    )
                )

                # st.plotly_chart(
                #     animate_activity_log(
                #             event_log = event_log[event_log["run_number"]==1],
                #             event_position_df=event_position_df,
                #             scenario=g(),
                #             setup_mode=True,
                #             debug_mode=True,
                #             every_x_time_units=10,
                #             display_stage_labels=True,
                #             limit_duration=60*24*sim_duration_input,
                #             time_display_units="dhm"
                #     )
                # )
