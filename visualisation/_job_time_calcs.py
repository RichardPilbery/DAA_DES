"""
File containing all calculations and visualisations arelating to the length of time
activities in the simulation take.

All of the below, split by vehicle type.
- Mobilisation Time
- Time to scene
- On-scene time
- Journey to hospital time
- Hospital to clear time
- Total job duration

Covers variation within the simulation, and comparison with real world data.
"""

import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from _utilisation_result_calculation import make_utilisation_model_dataframe
import plotly.graph_objects as go
from scipy.stats import ks_2samp
import streamlit as st

from _app_utils import DAA_COLORSCHEME, q10, q90, q25, q75, format_sigfigs

def create_simulation_event_duration_df(
      event_log_path="../data/run_results.csv"
):
    event_log_df = pd.read_csv(event_log_path)

    event_log_df['timestamp_dt'] = pd.to_datetime(event_log_df['timestamp_dt'])

    order_of_events = [
    'HEMS call start',
    # 'No HEMS available',
    'HEMS allocated to call',
    # 'HEMS stand down before mobile',
    'HEMS mobile',
    # 'HEMS stand down en route',
    'HEMS on scene',
    # 'HEMS landed but no patient contact',
    'HEMS leaving scene',
    # 'HEMS patient treated (not conveyed)',
    'HEMS arrived destination',
    'HEMS clear'
    ]


    event_log_df = event_log_df[event_log_df['time_type'].isin(order_of_events)]

    event_log_df.time_type = event_log_df.time_type.astype("category")
    event_log_df.time_type = event_log_df.time_type.cat.set_categories(order_of_events)
    event_log_df = event_log_df.sort_values(["run_number", "P_ID", "time_type"])

    # Calculate time difference within each group
    event_log_df["time_elapsed"] = event_log_df.groupby(["P_ID", "run_number"])['timestamp_dt'].diff()
    event_log_df['time_elapsed_minutes'] = event_log_df['time_elapsed'].apply(lambda x: x.total_seconds() / 60.0 if pd.notna(x) else None)

    return event_log_df

def summarise_event_times(event_log_path="../data/run_results.csv"):

    event_log_df =  create_simulation_event_duration_df(event_log_path)

    # event_durations_sim = (
    #     event_log_df.groupby('time_type')['time_elapsed']
    #     .agg(['mean', 'median', 'max', 'min'])
    #     .round(1)
    #     )

    # event_durations_sim['mean'] = (
    #     event_durations_sim['mean'].apply(lambda x: x.round('s') if pd.notna(x) else None)
    #     )

    # event_durations_sim['median'] = (
    #     event_durations_sim['median'].apply(lambda x: x.round('s') if pd.notna(x) else None)
    #     )

    event_durations_sim = (
        event_log_df.groupby('time_type')['time_elapsed_minutes']
        .agg(['mean', 'median', 'max', 'min'])
        .round(1)
        )

    return event_durations_sim

def get_historical_times_summary(
        historical_duration_path='../historical_data/historical_median_time_of_activities_by_month_and_resource_type.csv'
        ):
    historical_activity_times = pd.read_csv(
        historical_duration_path,
        parse_dates=False
        )

    # Parse month manually as more controllable
    historical_activity_times['month'] = pd.to_datetime(
        historical_activity_times['month'], format="%Y-%m-%d"
        )

    return historical_activity_times

def get_historical_times_breakdown(
        historical_duration_path='../historical_data/historical_job_durations_breakdown.csv'
        ):
    historical_activity_times = pd.read_csv(
        historical_duration_path
        )

    return historical_activity_times

def get_total_times_model(get_summary=False,
                          path="../data/run_results.csv",
                          params_path="../data/run_params_used.csv",
                          rota_path="../actual_data/HEMS_ROTA.csv",
                          service_path="../data/service_dates.csv",
                          callsign_path="../actual_data/callsign_registration_lookup.csv"
                          ):

    utilisation_model_df = make_utilisation_model_dataframe(
        path=path,
        params_path=params_path,
        rota_path=rota_path,
        service_path=service_path,
        callsign_path=callsign_path
    )[0]

    if get_summary:
        utilisation_by_vehicle_summary = (
            utilisation_model_df.groupby('vehicle_type')['resource_use_duration']
            .agg(['mean', 'median', 'min', 'max', q10, q90]
                ).round(1)
        )
        return utilisation_by_vehicle_summary

    else:

        return utilisation_model_df

def plot_historical_job_duration_vs_simulation_overall(
        historical_activity_times,
        utilisation_model_df,
        use_poppins=True,
        write_to_html=False,
        html_output_filepath="fig_job_durations_historical.html",
        violin=False

):

    fig = go.Figure()

    historical_activity_times_overall = historical_activity_times[historical_activity_times["name"]=="total_duration"]

    historical_activity_times_overall['what'] = 'Historical'
    utilisation_model_df['what'] = 'Simulated'
    historical_activity_times_overall.rename(columns={'value': 'resource_use_duration'}, inplace=True)

    full_activity_duration_df = pd.concat([historical_activity_times_overall, utilisation_model_df])

    if violin:
        fig = px.violin(full_activity_duration_df, x="vehicle_type", y="resource_use_duration",
                    color="what")
    else:
        fig = px.box(full_activity_duration_df, x="vehicle_type", y="resource_use_duration",
            color="what")

    fig.update_layout(title="Resource Utilisation Duration vs Historical Averages")

    if write_to_html:
        fig.write_html(html_output_filepath, full_html=False, include_plotlyjs='cdn')

    # Adjust font to match DAA style
    if use_poppins:
        fig.update_layout(font=dict(family="Poppins", size=18, color="black"))

    return fig

# def plot_activity_time_breakdowns(historical_activity_times,
#                                   event_log_df,
#                                   title,
#                                   vehicle_type="helicopter",
#                                   use_poppins=True):

#     single_vehicle_type_df = event_log_df[event_log_df["vehicle_type"]==vehicle_type]

#     historical_single_vehicle_type = historical_activity_times[historical_activity_times["vehicle_type"]==vehicle_type]

#     single_vehicle_type_df['what'] = 'Simulated'
#     historical_single_vehicle_type['what'] = 'Historical'

#     fig = go.Figure()

#     # Add **historical** duration figures for all event types
#     fig.add_trace(
#         go.Box(
#             y=historical_single_vehicle_type['resource_use_duration'],
#             x=historical_single_vehicle_type['name'],
#             name="Simulated Data"
#         )
#     )

#     # Add **simulated** duration figures for all event types
#     fig.add_trace(
#         go.Box(
#             x=single_vehicle_type_df['time_type'],
#             y=single_vehicle_type_df['time_elapsed_minutes'],
#             name="Simulated Range"
#         )
#     )

#     fig.update_layout(title=title)

#     order_of_events = [
#         'HEMS call start',
#         # 'No HEMS available',
#         'HEMS allocated to call',
#         # 'HEMS stand down before mobile',
#         'HEMS mobile',
#         # 'HEMS stand down en route',
#         'HEMS on scene',
#         # 'HEMS landed but no patient contact',
#         'HEMS leaving scene',
#         # 'HEMS patient treated (not conveyed)',
#         'HEMS arrived destination',
#         'HEMS clear'
#         ]

#     fig.update_xaxes(
#         categoryorder='array',
#         categoryarray=order_of_events
#         )

#     # Adjust font to match DAA style
#     if use_poppins:
#         fig.update_layout(font=dict(family="Poppins", size=18, color="black"))

#     return fig

def plot_total_times(utilisation_model_df, by_run=False):

    if not by_run:
        fig = px.box(utilisation_model_df,
            x="resource_use_duration",
            y="vehicle_type",
            color_discrete_sequence=list(DAA_COLORSCHEME.values()),
            labels={
                "resource_use_duration": "Resource Use Duration (minutes)",
                "vehicle_type": "Vehicle Type",
            })
    else:

        fig = px.box(utilisation_model_df,
            x="resource_use_duration",
            y="vehicle_type",
            color="run_number",
            color_discrete_sequence=list(DAA_COLORSCHEME.values()),
            labels={
                "resource_use_duration": "Resource Use Duration (minutes)",
                "vehicle_type": "Vehicle Type",
                "run_number": "Run Number"
            })

    return fig



def calculate_ks_for_job_durations(historical_data_series, simulated_data_series,
                                   what="cars"):

    statistic, p_value = ks_2samp(
        historical_data_series,
        simulated_data_series
        )

    if p_value > 0.05:
        st.success(f"""There is no statistically significant difference between
                    the distributions of overall job durations for **{what}** in historical data and the
                    simulation (p = {format_sigfigs(p_value)})

                    This means that the pattern of total job durations produced by the simulation
                    matches the pattern seen in the real-world data —
                    for example, the average duration and variability of overall job durations
                    is sufficiently similar to what has been observed historically.
                    """)
    else:
        if p_value < 0.0001:
            p_value_formatted = "< 0.0001"
        else:
            p_value_formatted = format_sigfigs(p_value)

        ks_text_string_sig = f"""
There is a statistically significant difference between the
distributions of overall job durations from historical data and the
simulation (p = {p_value_formatted}) for **{what}**.

This means that the pattern of total job durations produced by the simulation
does not match the pattern seen in the real-world data —
for example, the average duration or variability of overall job durations
may be different.

The simulation may need to be adjusted to better
reflect the patterns of job durations observed historically.

"""

        if statistic < 0.1:
            st.info(ks_text_string_sig + f"""Although the difference is
                    statistically significant, the actual magnitude
                    of the difference (D = {format_sigfigs(statistic, 3)}) is small.
                    This suggests the simulation's total job duration pattern is reasonably
                    close to reality.
                    """)

        elif statistic < 0.2:
            st.warning(ks_text_string_sig + f"""The KS statistic (D = {format_sigfigs(statistic, 3)})
                    indicates a moderate difference in
                    distribution. You may want to review the simulation model to
                    ensure it adequately reflects real-world variability.
                    """)

        else:
            st.error(ks_text_string_sig + f"""The KS statistic (D = {format_sigfigs(statistic, 3)})
                suggests a large difference in overall job duration patterns.
                The simulation may not accurately reflect historical
                patterns and may need adjustment.
                """)
