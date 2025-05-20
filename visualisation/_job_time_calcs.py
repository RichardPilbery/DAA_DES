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
import _processing_functions
import plotly.graph_objects as go
from scipy.stats import ks_2samp
import streamlit as st
import gc

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

    # Force 'Simulated' to always appear first (left) and 'Historical' second (right)
    historical_activity_times_overall.rename(columns={'value': 'resource_use_duration'}, inplace=True)

    full_activity_duration_df = pd.concat([historical_activity_times_overall, utilisation_model_df])

    full_activity_duration_df['what'] = pd.Categorical(
        full_activity_duration_df['what'],
        categories=['Simulated', 'Historical'],
        ordered=True
    )

    if violin:
        fig = px.violin(full_activity_duration_df, x="vehicle_type", y="resource_use_duration",
                    color="what", category_orders={"what": ["Simulated", "Historical"]})
    else:
        fig = px.box(full_activity_duration_df, x="vehicle_type", y="resource_use_duration",
            color="what", category_orders={"what": ["Simulated", "Historical"]})

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

def plot_total_times_by_hems_or_pt_outcome(run_results, y, color, column_of_interest="hems_result",
                                           show_group_averages=True):
    resource_use_only = run_results[run_results["event_type"].isin(["resource_use", "resource_use_end"])]

    resource_use_wide = (
        resource_use_only[["P_ID", "run_number", "event_type", "timestamp_dt",
                           "callsign_group", "vehicle_type", "callsign", column_of_interest]]
        .pivot(index=["P_ID", "run_number", "callsign_group", "vehicle_type", "callsign", column_of_interest],
               columns="event_type", values="timestamp_dt").reset_index()
               )

        # If utilisation start time is missing, then set to start of model + warm-up time (if relevant)
    # as can assume this is a call that started before the warm-up period elapsed but finished
    # after the warm-up period elapsed
    # TODO: need to add in a check to ensure this only happens for calls at the end of the model,
    # not due to errors elsewhere that could fail to assign a resource end time
    resource_use_wide = _processing_functions.fill_missing_values(
        resource_use_wide, "resource_use",
        _processing_functions.get_param("warm_up_end_date", pd.read_csv("data/run_params_used.csv"))
        )

    # Calculate number of minutes the attending resource was in use on each call
    resource_use_wide["resource_use_duration"] = _processing_functions.calculate_time_difference(
        resource_use_wide, 'resource_use', 'resource_use_end', unit='minutes'
        )

    # Calculate average duration per HEMS result
    mean_durations = (resource_use_wide
                      .groupby(y)["resource_use_duration"]
                      .mean()
                      .sort_values(ascending=True))

    # Create sorted list of HEMS results
    sorted_results = mean_durations.index.tolist()

    fig = px.box(resource_use_wide,
            x="resource_use_duration",
            y=y,
            color=color,
            color_discrete_sequence=list(DAA_COLORSCHEME.values()),
            category_orders={y: sorted_results},
            labels={
                "resource_use_duration": "Resource Use Duration (minutes)",
                "vehicle_type": "Vehicle Type",
                "hems_result": "HEMS Result",
                "outcome": "Patient Outcome",
                "callsign": "Callsign",
                "callsign_group": "Callsign Group"
            },
            height=900)

    if show_group_averages:
        # Add vertical lines for group means
        # Map hems_result to its numerical position on the y-axis
        # Reversed mapping: top of plot gets highest numeric y-position
        result_order = sorted_results
        n = len(result_order)
        y_positions = {result: n - i - 1 for i, result in enumerate(result_order)}

        for result, avg_duration in mean_durations.items():
            y_center = y_positions[result]
            # Plot horizontal line centered at this group
            fig.add_shape(
                type="line",
                x0=avg_duration,
                x1=avg_duration,
                y0=y_center - 0.4,
                y1=y_center + 0.4,
                xref="x",
                yref="y",
                line=dict(color="black", dash="dash"),
            )

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



def plot_time_breakdown(run_results_path="data/run_results.csv",
                        historical_data_path="historical_data/historical_job_durations_breakdown.csv"):

    run_results = pd.read_csv(run_results_path)

    job_times = ['time_allocation', 'time_mobile', 'time_to_scene', 'time_on_scene',
       'time_to_hospital', 'time_to_clear']

    run_results = run_results[run_results["event_type"].isin(job_times)][['P_ID', 'run_number', 'time_type', 'event_type', 'vehicle_type']]
    run_results['time_type'] = run_results['time_type'].astype('float')


    historical_job_duration_breakdown = pd.read_csv(historical_data_path)

    historical_job_duration_breakdown['what'] = 'Historical'
    run_results['what'] = 'Simulated'

    full_job_duration_breakdown_df = pd.concat(
        [run_results.rename(columns={"time_type":"value", "event_type": "name"}).drop(columns=["P_ID", "run_number"]),
        historical_job_duration_breakdown[historical_job_duration_breakdown["name"]!="total_duration"].drop(columns=["callsign", "job_identifier"])
        ])

    full_job_duration_breakdown_df['what'] = pd.Categorical(
        full_job_duration_breakdown_df['what'],
        categories=['Simulated', 'Historical'],
        ordered=True
    )

    full_job_duration_breakdown_df["name"] = full_job_duration_breakdown_df["name"].str.replace("_", " ").str.title()

    fig = px.box(
        full_job_duration_breakdown_df,
        y="value", x="name", color="what",
        facet_row="vehicle_type",
        category_orders={"what": ["Simulated", "Historical"]},
        labels={"value": "Duration (minutes)",
                # "vehicle_type": "Vehicle Type",
                "what": "Time Type (Historical Data vs Simulated Data)",
                "name": "Job Stage"
                },
        title="Comparison of Job Stage Durations by Vehicle Type",
        facet_row_spacing=0.2,
    )

    # Remove default facet titles
    fig.layout.annotations = [
        anno for anno in fig.layout.annotations
        if not anno.text.startswith("vehicle_type=")
    ]

    # Get the sorted unique vehicle types as used by Plotly (from top to bottom)
    # Plotly displays the first facet row (in terms of sorting) at the bottom
    vehicle_types = sorted(full_job_duration_breakdown_df["vehicle_type"].unique())

    n_rows = len(vehicle_types)
    row_heights = [1.0 - (i / n_rows) for i in range(n_rows)]

    for i, vehicle in enumerate(vehicle_types):
        fig.add_annotation(
            text=f"Vehicle Type: {vehicle.capitalize()}",
            xref="paper", yref="paper",
            x=0.5,
            y=row_heights[i] + 0.02,  # slightly above the subplot
            showarrow=False,
            font=dict(size=14, color="black"),
            xanchor="center"
        )

    # Increase spacing and top margin
    fig.update_layout(
        margin=dict(t=120),
        title_y=0.95,
        height=200 + 300 * n_rows,  # Adjust height based on number of rows
    )

    del run_results
    gc.collect()

    return fig
