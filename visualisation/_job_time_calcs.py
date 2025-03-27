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

from _app_utils import DAA_COLORSCHEME, q10, q90, q25, q75

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



def get_historical_times(
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

def get_total_times_model(get_summary=False,
                          path="../data/run_results.csv",
                          params_path="../data/run_params_used.csv",
                          rota_path="../actual_data/HEMS_ROTA.csv",
                          service_path="../data/service_dates.csv"
                          ):

    utilisation_model_df = make_utilisation_model_dataframe(
        path=path,
        params_path=params_path,
        rota_path=rota_path,
        service_path=service_path
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


def plot_historical_utilisation_vs_simulation_overall(
        historical_activity_times,
        utilisation_model_df,
        use_poppins=True
):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Car"],
            y=[q75(historical_activity_times['median_car_total_job_time']) - q25(historical_activity_times['median_car_total_job_time'])],  # Height of the box
            base=[q25(historical_activity_times['median_car_total_job_time'])],  # Start from q10
            marker=dict(color="rgba(0, 176, 185, 0.3)"),
            showlegend=True,
            name="Usual Historical Range - Car"
        )
    )

    fig.add_trace(
        go.Bar(
            x=["Helicopter"],
            y=[q75(historical_activity_times['median_helicopter_total_job_time']) - q25(historical_activity_times['median_helicopter_total_job_time'])],  # Height of the box
            base=[q25(historical_activity_times['median_helicopter_total_job_time'])],  # Start from q10
            marker=dict(color="rgba(0, 176, 185, 0.3)"),
            showlegend=True,
            name="Usual Historical Range - Helicoper"
        )
    )


    fig.add_trace(
        go.Box(
            y=utilisation_model_df['resource_use_duration'],
            x=utilisation_model_df['vehicle_type'].str.title(),
            name="Simulated Range"
        )
    )

    fig.update_layout(title="Resource Utilisation Duration vs Historical Averages")

    # Adjust font to match DAA style
    if use_poppins:
        fig.update_layout(font=dict(family="Poppins", size=18, color="black"))

    return fig


def plot_activity_time_breakdowns(historical_activity_times,
                                  event_log_df,
                                  title,
                                  vehicle_type="helicopter",
                                  use_poppins=True):

    single_vehicle_type_df = event_log_df[event_log_df["vehicle_type"]==vehicle_type]

    historical_single_vehicle_type = historical_activity_times.set_index("month").filter(like=vehicle_type).reset_index()

    str_map = {
        f"median_{vehicle_type}_time_allocation": "HEMS allocated to call",
        f"median_{vehicle_type}_time_mobile": "HEMS mobile",
        f'median_{vehicle_type}_time_on_scene': "HEMS on scene",
        f'median_{vehicle_type}_time_to_clear': "HEMS clear",
        f'median_{vehicle_type}_time_to_hospital': "HEMS leaving scene",
        f'median_{vehicle_type}_time_to_scene': "HEMS arrived destination",
        f'median_{vehicle_type}_total_job_time': "HEMS total job time",
    }

    historical_single_vehicle_type_long = historical_single_vehicle_type.melt(id_vars="month")

    historical_single_vehicle_type_long['Event'] = historical_single_vehicle_type_long['variable'].apply(
        lambda x: str_map[x]
        )

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            x=single_vehicle_type_df['time_type'],
            y=single_vehicle_type_df['time_elapsed_minutes'],
            name="Simulated Range"
        )
    )

    for idx, event_type in enumerate(single_vehicle_type_df['time_type'].unique()):

        historical_df = historical_single_vehicle_type_long[historical_single_vehicle_type_long["Event"]==event_type]

        fig.add_trace(
            go.Bar(
                x=[event_type],
                y=[q90(historical_df['value']) - q10(historical_df['value'])],  # Height of the box
                base=[q10(historical_df['value'])],  # Start from q10
                marker=dict(color="rgba(0, 176, 185, 0.3)"),
                showlegend=True if idx==0 else False,
                name="Usual Historical Range"
            )
        )


    fig.update_layout(title=title)

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

    fig.update_xaxes(
        categoryorder='array',
        categoryarray=order_of_events
        )

    # Adjust font to match DAA style
    if use_poppins:
        fig.update_layout(font=dict(family="Poppins", size=18, color="black"))

    return fig

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
