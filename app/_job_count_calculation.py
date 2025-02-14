"""
File containing all calculations and visualisations relating to the number of jobs undertaken
in the simulation.
- Total number of jobs
- Total number of jobs by callsign
- Total number of jobs by vehicle type
- Total number of jobs by callsign group
- Jobs attended of those received (missed jobs)
- Day/night job split
- Jobs across the course of the year
- Jobs across the course of the day
- Jobs by day of the week

Covers variation within the simulation, and comparison with real world data.
"""

import pandas as pd
import _processing_functions
import plotly.express as px
from _app_utils import DAA_COLORSCHEME

def make_job_count_df(path="../data/run_results.csv",
                      params_path="../data/run_params_used.csv"):
    df = pd.read_csv(path)
    params_df = pd.read_csv(params_path)
    n_runs = len(df["run_number"].unique())

    # Add callsign column if not already present in the dataframe passed to the function
    if 'callsign' not in df.columns:
        df = _processing_functions.make_callsign_column(df)

    # hems_result and outcome columns aren't determined until a later step
    # backfill this per patient/run so we'll have access to it from the row for
    # the patient's arrival
    df["hems_result"] = df.groupby(['P_ID', 'run_number']).hems_result.bfill()
    df["outcome"] = df.groupby(['P_ID', 'run_number']).outcome.bfill()

    # TODO - see what we can do about any instances where these columns remain NA
    # Think this is likely to relate to instances where there was no resource available?
    # Would be good to populate these columns with a relevant indicator if that's the case
    call_df = df[df["time_type"] == "arrival"].drop(columns=['time_type', "event_type"])
    return call_df

def get_calls_per_run(call_df):
    return call_df.groupby('run_number')[['P_ID']].count().reset_index()

def get_AVERAGE_calls_per_run(call_df):
    return call_df.groupby('run_number')[['P_ID']].count().reset_index().mean()['P_ID'].round(2)

def plot_hourly_call_counts(call_df, params_df, box_plot=False, average_per_hour=False,
                            bar_colour="teal", title="Calls Per Hour", use_poppins=False,
                            error_bar_colour="charcoal", show_error_bars_bar=True):
    hourly_calls_per_run = call_df.groupby(['hour', 'run_number'])[['P_ID']].count().reset_index().rename(columns={"P_ID": "count"})

    if box_plot:
        if average_per_hour:
            hourly_calls_per_run['average_per_day'] = hourly_calls_per_run['count'] / (float(_processing_functions.get_param("sim_duration", params_df))/60/24)
            fig = px.box(hourly_calls_per_run,
                          x="hour", y="average_per_day",
                          color_discrete_sequence=[DAA_COLORSCHEME[bar_colour]],
                          labels={"average_per_day": "Average Daily Calls Per Hour Across Simulation<br>Averaged Across Simulation Runs",
                                  "hour": "Hour"},
                          title=title).update_xaxes(dtick=1)

        else:
            fig = px.box(hourly_calls_per_run,
                          x="hour", y="count",
                          color_discrete_sequence=[DAA_COLORSCHEME[bar_colour]],
                          labels={"count": "Total Calls Per Hour Across Simulation<br>Averaged Across Simulation Runs",
                        "hour": "Hour"},
                          title=title).update_xaxes(dtick=1)

    else:
        aggregated_data = hourly_calls_per_run.groupby("hour").agg(
            mean_count=("count", "mean"),
            std_count=("count", "std")
        ).reset_index()

        if show_error_bars_bar:
                error_y = "std_count"
        else:
            error_y=None

        if average_per_hour:
            aggregated_data['mean_count'] = aggregated_data['mean_count'] / (float(_processing_functions.get_param("sim_duration", params_df))/60/24)
            aggregated_data['std_count'] = aggregated_data['std_count'] / (float(_processing_functions.get_param("sim_duration", params_df))/60/24)

            fig =  px.bar(
                aggregated_data,
                x="hour",
                y="mean_count",
                color_discrete_sequence=[DAA_COLORSCHEME[bar_colour]],
                error_y=error_y,
                labels={"mean_count": "Average Daily Calls Across Simulation<br>Averaged Across Simulation Runs",
                        "hour": "Hour"},
                title=title
            ).update_xaxes(dtick=1)

        else:
            fig = px.bar(
                aggregated_data,
                x="hour",
                y="mean_count",
                color_discrete_sequence=[DAA_COLORSCHEME[bar_colour]],
                error_y=error_y,
                labels={"mean_count": "Total Calls Per Hour Across Simulation<br>Averaged Across Simulation Runs",
                        "hour": "Hour"},
                title=title
            ).update_xaxes(dtick=1)

    if not box_plot:
        fig = fig.update_traces(error_y_color=DAA_COLORSCHEME[error_bar_colour])

    if use_poppins:
        return fig.update_layout(font=dict(family="Poppins", size=18, color="black"))
    else:
        return fig
