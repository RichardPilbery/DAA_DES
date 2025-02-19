"""
File containing all calculations and visualisations relating to the number of jobs undertaken
in the simulation.
[ ] Total number of jobs
[ ] Total number of jobs by callsign
[ ] Total number of jobs by vehicle type
[ ] Total number of jobs by callsign group
[ ] Jobs attended of those received (missed jobs)
[ ] Day/night job split
[ ] Jobs across the course of the year
[x] Jobs across the course of the day
[ ] Jobs by day of the week

Covers variation within the simulation, and comparison with real world data.
"""

import pandas as pd
import _processing_functions
import plotly.express as px
from _app_utils import DAA_COLORSCHEME
import plotly.graph_objects as go
import numpy as np
import datetime

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
    """
    Returns a series of the calls per simulation run
    """
    return call_df.groupby('run_number')[['P_ID']].count().reset_index()

def get_AVERAGE_calls_per_run(call_df):
    """
    Returns a count of the calls per run, averaged across all runs
    """
    calls_per_run = get_calls_per_run(call_df)
    return calls_per_run.mean()['P_ID'].round(2)

def get_UNATTENDED_calls_per_run(call_df):
    return call_df[call_df['callsign'].isna()].groupby('run_number')[['P_ID']].count().reset_index()

def get_AVERAGE_UNATTENDED_calls_per_run(call_df):
    unattended_calls_per_run = get_UNATTENDED_calls_per_run(call_df)
    return unattended_calls_per_run.mean()['P_ID'].round(2)

def display_UNTATTENDED_calls_per_run(call_df):
    """
    Alternative to get_perc_unattended_string()

    This approach looks at calls that never got a callsign assigned
    """
    total_calls = get_AVERAGE_calls_per_run(call_df)
    unattended_calls = get_AVERAGE_UNATTENDED_calls_per_run(call_df)

    return f"{unattended_calls:.0f} of {total_calls:.0f} ({(unattended_calls/total_calls):.1%})"

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


def plot_monthly_calls(call_df, show_individual_runs=False, use_poppins=False,
                       show_historical=False,
                       historical_monthly_job_data_path="../actual_data/historical_jobs_per_month.csv",
                       show_historical_individual_years=False):
    call_df['timestamp_dt'] = pd.to_datetime(call_df['timestamp_dt'])
    call_df['month_start'] = call_df['timestamp_dt'].dt.to_period('M').dt.to_timestamp()

    call_counts_monthly = call_df.groupby(
        ['run_number', 'month_start']
        )[['P_ID']].count().reset_index().rename(columns={"P_ID": "monthly_calls"})

    # Identify first and last month in the dataset
    first_month = call_counts_monthly["month_start"].min()
    last_month = call_counts_monthly["month_start"].max()

    # Filter out the first and last month
    call_counts_monthly = (
        call_counts_monthly[
            (call_counts_monthly["month_start"] != first_month) &
            (call_counts_monthly["month_start"] != last_month)
            ]
    )

    # Compute mean of number of patients, standard deviation, and total number of monthly calls
    # 90th Percentile
    def q90(x):
        return x.quantile(0.9)

    # 90th Percentile
    def q10(x):
        return x.quantile(0.1)

    summary = (
            call_counts_monthly.groupby("month_start")["monthly_calls"]
            .agg(["mean", "std", "count", "max", "min", q10, q90])
            .reset_index()
            )

    # summary["ci95_hi"] = summary["mean"] + 1.96 * (summary["std"] / np.sqrt(summary["count"]))
    # summary["ci95_lo"] = summary["mean"] - 1.96 * (summary["std"] / np.sqrt(summary["count"]))

    # Create the plot
    fig = px.line(summary, x="month_start", y="mean",
                markers=True,
                labels={"mean": "Average Calls Per Month",
                        "month_start": "Month"},
                title="Number of Monthly Calls Received in Simulation",
                color_discrete_sequence=[DAA_COLORSCHEME["navy"]]
                )

    if show_individual_runs:
        # Get and reverse the list of runs as plotting in reverse will give a more logical
        # legend at the end
        run_numbers = list(call_counts_monthly["run_number"].unique())
        run_numbers.sort()
        run_numbers.reverse()

        for run in run_numbers:
            run_data = call_counts_monthly[call_counts_monthly["run_number"] == run]
            fig.add_trace(
                go.Scatter(
                    x=run_data["month_start"], y=run_data["monthly_calls"],
                    mode="lines", line=dict(color="gray", width=2, dash='dot'),
                    opacity=0.6, name=f"Simulation Run {run}", showlegend=True,
                )
            )

    # Add full range as a shaded region
    fig.add_traces([
        go.Scatter(
            x=summary["month_start"], y=summary["max"], mode="lines", line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            x=summary["month_start"], y=summary["min"], mode="lines", fill="tonexty",
            line=dict(width=0), fillcolor="rgba(0, 176, 185, 0.15)",
            # fillcolor=DAA_COLORSCHEME['verylightblue'],
            # opacity=0.1,
            showlegend=True, name="Full Range Across Simulation Runs"
        )
    ])

    # Add 10th-90th percentile interval as a shaded region
    fig.add_traces([
        go.Scatter(
            x=summary["month_start"], y=summary["q90"], mode="lines", line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            x=summary["month_start"], y=summary["q10"], mode="lines", fill="tonexty",
            line=dict(width=0), fillcolor="rgba(0, 176, 185, 0.3)",
            # fillcolor=DAA_COLORSCHEME['verylightblue'],
            # opacity=0.1,
            showlegend=True, name="80% Range Across Simulation Runs"
        )
    ])




    fig = fig.update_yaxes({'range': (0, call_counts_monthly["monthly_calls"].max()*1.1)})

    if show_historical:
        historical_jobs_per_month = pd.read_csv(historical_monthly_job_data_path, parse_dates=False)
        historical_jobs_per_month["Month"] = pd.to_datetime(historical_jobs_per_month['Month'])
        historical_jobs_per_month["Month_Numeric"] = historical_jobs_per_month["Month"].apply(lambda x: x.month)
        historical_jobs_per_month["Year_Numeric"] = historical_jobs_per_month["Month"].apply(lambda x: x.year)
        historical_jobs_per_month["New_Date"] = historical_jobs_per_month["Month"].apply(lambda x: datetime.date(year=first_month.year,day=1,month=x.month))

        if (historical_jobs_per_month["Jobs"].max() * 1.1) > (call_counts_monthly["monthly_calls"].max()*1.1):
            fig = fig.update_yaxes({'range': (0, historical_jobs_per_month["Jobs"].max() * 1.1)})

        if show_historical_individual_years:
            for idx, year  in enumerate(historical_jobs_per_month["Year_Numeric"].unique()):
                # Filter the data for the current year
                year_data = historical_jobs_per_month[historical_jobs_per_month["Year_Numeric"] == year]

                # Add the trace for the current year
                fig.add_trace(go.Scatter(
                    x=year_data["New_Date"],
                    y=year_data["Jobs"],
                    mode='lines',
                    name=str(year),  # Using the year as the trace name
                    line=dict(
                        color=list(DAA_COLORSCHEME.values())[idx],
                        dash="dash"
                        )  # Default to gray if no specific color found
                ))
        else:
            # Now, add the filled range showing the entire historical range
            min_jobs = historical_jobs_per_month.groupby('New_Date')['Jobs'].min()  # Minimum Jobs for each date
            max_jobs = historical_jobs_per_month.groupby('New_Date')['Jobs'].max()  # Maximum Jobs for each date

            # Add a filled range (shaded area) for the historical range
            fig.add_trace(go.Scatter(
                x=max_jobs.index,
                y=max_jobs.values,
                mode='lines',
                showlegend=False,
                line=dict(color='rgba(0,0,0,0)'),  # Invisible line (just the area)
            ))

            fig.add_trace(go.Scatter(
                x=min_jobs.index,
                y=min_jobs.values,
                mode='lines',
                name='Historical Range',
                line=dict(color='rgba(0,0,0,0)'),  # Invisible line (just the area)
                fill='tonexty',  # Fill the area between this and the next trace
                fillcolor='rgba(255, 164, 0, 0.15)',  # Semi-transparent fill color
            ))

    # Adjust font to match DAA style
    if use_poppins:
        return fig.update_layout(font=dict(family="Poppins", size=18, color="black"))
    else:
        return fig
