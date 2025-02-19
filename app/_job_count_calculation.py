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
from calendar import monthrange

# 90th Percentile
def q90(x):
    return x.quantile(0.9)

# 90th Percentile
def q10(x):
    return x.quantile(0.1)

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

def plot_hourly_call_counts(call_df, params_df, box_plot=False, average_per_month=False,
                            bar_colour="teal", title="Calls Per Hour", use_poppins=False,
                            error_bar_colour="charcoal", show_error_bars_bar=True,
                            show_historical=True,
                            historical_data_path="../actual_data/jobs_by_hour.csv"):
    hourly_calls_per_run = call_df.groupby(['hour', 'run_number'])[['P_ID']].count().reset_index().rename(columns={"P_ID": "count"})

    fig = go.Figure()

    if show_historical:
        jobs_per_hour_historic = pd.read_csv(historical_data_path)

        jobs_per_hour_historic['month'] = pd.to_datetime(jobs_per_hour_historic['month'],dayfirst=True)
        jobs_per_hour_historic['year_numeric'] = jobs_per_hour_historic['month'].apply(lambda x: x.year)
        jobs_per_hour_historic['month_numeric'] = jobs_per_hour_historic['month'].apply(lambda x: x.month)
        jobs_per_hour_historic_long = jobs_per_hour_historic.melt(id_vars=['month','month_numeric', 'year_numeric'])
        jobs_per_hour_historic_long["hour"] = jobs_per_hour_historic_long['variable'].str.extract(r"(\d+)\s")
        jobs_per_hour_historic_long["hour"] = jobs_per_hour_historic_long["hour"].astype('int')
        jobs_per_hour_historic_long = jobs_per_hour_historic_long[~jobs_per_hour_historic_long['value'].isna()]

        if not average_per_month:
            jobs_per_hour_historic_long['value'] = jobs_per_hour_historic_long['value'] * (float(_processing_functions.get_param("sim_duration", params_df))/60/24/ 30)

        jobs_per_hour_historic_agg = (
            jobs_per_hour_historic_long.groupby(['hour'])['value']
            .agg(['min','max', q10, q90])
            ).reset_index()

        fig.add_trace(go.Bar(
            x=jobs_per_hour_historic_agg["hour"],
            y=jobs_per_hour_historic_agg["max"] - jobs_per_hour_historic_agg["min"],  # The range
            base=jobs_per_hour_historic_agg["min"],  # Starts from the minimum
            name="Historical Range",
            marker_color="rgba(100, 100, 255, 0.2)",  # Light blue with transparency
            hoverinfo="skip",  # Hide hover info for clarity
            showlegend=True,
            width=1.0,  # Wider bars to make them contiguous
            offsetgroup="historical"  # Grouping ensures alignment
        ))

        fig.add_trace(go.Bar(
            x=jobs_per_hour_historic_agg["hour"],
            y=jobs_per_hour_historic_agg["q90"] - jobs_per_hour_historic_agg["q10"],  # The range
            base=jobs_per_hour_historic_agg["q10"],  # Starts from the minimum
            name="Historical 80% Range",
            marker_color="rgba(100, 100, 255, 0.3)",  # Light blue with transparency
            hoverinfo="skip",  # Hide hover info for clarity
            showlegend=True,
            width=1.0,  # Wider bars to make them contiguous
            offsetgroup="historical"  # Grouping ensures alignment
        ))

        fig.update_layout(
            xaxis=dict(dtick=1),
            barmode="overlay",  # Ensures bars overlay instead of stacking
            title="Comparison of Simulated and Historical Call Counts"
        )

    if box_plot:


        if average_per_month:
            hourly_calls_per_run['average_per_day'] = hourly_calls_per_run['count'] / (float(_processing_functions.get_param("sim_duration", params_df))/60/24 / 30)
            # hourly_calls_per_run['average_per_day'] = hourly_calls_per_run['count'] / 30
            # fig = px.box(hourly_calls_per_run,
            #               x="hour", y="average_per_day",
            #               color_discrete_sequence=[DAA_COLORSCHEME[bar_colour]],
            #               labels={"average_per_day": "Average Monthly Calls Per Hour Across Simulation<br>Averaged Across Simulation Runs",
            #                       "hour": "Hour"},
            #               title=title).update_xaxes(dtick=1)
            y_column = "average_per_day"
            y_label = "Average Monthly Calls Per Hour Across Simulation<br>Averaged Across Simulation Runs"
        else:
            y_column = "count"
            y_label = "Total Calls Per Hour Across Simulation<br>Averaged Across Simulation Runs"


        # Add box plot trace
        fig.add_trace(go.Box(
            x=hourly_calls_per_run["hour"],
            y=hourly_calls_per_run[y_column],
            name="Simulated Mean",
            width=0.4,
            marker=dict(color=DAA_COLORSCHEME[bar_colour]),
            showlegend=True,
            boxpoints="outliers",  # Show all data points
            # jitter=0.3,  # Spread out points slightly for visibility
            # pointpos=-1.8  # Offset points to avoid overlap with the box
        ))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(title="Hour", dtick=1),
            yaxis=dict(title=y_label)
        )
            # fig = px.box(hourly_calls_per_run,
            #               x="hour", y="count",
            #               color_discrete_sequence=[DAA_COLORSCHEME[bar_colour]],
            #               labels={"count": "Total Calls Per Hour Across Simulation<br>Averaged Across Simulation Runs",
            #             "hour": "Hour"},
            #               title=title).update_xaxes(dtick=1)

    else:

        # Create required dataframe for simulation output display
        aggregated_data = hourly_calls_per_run.groupby("hour").agg(
            mean_count=("count", "mean"),
            std_count=("count", "std")
        ).reset_index()

        if show_error_bars_bar:
                error_y = "std_count"
        else:
            error_y=None

        if average_per_month:
            # aggregated_data['mean_count'] = aggregated_data['mean_count'] / (float(_processing_functions.get_param("sim_duration", params_df))/60/24)
            # aggregated_data['std_count'] = aggregated_data['std_count'] / (float(_processing_functions.get_param("sim_duration", params_df))/60/24)

            aggregated_data['mean_count'] = aggregated_data['mean_count'] / (float(_processing_functions.get_param("sim_duration", params_df))/60/24/ 30)
            aggregated_data['std_count'] = aggregated_data['std_count'] / (float(_processing_functions.get_param("sim_duration", params_df))/60/24/ 30)


            fig.add_trace(go.Bar(
                x=aggregated_data["hour"],
                y=aggregated_data["mean_count"],
                name="Simulated Mean",
                marker=dict(color=DAA_COLORSCHEME[bar_colour]),  # Use your color scheme
                error_y=dict(
                    type="data",
                    array=error_y,
                    visible=True
                ) if error_y is not None else None,
                width=0.4,  # Narrower bars in front
                offsetgroup="simulated"
            ))

            fig.update_layout(
                xaxis=dict(dtick=1),
                barmode="overlay",  # Ensures bars overlay instead of stacking
                title=title,
                yaxis_title="Average Monthly Calls Per Hour Across Simulation<br>Averaged Across Simulation Runs",
                xaxis_title="Hour"
            )


        else:
            fig.add_trace(go.Bar(
                x=aggregated_data["hour"],
                y=aggregated_data["mean_count"],
                name="Simulated Mean",
                marker=dict(color=DAA_COLORSCHEME[bar_colour]),  # Use your color scheme
                error_y=dict(
                    type="data",
                    array=error_y,
                    visible=True
                ) if error_y is not None else None,
                width=0.4,  # Narrower bars in front
                offsetgroup="simulated"
            ))

            fig.update_layout(
                xaxis=dict(dtick=1),
                barmode="overlay",  # Ensures bars overlay instead of stacking
                title=title,
                yaxis_title="Total Calls Per Hour Across Simulation",
                xaxis_title="Hour"
            )




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

    summary = (
            call_counts_monthly.groupby("month_start")["monthly_calls"]
            .agg(["mean", "std", "count", "max", "min", q10, q90])
            .reset_index()
            )

    # Create the plot
    fig = px.line(summary, x="month_start", y="mean",
                markers=True,
                labels={"mean": "Average Calls Per Month",
                        "month_start": "Month"},
                title="Number of Monthly Calls Received in Simulation",
                color_discrete_sequence=[DAA_COLORSCHEME["navy"]]
                ).update_traces(line=dict(width=2.5))

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
            showlegend=True, name="80% Range Across Simulation Runs"
        )
    ])


    # Increase upper y limit to be sightly bigger than the max number of calls observed in a month
    # Ensure lower y limit is 0
    fig = fig.update_yaxes({'range': (0, call_counts_monthly["monthly_calls"].max() * 1.1)})

    if show_historical:
        historical_jobs_per_month = pd.read_csv(historical_monthly_job_data_path, parse_dates=False)
        # Convert to datetime
        # (using 'parse_dates=True' in read_csv isn't reliably doing that, so make it explicit here)
        historical_jobs_per_month["Month"] = pd.to_datetime(historical_jobs_per_month['Month'])

        historical_jobs_per_month["Month_Numeric"] = (
            historical_jobs_per_month["Month"].apply(lambda x: x.month)
            )

        historical_jobs_per_month["Year_Numeric"] = (
            historical_jobs_per_month["Month"]
            .apply(lambda x: x.year)
            )

        historical_summary = (
            historical_jobs_per_month
            .groupby('Month_Numeric')['Jobs']
            .agg(["max","min"])
            .reset_index()
            .rename(columns={"max": "historic_max", "min": "historic_min"})
            )

        call_counts_monthly["Month_Numeric"] = (
                call_counts_monthly["month_start"].apply(lambda x: x.month)
                )

        # historical_jobs_per_month["New_Date"] = (
        #     historical_jobs_per_month["Month"]
        #     .apply(lambda x: datetime.date(year=first_month.year,day=1,month=x.month))
        #     )

        if (historical_jobs_per_month["Jobs"].max() * 1.1) > (call_counts_monthly["monthly_calls"].max()*1.1):
            fig = fig.update_yaxes({'range': (0, historical_jobs_per_month["Jobs"].max() * 1.1)})

        if show_historical_individual_years:
            for idx, year  in enumerate(historical_jobs_per_month["Year_Numeric"].unique()):
                # Filter the data for the current year
                year_data = historical_jobs_per_month[historical_jobs_per_month["Year_Numeric"] == year]

                new_df = (
                    call_counts_monthly.drop_duplicates('month_start')
                    .merge(year_data, on="Month_Numeric", how="left")
                    )

                # Add the trace for the current year
                fig.add_trace(go.Scatter(
                    x=new_df["month_start"],
                    y=new_df["Jobs"],
                    mode='lines+markers',
                    opacity=0.7,
                    name=str(year),  # Using the year as the trace name
                    line=dict(
                        color=list(DAA_COLORSCHEME.values())[idx],
                        dash="dash"
                        )  # Default to gray if no specific color found
                ))
        else:
            # Add a filled range showing the entire historical range
            call_counts_monthly = call_counts_monthly.merge(
                historical_summary, on="Month_Numeric",
                how="left"
                )


            # Add a filled range (shaded area) for the historical range
            fig.add_trace(go.Scatter(
                x=call_counts_monthly["month_start"],
                y=call_counts_monthly["historic_max"],
                mode='lines',
                showlegend=False,
                line=dict(color='rgba(0,0,0,0)'),  # Invisible line (just the area)
            ))

            fig.add_trace(go.Scatter(
                x=call_counts_monthly["month_start"],
                y=call_counts_monthly["historic_min"],
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
