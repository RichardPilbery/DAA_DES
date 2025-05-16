"""
File containing all calculations and visualisations relating to the number of jobs undertaken
in the simulation.
[x] Jobs across the course of the year
[x] Jobs across the course of the day
[x] Jobs by day of the week

[ ] Total number of jobs
[x] Total number of jobs by callsign
[ ] Total number of jobs by vehicle type
[ ] Total number of jobs by callsign group
[x] Jobs attended of those received (missed jobs)

Covers variation within the simulation, and comparison with real world data.
"""

import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Ensure this folder is in sys.path
import _processing_functions
import plotly.express as px

import plotly.graph_objects as go
import numpy as np
import datetime
from calendar import monthrange, day_name
import itertools

from _app_utils import DAA_COLORSCHEME, q10, q90

def make_job_count_df(path="../data/run_results.csv",
                      params_path="../data/run_params_used.csv"):

    """
    Given the event log produced by running the model, create a dataframe with one row per
    patient, but all pertinent information about each call added to that row if it would not
    usually be present until a later entry in the log
    """

    df = pd.read_csv(path)

    # Add callsign column if not already present in the dataframe passed to the function
    if 'callsign' not in df.columns:
        df = _processing_functions.make_callsign_column(df)

    # hems_result and outcome columns aren't determined until a later step
    # backfill this per patient/run so we'll have access to it from the row for
    # the patient's arrival
    df["hems_result"] = df.groupby(['P_ID', 'run_number']).hems_result.bfill()
    df["outcome"] = df.groupby(['P_ID', 'run_number']).outcome.bfill()
    # same for various things around allocated resource
    df["vehicle_type"] = df.groupby(['P_ID', 'run_number']).vehicle_type.bfill()
    df["callsign"] = df.groupby(['P_ID', 'run_number']).callsign.bfill()
    df["registration"] = df.groupby(['P_ID', 'run_number']).registration.bfill()

    # TODO - see what we can do about any instances where these columns remain NA
    # Think this is likely to relate to instances where there was no resource available?
    # Would be good to populate these columns with a relevant indicator if that's the case

    # Reduce down to just the 'arrival' row for each patient, giving us one row per patient
    # per run
    call_df = df[df["time_type"] == "arrival"].drop(columns=['time_type', "event_type"])
    call_df.to_csv("data/call_df.csv", index=False)
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
    """
    Returns a count of the unattended calls per run

    This is done by looking for any instances where no callsign was assigned, indicating that
    no resource was sent
    """
    return call_df[call_df['callsign'].isna()].groupby('run_number')[['P_ID']].count().reset_index()

def get_AVERAGE_UNATTENDED_calls_per_run(call_df):
    """
    Returns a count of the calls per run, averaged across all runs

    This is done by looking for any instances where no callsign was assigned, indicating that
    no resource was sent
    """
    unattended_calls_per_run = get_UNATTENDED_calls_per_run(call_df)
    return unattended_calls_per_run.mean()['P_ID'].round(2)

def display_UNTATTENDED_calls_per_run(call_df):
    """
    Alternative to get_perc_unattended_string(), using a different approach, allowing for
    robustness testing

    Here, this is done by looking for any instances where no callsign was assigned, indicating that
    no resource was sent
    """
    total_calls = get_AVERAGE_calls_per_run(call_df)
    unattended_calls = get_AVERAGE_UNATTENDED_calls_per_run(call_df)

    return f"{unattended_calls:.0f} of {total_calls:.0f} ({(unattended_calls/total_calls):.1%})"

def plot_hourly_call_counts(call_df, params_df,
                            box_plot=False, average_per_month=False,
                            bar_colour="teal", title="Calls Per Hour", use_poppins=False,
                            error_bar_colour="charcoal", show_error_bars_bar=True,
                            show_historical=True,
                            historical_data_path="../actual_data/jobs_by_hour.csv"):
    """
    Produces an interactive plot showing the number of calls that were received per hour in
    the simulation

    This can be compared with the processed historical data used to inform the simulation
    """
    hourly_calls_per_run = call_df.groupby(['hour', 'run_number'])[['P_ID']].count().reset_index().rename(columns={"P_ID": "count"})

    fig = go.Figure()

    if show_historical:
        jobs_per_hour_historic = pd.read_csv(historical_data_path)

        jobs_per_hour_historic['month'] = pd.to_datetime(jobs_per_hour_historic['month'],dayfirst=True)
        jobs_per_hour_historic['year_numeric'] = jobs_per_hour_historic['month'].apply(lambda x: x.year)
        jobs_per_hour_historic['month_numeric'] = jobs_per_hour_historic['month'].apply(lambda x: x.month)
        jobs_per_hour_historic_long = jobs_per_hour_historic.melt(id_vars=['month','month_numeric', 'year_numeric'])
        # jobs_per_hour_historic_long["hour"] = jobs_per_hour_historic_long['variable'].str.extract(r"(\d+)\s")
        jobs_per_hour_historic_long.rename(columns={'variable': 'hour'}, inplace=True)
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
        ))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(title="Hour", dtick=1),
            yaxis=dict(title=y_label)
        )

    else:

        # Create required dataframe for simulation output display
        aggregated_data = hourly_calls_per_run.groupby("hour").agg(
            mean_count=("count", "mean"),
            # std_count=("count", "std")
            se_count=("count", lambda x: x.std() / np.sqrt(len(x)))  # Standard Error
        ).reset_index()

        if show_error_bars_bar:
                # error_y = aggregated_data["std_count"]
                error_y = aggregated_data["se_count"]
        else:
            error_y=None

        if average_per_month:
            aggregated_data['mean_count'] = aggregated_data['mean_count'] / (float(_processing_functions.get_param("sim_duration", params_df))/60/24/ 30)
            # aggregated_data['std_count'] = aggregated_data['std_count'] / (float(_processing_functions.get_param("sim_duration", params_df))/60/24/ 30)
            aggregated_data['se_count'] = aggregated_data['se_count'] / (float(_processing_functions.get_param("sim_duration", params_df))/60/24/ 30)


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
                       historical_monthly_job_data_path="../historical_data/historical_jobs_per_month.csv",
                       show_historical_individual_years=False,
                       job_count_col="total_jobs"):

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
        historical_jobs_per_month["month"] = pd.to_datetime(historical_jobs_per_month['month'])

        historical_jobs_per_month["Month_Numeric"] = (
            historical_jobs_per_month["month"].apply(lambda x: x.month)
            )

        historical_jobs_per_month["Year_Numeric"] = (
            historical_jobs_per_month["month"]
            .apply(lambda x: x.year)
            )

        historical_summary = (
            historical_jobs_per_month
            .groupby('Month_Numeric')[job_count_col]
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

        if (historical_jobs_per_month[job_count_col].max() * 1.1) > (call_counts_monthly["monthly_calls"].max()*1.1):
            fig = fig.update_yaxes({'range': (0, historical_jobs_per_month[job_count_col].max() * 1.1)})

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
                    y=new_df[job_count_col],
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
            print("==_job_count_calculation plot_monthly_calls(): call_counts_monthly")
            print(call_counts_monthly)

            # Ensure we only have one row per month to avoid issues with filling the historical range
            call_counts_historical_plotting_min_max = (
                call_counts_monthly[['month_start','historic_max','historic_min']]
                .drop_duplicates()
                )

            fig.add_trace(go.Scatter(
                x=call_counts_historical_plotting_min_max["month_start"],
                y=call_counts_historical_plotting_min_max["historic_max"],
                mode='lines',
                showlegend=False,
                line=dict(color='rgba(0,0,0,0)'),  # Invisible line (just the area)
            ))

            fig.add_trace(go.Scatter(
                x=call_counts_historical_plotting_min_max["month_start"],
                y=call_counts_historical_plotting_min_max["historic_min"],
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

def plot_daily_call_counts(call_df, params_df, box_plot=False, average_per_month=False,
                            bar_colour="teal", title="Calls Per Day", use_poppins=False,
                            error_bar_colour="charcoal", show_error_bars_bar=True,
                            show_historical=True,
                            historical_data_path="../historical_data/historical_monthly_totals_by_day_of_week.csv"):
    # Create a lookup dict to ensure formatting of weekday is consistent across simulated
    # and historical datasets
    day_dict = {
        'Mon': 'Monday',
        'Tue': 'Tuesday',
        'Wed': 'Wednesday',
        'Thu': 'Thursday',
        'Fri': 'Friday',
        'Sat': 'Saturday',
        'Sun': 'Sunday'
    }

    # Create a list to ensure days of the week are displayed in the correct order in plot
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Calculate the number of calls per day across a full run
    daily_calls_per_run = (
        call_df.groupby(['day', 'run_number'])[['P_ID']].count()
        .reset_index().rename(columns={"P_ID": "count"})
        )

    # Ensure the days in the simulated dataset are formatted the same as the historical dataset
    daily_calls_per_run['day'] = daily_calls_per_run['day'].apply(lambda x: day_dict[x])

    print(daily_calls_per_run)

    # Create a blank figure to build on
    fig = go.Figure()

    ###########
    # Add historical data if option selected
    ###########
    if show_historical:
        jobs_per_day_historic = pd.read_csv(historical_data_path)
        jobs_per_day_historic['month'] = pd.to_datetime(jobs_per_day_historic['month'],dayfirst=True)
        def count_weekdays_in_month(year, month):
            """Returns a dictionary with the count of each weekday in a given month."""
            weekday_counts = {day: 0 for day in day_name}

            _, num_days = monthrange(year, month)  # Get total days in month
            for day in range(1, num_days + 1):
                weekday = day_name[datetime.datetime(year, month, day).weekday()]
                weekday_counts[weekday] += 1

            return weekday_counts

        def compute_average_calls(df):
            """Computes the average calls received per day of the week for each month."""
            results = []
            for _, row in df.iterrows():
                year, month = row["month"].year, row["month"].month
                weekday_counts = count_weekdays_in_month(year, month)

                averages = {day: row[day] / weekday_counts[day] for day in weekday_counts}
                averages["month"] = row["month"].strftime("%Y-%m")
                results.append(averages)

            return pd.DataFrame(results)

        print(jobs_per_day_historic)
        # Compute the average calls per day
        jobs_per_day_historic = compute_average_calls(jobs_per_day_historic)
        jobs_per_day_historic['month'] = pd.to_datetime(jobs_per_day_historic['month'],dayfirst=True)

        jobs_per_day_historic['year_numeric'] = jobs_per_day_historic['month'].apply(lambda x: x.year)
        jobs_per_day_historic['month_numeric'] = jobs_per_day_historic['month'].apply(lambda x: x.month)
        print(jobs_per_day_historic)

        jobs_per_day_historic_long = jobs_per_day_historic.melt(
            id_vars=['month','month_numeric', 'year_numeric']
            )
        # jobs_per_day_historic_long["hour"] = jobs_per_day_historic_long['variable'].str.extract(r"(\d+)\s")
        jobs_per_day_historic_long.rename(columns={'variable': 'day'}, inplace=True)
        # jobs_per_day_historic_long["hour"] = jobs_per_day_historic_long["hour"].astype('int')
        jobs_per_day_historic_long = jobs_per_day_historic_long[~jobs_per_day_historic_long['value'].isna()]

        if not average_per_month:
            jobs_per_day_historic_long['value'] = (
                jobs_per_day_historic_long['value'] *
                (float(_processing_functions.get_param("sim_duration", params_df))/ 60 / 24 / 7)
                )

        jobs_per_day_historic_agg = (
            jobs_per_day_historic_long.groupby(['day'])['value']
            .agg(['min','max', q10, q90])
            ).reset_index()

        fig.add_trace(go.Bar(
            x=jobs_per_day_historic_agg["day"],
            y=jobs_per_day_historic_agg["max"] - jobs_per_day_historic_agg["min"],  # The range
            base=jobs_per_day_historic_agg["min"],  # Starts from the minimum
            name="Historical Range",
            marker_color="rgba(100, 100, 255, 0.2)",  # Light blue with transparency
            hoverinfo="skip",  # Hide hover info for clarity
            showlegend=True,
            width=1.0,  # Wider bars to make them contiguous
            offsetgroup="historical"  # Grouping ensures alignment
        ))

        fig.add_trace(go.Bar(
            x=jobs_per_day_historic_agg["day"],
            y=jobs_per_day_historic_agg["q90"] - jobs_per_day_historic_agg["q10"],  # The range
            base=jobs_per_day_historic_agg["q10"],  # Starts from the minimum
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

    ################################
    # Add in the actual data
    ################################

    if box_plot:

        if average_per_month:
            daily_calls_per_run['average_per_month'] = (
                daily_calls_per_run['count'] /
                (float(_processing_functions.get_param("sim_duration", params_df))/ 60 / 24 / 7)
                )
            y_column = "average_per_month"
            y_label = "Average Monthly Calls Per Day Across Simulation<br>Averaged Across Simulation Runs"
        else:
            y_column = "count"
            y_label = "Total Calls Per Hour Day Simulation<br>Averaged Across Simulation Runs"


        # Add box plot trace
        fig.add_trace(go.Box(
            x=daily_calls_per_run["day"],
            y=daily_calls_per_run[y_column],
            name="Simulated Mean",
            width=0.4,
            marker=dict(color=DAA_COLORSCHEME[bar_colour]),
            showlegend=True,
            boxpoints="outliers",  # Show all data points
        ))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(title="Day", dtick=1),
            yaxis=dict(title=y_label)
        )

    else:

        # Create required dataframe for simulation output display
        aggregated_data = daily_calls_per_run.groupby("day").agg(
            mean_count=("count", "mean"),
            # std_count=("count", "std")
            se_count=("count", lambda x: x.std() / np.sqrt(len(x)))  # Standard Error
        ).reset_index()

        if show_error_bars_bar:
                # error_y = aggregated_data["std_count"]
                error_y = aggregated_data["se_count"]
        else:
            error_y=None

        # Add the bar trace if plotting averages across
        if average_per_month:
            aggregated_data['mean_count'] = (
                aggregated_data['mean_count'] /
                (float(_processing_functions.get_param("sim_duration", params_df))/ 60 / 24/ 7)
                )
            # aggregated_data['std_count'] = (
            #     aggregated_data['std_count'] /
            #     (float(_processing_functions.get_param("sim_duration", params_df))/ 60 / 24/ 7)
            #     )

            aggregated_data['se_count'] = (
                aggregated_data['se_count'] /
                (float(_processing_functions.get_param("sim_duration", params_df))/ 60 / 24/ 7)
                )

            fig.add_trace(go.Bar(
                x=aggregated_data["day"],
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
                yaxis_title="Average Monthly Calls Per Day Across Simulation<br>Averaged Across Simulation Runs",
                xaxis_title="Day"
            )

        # Add the bar trace if plotting total calls over the course of the simulation
        else:
            fig.add_trace(go.Bar(
                x=aggregated_data["day"],
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
                yaxis_title="Total Calls Per Day Across Simulation",
                xaxis_title="Day"
            )

    if not box_plot:
        fig = fig.update_traces(error_y_color=DAA_COLORSCHEME[error_bar_colour])

    fig.update_xaxes(categoryorder='array', categoryarray= day_order)


    if use_poppins:
        return fig.update_layout(font=dict(family="Poppins", size=18, color="black"))
    else:
        return fig


def get_historical_attendance_df(
        data_path="historical_data/historical_missed_calls_by_month.csv"
):

    full_jobs_df = pd.read_csv(data_path)
    full_jobs_df = full_jobs_df.pivot(columns="callsign_group_simplified", index="month_start", values="count").reset_index()

    full_jobs_df.rename(columns={'HEMS (helo or car) available and sent': 'jobs_attended',
                                'No HEMS available':'jobs_not_attended'}, inplace=True)

    full_jobs_df['all_received_calls'] = full_jobs_df['jobs_attended'] + full_jobs_df['jobs_not_attended']
    full_jobs_df['perc_unattended_historical'] = full_jobs_df['jobs_not_attended']/full_jobs_df['all_received_calls'].round(2)

    return full_jobs_df

def plot_historical_missed_jobs_data(
        data_path="historical_data/historical_missed_calls_by_month.csv",
        format="stacked_bar"
        ):

    full_jobs_df = get_historical_attendance_df(data_path=data_path)

    if format=="stacked_bar":
        return px.bar(
                full_jobs_df[['month','jobs_not_attended','jobs_attended']].melt(id_vars="month"),
                x="month",
                y="value",
                color="variable"
                )

    elif format=="line_not_attended_count":
        return px.line(full_jobs_df, x="month", y="jobs_not_attended")

    elif format=="line_not_attended_perc":
        return px.line(full_jobs_df, x="month", y="perc_unattended_historical")

    elif format=="string":
        # This approach can distort the result by giving more weight to months with higher numbers of calls
        # However, for system-level performance, which is what we care about here, it's a reasonable option
        all_received_calls_period = full_jobs_df['all_received_calls'].sum()
        all_attended_jobs_period = full_jobs_df['jobs_attended'].sum()
        return (((all_received_calls_period - all_attended_jobs_period) / all_received_calls_period)*100)

        # Alternative is to take the mean of means
        # return full_jobs_df['perc_unattended_historical'].mean()*100


    else:
        # Melt the DataFrame to long format
        df_melted = full_jobs_df[['month', 'jobs_not_attended', 'jobs_attended']].melt(id_vars='month')

        # Calculate proportions per month
        df_melted['proportion'] = df_melted.groupby('month')['value'].transform(lambda x: x / x.sum())

        # Plot proportions
        fig = px.bar(
            df_melted,
            x='month',
            y='proportion',
            color='variable',
            text='value',  # Optional: to still show raw values on hover
        )

        fig.update_layout(barmode='stack', yaxis_tickformat='.0%', yaxis_title='Proportion')
        fig.show()


def plot_missed_jobs(historical_df_path="historical_data/historical_missed_calls_by_hour.csv",
                     historical_df_path_quarter="historical_data/historical_missed_calls_by_quarter_and_hour.csv",
                     simulated_df_path="data/run_results.csv",
                     show_proportions_per_hour=False,
                     by_quarter=False):
    if not by_quarter:
        historical_df = pd.read_csv(historical_df_path)
        simulated_df = pd.read_csv(simulated_df_path)

        simulated_df_resource_preferred_outcome = simulated_df[simulated_df["event_type"] == "resource_preferred_outcome"]

        simulated_df_resource_preferred_outcome["outcome_simplified"] = (
            simulated_df_resource_preferred_outcome["time_type"].apply(
                lambda x: "No HEMS available" if "No HEMS resource available" in x
                else "HEMS (helo or car) available and sent")
                )

        historical_df.rename(columns={"callsign_group_simplified": "outcome_simplified"}, inplace=True)
        historical_df["what"] = "Historical"

        simulated_df_counts = simulated_df_resource_preferred_outcome.groupby(['outcome_simplified','hour'])[['P_ID']].count().reset_index().rename(columns={"P_ID": "count"})
        simulated_df_counts["what"] = "Simulated"

        full_df = pd.concat([simulated_df_counts, historical_df])

        if not show_proportions_per_hour:
            fig = px.bar(
                full_df,
                x="hour",
                y="count",
                color="outcome_simplified",
                barmode="stack",
                facet_row="what",
                facet_row_spacing=0.2,
                labels={"outcome_simplified": "Job Outcome", "count": "Count of Jobs", "hour": "Hour"}
            )

            # Allow each y-axis to be independent
            fig.update_yaxes(matches=None)

            # Move facet row labels above each subplot, aligned left
            fig.for_each_annotation(lambda a: a.update(
                text=a.text.split("=")[-1],  # remove 'what='
                x=0,                         # align left
                xanchor="left",
                y=a.y + 0.35,                # move label above the plot
                yanchor="top",
                textangle=0,                # horizontal
                font=dict(size=24)
            ))

            # Ensure x axis tick labels appear on both facets
            fig.for_each_xaxis(
                lambda xaxis: xaxis.update(showticklabels=True, tickmode = 'linear',
                tick0 = 0,
                dtick = 1)
                )

            # Increase top margin to prevent overlap
            fig.update_layout(margin=dict(t=100))

            return fig
        else:
            # Compute proportions within each hour + source
            df_prop = (
                full_df
                .groupby(["hour", "what"])
                .apply(lambda d: d.assign(proportion=d["count"] / d["count"].sum()))
                .reset_index(drop=True)
            )

            fig = px.bar(
                df_prop,
                x="what",
                y="proportion",
                color="outcome_simplified",
                barmode="stack",
                facet_col="hour",
                category_orders={"hour": sorted(full_df["hour"].unique())},
                title="Proportion of HEMS Outcomes by Hour and Data Source",
                labels={"proportion": "Proportion", "what": ""}
            )

            fig.update_yaxes(range=[0, 1], matches="y")  # consistent y-axis
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # clean facet labels

            fig.update_layout(
                hovermode="x unified",
                legend_title_text="Outcome",
            )

            fig.update_layout(
                legend=dict(
                    orientation="h",           # horizontal layout
                    yanchor="bottom",
                    y=1.12,                    # a bit above the plot
                    xanchor="center",
                    x=0.5                      # center aligned
                )
            )

            # Increase spacing below title
            fig.update_layout(margin=dict(t=150))

            fig.for_each_xaxis(lambda axis: axis.update(
                # Rotate labels
                tickangle=90,
                # Force display of both labels even on narrow screens
                showticklabels=True,
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1)
                )

            return fig
    # if by_quarter
    else:
        historical_df = pd.read_csv(historical_df_path_quarter)
        historical_df.rename(columns={"callsign_group_simplified": "outcome_simplified"}, inplace=True)
        historical_df["what"] = "Historical"

        simulated_df = pd.read_csv(simulated_df_path)
        simulated_df_resource_preferred_outcome = simulated_df[simulated_df["event_type"] == "resource_preferred_outcome"].copy()
        simulated_df_resource_preferred_outcome.rename(columns={"qtr":"quarter"}, inplace=True)

        simulated_df_resource_preferred_outcome["outcome_simplified"] = (
            simulated_df_resource_preferred_outcome["time_type"].apply(
                lambda x: "No HEMS available" if "No HEMS resource available" in x
                else "HEMS (helo or car) available and sent")
        )

        simulated_df_counts = (
            simulated_df_resource_preferred_outcome
            .groupby(['outcome_simplified','quarter','hour'])[['P_ID']]
            .count().reset_index()
            .rename(columns={"P_ID": "count"})
            )
        simulated_df_counts["what"] = "Simulated"
        full_df = pd.concat([simulated_df_counts, historical_df])

        if not show_proportions_per_hour:

            fig = px.bar(
                full_df,
                x="hour",
                y="count",
                color="outcome_simplified",
                barmode="stack",
                facet_row="what",
                facet_col="quarter",
                facet_row_spacing=0.2,
                labels={"outcome_simplified": "Job Outcome", "count": "Count of Jobs", "hour": "Hour"}
            )

            # Allow each y-axis to be independent
            fig.update_yaxes(matches=None)

            fig.for_each_xaxis(
                lambda xaxis: xaxis.update(showticklabels=True, tickmode = 'linear',
                tick0 = 0,
                dtick = 1))

            # Increase top margin to prevent overlap
            fig.update_layout(margin=dict(t=100))

            fig.update_layout(
                legend=dict(
                    orientation="h",           # horizontal layout
                    yanchor="bottom",
                    y=1.12,                    # a bit above the plot
                    xanchor="center",
                    x=0.5                      # center aligned
                )
            )

            return fig

        else:
            # Step 1: Compute proportions within each hour + source
            df_prop = (
                full_df
                .groupby(["quarter", "hour", "what"])
                .apply(lambda d: d.assign(proportion=d["count"] / d["count"].sum()))
                .reset_index(drop=True)
            )

            # Step 2: Plot
            fig = px.bar(
                df_prop,
                x="what",
                y="proportion",
                color="outcome_simplified",
                barmode="stack",
                facet_col="hour",
                facet_row="quarter",
                category_orders={"hour": sorted(full_df["hour"].unique())},
                title="Proportion of HEMS Outcomes by Hour and Data Source",
                labels={"proportion": "Proportion", "what": ""}
            )

            fig.update_yaxes(range=[0, 1], matches="y")  # consistent y-axis
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # clean facet labels

            fig.update_layout(
                hovermode="x unified",
                legend_title_text="Outcome",
            )

            fig.update_layout(
                legend=dict(
                    orientation="h",           # horizontal layout
                    yanchor="bottom",
                    y=1.12,                    # a bit above the plot
                    xanchor="center",
                    x=0.5                      # center aligned
                )
            )

            fig.update_layout(margin=dict(t=150))

            fig.for_each_xaxis(lambda axis: axis.update(
                tickangle=90,
                # showticklabels=True,
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1)
                )

            return fig

def plot_jobs_per_callsign(
        historical_df_path="historical_data/historical_jobs_per_day_per_callsign.csv",
        simulated_df_path="data/run_results.csv"
        ):
    # Create a count of the number of days in the sim that each resource had that many jobs
    # i.e. how many days did CC70 have 0 jobs, 1 job, 2 jobs, etc.
    df = pd.read_csv(simulated_df_path)

    df["date"] = pd.to_datetime(df["timestamp_dt"]).dt.date
    all_counts = df[df["event_type"]=="resource_use"].groupby(["time_type", "date", "run_number"])["P_ID"].count().reset_index()
    all_counts.rename(columns={'P_ID': "jobs_in_day"}, inplace=True)

    # We must assume any missing day in our initial count df is a 0 count
    # So generate a df with every possible combo
    all_combinations = pd.DataFrame(
        list(itertools.product(all_counts['time_type'].unique(), df['date'].unique(), df['run_number'].unique())),
        columns=['time_type', 'date', 'run_number']
    )
    # Join this in
    merged = all_combinations.merge(
        all_counts,
        on=['time_type', 'date', 'run_number'],
        how='left'
        )
    # Fill na values with 0
    merged['jobs_in_day'] = merged['jobs_in_day'].fillna(0).astype(int)
    # Finally transform into pure counts
    sim_count_df = (merged.groupby(
        ['time_type', 'jobs_in_day']
        )[['date']].count()
        .reset_index()
        .rename(columns={'date':'count', 'time_type': 'callsign'})
        )

    sim_count_df['what'] = 'Simulated'

    # Bring in historical data
    jobs_per_day_per_callsign_historical = pd.read_csv(historical_df_path)
    jobs_per_day_per_callsign_historical['what'] = 'Historical'

    # Join the two together
    full_df = pd.concat([jobs_per_day_per_callsign_historical, sim_count_df])

    # Plot as histograms
    fig = px.histogram(full_df,
             x="jobs_in_day", y="count", color="what",
             facet_col="callsign",
             facet_row="what",
             histnorm="percent",
            #  barmode="overlay",
            #  opacity=1
             )

    return fig
