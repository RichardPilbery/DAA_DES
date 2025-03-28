import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Ensure this folder is in sys.path
import _processing_functions
import pandas as pd
import plotly.express as px

import _vehicle_calculation
import _job_count_calculation
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re

from _app_utils import DAA_COLORSCHEME


def make_utilisation_model_dataframe(path="../data/run_results.csv",
                                     params_path="../data/run_params_used.csv",
                                     rota_path="../data/HEMS_ROTA.csv",
                                     callsign_path="../actual_data/callsign_registration_lookup.csv",
                                     service_path="../data/service_dates.csv",
                                     ):
    df = pd.read_csv(path)
    params_df = pd.read_csv(params_path)
    n_runs = len(df["run_number"].unique())

    daily_availability, total_avail_minutes = (
        _vehicle_calculation.calculate_available_hours_v2(
            params_df,
            rota_data=pd.read_csv(rota_path),
            service_data=pd.read_csv(service_path),
            callsign_data=pd.read_csv(callsign_path),
            output_by_month=False,
            long_format_df=False
                    )
                )


    print("==_utilisation_result_calculation - make_utilisation_model_dataframe - run_vehicle_calculation.calculate_available_hours_v2==")
    print("daily_availability")
    print(daily_availability)

    print("total_avail_minutes")
    print(total_avail_minutes)

    # total_avail_minutes["callsign"] = total_avail_minutes["callsign"].str.replace("CC", "C")

    print("df")
    print(df)
    # Add callsign column if not already present in the dataframe passed to the function
    if 'callsign' not in df.columns:
        df = _processing_functions.make_callsign_column(df)

    # Restrict to only events in the event log where resource use was starting or ending
    resource_use_only = df[df["event_type"].isin(["resource_use", "resource_use_end"])]

    del df

    print("==resource_use_only==")
    print(resource_use_only)
    # Pivot to wide-format dataframe with one row per patient/call
    # and columns for start and end types
    resource_use_wide = (
        resource_use_only[["P_ID", "run_number", "event_type", "timestamp_dt",
                           "callsign_group", "vehicle_type", "callsign"]]
        .pivot(index=["P_ID", "run_number", "callsign_group", "vehicle_type", "callsign"],
               columns="event_type", values="timestamp_dt").reset_index()
               )

    del resource_use_only

    print("==resource_use_wide - initial==")
    print(resource_use_wide)
    # If utilisation end date is missing then set to end of model
    # as we can assume this is a call that didn't finish before the model did
    resource_use_wide = _processing_functions.fill_missing_values(
        resource_use_wide, "resource_use_end",
        _processing_functions.get_param("sim_end_date", params_df)
        )

    # If utilisation start time is missing, then set to start of model + warm-up time (if relevant)
    # as can assume this is a call that started before the warm-up period elapsed but finished
    # after the warm-up period elapsed
    # TODO: need to add in a check to ensure this only happens for calls at the end of the model,
    # not due to errors elsewhere that could fail to assign a resource end time
    resource_use_wide = _processing_functions.fill_missing_values(
        resource_use_wide, "resource_use",
        _processing_functions.get_param("warm_up_end_date", params_df)
        )

    # Calculate number of minutes the attending resource was in use on each call
    resource_use_wide["resource_use_duration"] = _processing_functions.calculate_time_difference(
        resource_use_wide, 'resource_use', 'resource_use_end', unit='minutes'
        )

    # ============================================================ #
    # Calculage per-run utilisation,
    # stratified by callsign and vehicle type (car/helicopter)
    # ============================================================ #
    utilisation_df_per_run = (
        resource_use_wide.groupby(['run_number', 'vehicle_type', 'callsign'])
        [["resource_use_duration"]]
        .sum()
        )

    # Join with df of how long each resource was available for in the sim
    # We will for now assume this is the same across each run
    utilisation_df_per_run = utilisation_df_per_run.reset_index(drop=False).merge(
        total_avail_minutes, on="callsign", how="left"
        )

    utilisation_df_per_run["perc_time_in_use"] = (
        utilisation_df_per_run["resource_use_duration"].astype(float) /
        # float(_processing_functions.get_param("sim_duration", params_df))
        utilisation_df_per_run["total_available_minutes_in_sim"].astype(float)
        )

    # Add column of nicely-formatted values to make printing values more streamlined
    utilisation_df_per_run["PRINT_perc"] = utilisation_df_per_run["perc_time_in_use"].apply(
        lambda x: f"{x:.1%}")

    print("==utilisation_df_per_run==")
    print(utilisation_df_per_run)

    # ============================================================ #
    # Calculage averge utilisation across simulation,
    # stratified by callsign group
    # ============================================================ #
    print("==resource_use_wide==")
    print(resource_use_wide)

    utilisation_df_per_run_by_csg = (
        resource_use_wide.groupby(['callsign_group'])
        [["resource_use_duration"]]
        .sum()
        )

    utilisation_df_per_run_by_csg["resource_use_duration"] = (
        utilisation_df_per_run_by_csg["resource_use_duration"] /
        n_runs
        )

    utilisation_df_per_run_by_csg = utilisation_df_per_run_by_csg.reset_index()

    print("==utilisation_df_per_run_by_csg==")
    print(utilisation_df_per_run_by_csg)

    total_avail_minutes_per_csg = total_avail_minutes.groupby('callsign_group').head(1).drop(columns='callsign')
    print("==utilisation_df_per_run_by_csg==")
    print(utilisation_df_per_run_by_csg)

    print("==total_avail_minutes_per_csg")
    print(total_avail_minutes_per_csg)
    total_avail_minutes_per_csg['callsign_group'] =  total_avail_minutes_per_csg['callsign_group'].astype('float')

    utilisation_df_per_run_by_csg = utilisation_df_per_run_by_csg.merge(
        total_avail_minutes_per_csg, on="callsign_group", how="left"
        )

    utilisation_df_per_run_by_csg["perc_time_in_use"] = (
        utilisation_df_per_run_by_csg["resource_use_duration"].astype(float) /
        # float(_processing_functions.get_param("sim_duration", params_df))
        utilisation_df_per_run_by_csg["total_available_minutes_in_sim"].astype(float)
        )

    utilisation_df_per_run_by_csg["PRINT_perc"] = utilisation_df_per_run_by_csg["perc_time_in_use"].apply(
        lambda x: f"{x:.1%}"
        )


    # ============================================================ #
    # Calculage averge utilisation across simulation,
    # stratified by callsign and vehicle type (car/helicopter)
    # ============================================================ #
    utilisation_df_overall = (
        utilisation_df_per_run.groupby(['callsign', 'vehicle_type'])
        [["resource_use_duration"]]
        .sum()
        )

    utilisation_df_overall["resource_use_duration"] = (
        utilisation_df_overall["resource_use_duration"] /
        n_runs
        )

    utilisation_df_overall = utilisation_df_overall.reset_index(drop=False).merge(
        total_avail_minutes, on="callsign", how="left"
        )

    utilisation_df_overall["perc_time_in_use"] = (
        utilisation_df_overall["resource_use_duration"].astype(float) /
        # float(_processing_functions.get_param("sim_duration", params_df))
        utilisation_df_overall["total_available_minutes_in_sim"].astype(float)
    )

    # Add column of nicely-formatted values to make printing values more streamlined
    utilisation_df_overall["PRINT_perc"] = utilisation_df_overall["perc_time_in_use"].apply(
        lambda x: f"{x:.1%}"
        )


    # Return tuple of values
    return (resource_use_wide, utilisation_df_overall,
            utilisation_df_per_run, utilisation_df_per_run_by_csg)


def make_SIMULATION_utilisation_variation_plot(utilisation_df_per_run,
                                               car_colour="blue",
                                               helicopter_colour="red",
                                               use_poppins=False):
    """
    Creates a box plot to visualize the variation in resource utilization
    across all simulation runs.

    Parameters
    ----------
    utilisation_df_per_run : pandas.DataFrame
        A DataFrame containing utilization data per simulation run, generated by
        make_utilisation_model_dataframe().
        It must include the columns:
        - "callsign": Identifier for the resource.
        - "perc_time_in_use": Percentage of time the resource was in use.
        - "vehicle_type": Type of vehicle (e.g., "Car", "Helicopter").

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly box plot showing the distribution of utilization percentages
        for each resource, grouped by vehicle type.

    Notes
    -----
    - The `vehicle_type` column values are capitalized for consistency.
    - The x-axis values are formatted as percentages with no decimal places.
    - A custom color scheme is applied based on vehicle type:
      - "Car" is mapped to `DAA_COLORSCHEME["blue"]`.
      - "Helicopter" is mapped to `DAA_COLORSCHEME["red"]`.

    Example
    -------
    >>> fig = make_SIMULATION_utilisation_variation_plot(utilisation_df)
    >>> fig.show()
    """
    utilisation_df_per_run = utilisation_df_per_run.reset_index()
    utilisation_df_per_run["vehicle_type"] = utilisation_df_per_run["vehicle_type"].str.title()



    fig = (px.box(utilisation_df_per_run,
        x="perc_time_in_use",
        y="callsign",
        color="vehicle_type",
        title="Variation in Resource Utilisation Across All Simulation Runs",
                    labels={
                     "callsign": "Callsign",
                     "perc_time_in_use": "Average Percentage of Available<br>Time Spent in Use",
                     "vehicle_type": "Vehicle Type"
                 },
                 color_discrete_map={'Car': DAA_COLORSCHEME[car_colour],
                                     "Helicopter": DAA_COLORSCHEME[helicopter_colour]})
        .update_layout(
                    xaxis={
                        "tickformat": ".0%"  # Formats as percentage with no decimal places
                    }))

    # TODO: Add indications of good/bad territory for utilisation levels

    if use_poppins:
        return fig.update_layout(font=dict(family="Poppins", size=18, color="black"))
    else:
        return fig

def make_SIMULATION_utilisation_summary_plot(utilisation_df_overall,
                                               car_colour="blue",
                                               helicopter_colour="red",
                                               use_poppins=False):
    """
    Creates a bar plot to summarize the average resource utilization
    across all simulation runs.

    Parameters
    ----------
    utilisation_df_overall : pandas.DataFrame
        A DataFrame containing overall utilization data, generated by
        make_utilisation_model_dataframe()..
        It must include the columns:
        - "callsign": Identifier for the resource.
        - "perc_time_in_use": Average percentage of time the resource was in use.
        - "vehicle_type": Type of vehicle (e.g., "Car", "Helicopter").

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly bar plot showing the average utilization percentage
        for each resource, grouped by vehicle type.

    Notes
    -----
    - The `vehicle_type` column values are capitalized for consistency.
    - The y-axis values are formatted as percentages with no decimal places.
    - A custom color scheme is applied based on vehicle type:
      - "Car" is mapped to `DAA_COLORSCHEME["blue"]`.
      - "Helicopter" is mapped to `DAA_COLORSCHEME["red"]`.

    Example
    -------
    >>> fig = make_SIMULATION_utilisation_summary_plot(utilisation_df)
    >>> fig.show()
    """
    utilisation_df_overall = utilisation_df_overall.reset_index()
    utilisation_df_overall["vehicle_type"] = utilisation_df_overall["vehicle_type"].str.title()

    fig = (px.bar(utilisation_df_overall,
                    y="perc_time_in_use",
                    x="callsign",
                    color="vehicle_type",
                    title="Average Resource Utilisation Across All Simulation Runs",
                    labels={
                     "callsign": "Callsign",
                     "perc_time_in_use": "Average Percentage of Available<br>Time Spent in Use",
                     "vehicle_type": "Vehicle Type"
                 },
                 color_discrete_map={'Car': DAA_COLORSCHEME[car_colour],
                                     "Helicopter": DAA_COLORSCHEME[helicopter_colour]})
            .update_layout(
                yaxis={
                    "tickformat": ".0%"  # Formats as percentage with no decimal places
                })
                )
    # TODO: Add indications of good/bad territory for utilisation levels

    if use_poppins:
        return fig.update_layout(font=dict(family="Poppins", size=18, color="black"))
    else:
        return fig

def make_RWC_utilisation_dataframe(
        historical_df_path="../historical_data/historical_monthly_resource_utilisation.csv",
        rota_path="../actual_data/HEMS_ROTA.csv",
        callsign_path="../actual_data/callsign_registration_lookup.csv",
        service_path="../data/service_dates.csv"):

    historical_utilisation_df = pd.read_csv(historical_df_path)

    def calculate_theoretical_time(
        historical_df,
        rota_df,
        service_df,
        callsign_df,
        long_format_df=True):
        """
        Note that this function has been partially provided by ChatGPT.
        """

        rota_df = rota_df.merge(callsign_df, on="callsign")
        service_df = service_df.merge(callsign_df, on="registration")
        print("==calculate_theoretical_time - rota_df after merging with callsign_df==")
        print(rota_df)

        # Convert date columns to datetime format
        historical_df['month'] = pd.to_datetime(historical_df['month'])
        print("==historical_df==")
        print(historical_df)

        service_df['service_start_date'] = pd.to_datetime(service_df['service_start_date'])
        service_df['service_end_date'] = pd.to_datetime(service_df['service_end_date'])

        # Initialize dictionary to store results
        theoretical_availability = {}

        # Iterate over each row in the historical dataset
        for index, row in historical_df.iterrows():
            month_start = row['month']
            month_end = month_start + pd.offsets.MonthEnd(0)
            days_in_month = (month_end - month_start).days + 1

            # Store theoretical available time for each resource
            month_data = {}

            for _, rota in rota_df.iterrows():
                callsign = rota['callsign']
                print(callsign)

                # Determine summer or winter schedule
                is_summer = month_start.month in range(3, 11)
                start_hour = rota['summer_start'] if is_summer else rota['winter_start']
                end_hour = rota['summer_end'] if is_summer else rota['winter_end']

                # Handle cases where the shift ends after midnight
                if end_hour < start_hour:
                    daily_available_time = (24 - start_hour) + end_hour
                else:
                    daily_available_time = end_hour - start_hour

                total_available_time = daily_available_time * days_in_month * 60  # Convert to minutes

                # Adjust for servicing periods
                service_downtime = 0
                for _, service in service_df[service_df['callsign'] == callsign].iterrows():
                    service_start = max(service['service_start_date'], month_start)
                    service_end = min(service['service_end_date'], month_end)

                    if service_start <= service_end:  # Overlapping service period
                        service_days = (service_end - service_start).days + 1
                        service_downtime += service_days * daily_available_time * 60

                # Final available time after accounting for servicing
                # print("==_utilisation_result_calculation.py - make_RWC_utilisation_dataframe - calculate_theoretical_time: Monthly Available Time==")
                # print(month_data)
                month_data[callsign] = total_available_time - service_downtime

            theoretical_availability[month_start.strftime('%Y-%m-01')] = month_data

        theoretical_availability_df = pd.DataFrame(theoretical_availability).T
        theoretical_availability_df.index.name = "month"
        theoretical_availability_df = theoretical_availability_df.reset_index()
        # theoretical_availability_df = theoretical_availability_df.add_prefix("theoretical_availability_")

        theoretical_availability_df.fillna(0.0)

        print("==_utilisation_result_calculation.py - make_RWC_utilisation_dataframe - theoretical availability df==")
        print(theoretical_availability_df)

        if long_format_df:
            theoretical_availability_df = (
                theoretical_availability_df
                .melt(id_vars="month")
                .rename(columns={"value":"theoretical_availability", "variable": "callsign"})
                )

            theoretical_availability_df['theoretical_availability'] = (
                theoretical_availability_df['theoretical_availability'].astype('float')
                )

        return theoretical_availability_df

    theoretical_availability_df = calculate_theoretical_time(
        historical_df=historical_utilisation_df,
        rota_df=pd.read_csv(rota_path),
        callsign_df=pd.read_csv(callsign_path),
        service_df=pd.read_csv(service_path),
        long_format_df=True
    )

    print("==theoretical_availability_df==")
    print(theoretical_availability_df)
    theoretical_availability_df['month'] = pd.to_datetime(theoretical_availability_df['month'])

    historical_utilisation_df_times = (
        historical_utilisation_df.set_index('month')
        .filter(like='total_time').reset_index()
        )

    historical_utilisation_df_times.columns = [
        x.replace("total_time_","")
        for x in historical_utilisation_df_times.columns
        ]

    historical_utilisation_df_times = (
        historical_utilisation_df_times.melt(id_vars="month")
        .rename(columns={"value":"usage", "variable": "callsign"})
        )

    historical_utilisation_df_times = historical_utilisation_df_times.fillna(0)

    print(historical_utilisation_df_times)
    print(theoretical_availability_df)

    historical_utilisation_df_complete = pd.merge(
        left=historical_utilisation_df_times,
        right=theoretical_availability_df,
        on=["callsign", "month"],
        how="left"
    )

    historical_utilisation_df_complete["percentage_utilisation"] = (
        historical_utilisation_df_complete["usage"] /
        historical_utilisation_df_complete["theoretical_availability"]
        )

    historical_utilisation_df_complete["percentage_utilisation_display"] = (
        historical_utilisation_df_complete["percentage_utilisation"].apply(lambda x: f"{x:.1%}")
        )

    historical_utilisation_df_summary = (
        historical_utilisation_df_complete
        .groupby('callsign')['percentage_utilisation']
        .agg(['min', 'max', 'mean', 'median'])*100
        ).round(1)

    print("==historical_utilisation_df_complete==")
    print(historical_utilisation_df_complete)

    print("==historical_utilisation_df_summary==")
    print(historical_utilisation_df_summary)

    return historical_utilisation_df_complete, historical_utilisation_df_summary

def get_hist_util_fig(historical_utilisation_df_summary, callsign="H70", average="mean"):
    return historical_utilisation_df_summary[historical_utilisation_df_summary.index==callsign][average].values[0]

def make_RWC_utilisation_plot(historical_df_path="../historical_data/historical_monthly_resource_utilisation.csv"):
    historical_utilisation_df_complete, historical_utilisation_df_summary = (
                make_RWC_utilisation_dataframe(
                    historical_df_path=historical_df_path
                    )
                )
    fig = px.box(historical_utilisation_df_complete, x="percentage_utilisation", y="callsign")

    return fig

def make_SIMULATION_utilisation_headline_figure(vehicle_type, utilisation_df_overall):
    """
    Options:
        - helicopter
        - solo car
        - helicopter backup car
    """

    if vehicle_type == "helicopter":
        return utilisation_df_overall[utilisation_df_overall["vehicle_type"] == "helicopter"].mean(numeric_only=True)['perc_time_in_use']

    else:
        # assume anything with >= 1 entries in a callsign group is helicopter + backup car
        # NOTE: This assumption may not hold forever! It assumes
        vehicles_per_callsign_group = utilisation_df_overall.groupby('callsign_group').count()[['callsign']]

        if vehicle_type == "solo car":
            car_only = vehicles_per_callsign_group[vehicles_per_callsign_group['callsign'] == 1]
            return utilisation_df_overall[utilisation_df_overall["callsign_group"].isin(car_only.reset_index().callsign_group.values)].mean(numeric_only=True)['perc_time_in_use']

        elif vehicle_type == "helicopter backup car":
            backupcar_only = vehicles_per_callsign_group[vehicles_per_callsign_group['callsign'] == 2]
            utilisation_df_overall = utilisation_df_overall[utilisation_df_overall["vehicle_type"] == "car"]
            return utilisation_df_overall[utilisation_df_overall["callsign_group"].isin(backupcar_only.reset_index().callsign_group.values)].mean(numeric_only=True)['perc_time_in_use']

        else:
            print("Invalid vehicle type entered. Please use 'helicopter', 'solo car' or 'helicopter backup car'")


def prep_util_df_from_call_df(call_df):

    call_df['timestamp_dt'] = pd.to_datetime(call_df['timestamp_dt'])
    call_df['month_start'] = call_df['timestamp_dt'].dt.to_period('M').dt.to_timestamp()

    print("==prep_util_df_from_call_df: call_df==")
    print(call_df)

    jobs_counts_by_callsign_monthly_sim = (
        call_df
        .groupby(['run_number', 'month_start', 'callsign', 'callsign_group', 'vehicle_type'])['P_ID']
        .count().reset_index().rename(columns={'P_ID': 'jobs'})
        )

    print("==jobs_counts_by_callsign_monthly_sim==")
    print(jobs_counts_by_callsign_monthly_sim)

    jobs_counts_by_callsign_monthly_sim = jobs_counts_by_callsign_monthly_sim[~jobs_counts_by_callsign_monthly_sim['callsign'].isna()]

    print("==jobs_counts_by_callsign_monthly_sim - after filtering by mising callsign==")
    print(jobs_counts_by_callsign_monthly_sim)

    all_combinations = pd.MultiIndex.from_product([
        jobs_counts_by_callsign_monthly_sim["month_start"].unique(),
        jobs_counts_by_callsign_monthly_sim["run_number"].unique(),
        jobs_counts_by_callsign_monthly_sim["callsign"].unique()
    ], names=["month_start", "run_number", "callsign"])

    # Reindex the dataframe to include missing callsigns
    jobs_counts_by_callsign_monthly_sim = jobs_counts_by_callsign_monthly_sim.set_index(
        ["month_start", "run_number", "callsign"]
    ).reindex(all_combinations, fill_value=0).reset_index()

    jobs_counts_by_callsign_monthly_sim['callsign_group'] = jobs_counts_by_callsign_monthly_sim['callsign'].str.extract(r'(\d+)')
    jobs_counts_by_callsign_monthly_sim['vehicle_type'] = jobs_counts_by_callsign_monthly_sim['callsign'].apply(lambda x: 'car' if "C" in x else 'helicopter')

    # Compute total jobs per callsign_group per month
    jobs_counts_by_callsign_monthly_sim["total_jobs_per_group"] = jobs_counts_by_callsign_monthly_sim.groupby(["month_start", "callsign_group", "run_number"])["jobs"].transform("sum")

    # Compute percentage of calls per row
    jobs_counts_by_callsign_monthly_sim["percentage_of_group"] = (jobs_counts_by_callsign_monthly_sim["jobs"] / jobs_counts_by_callsign_monthly_sim["total_jobs_per_group"]) * 100

    # Handle potential division by zero (if total_jobs_per_group is 0)
    jobs_counts_by_callsign_monthly_sim["percentage_of_group"] = jobs_counts_by_callsign_monthly_sim["percentage_of_group"].fillna(0)

    sim_averages = jobs_counts_by_callsign_monthly_sim.groupby(["callsign_group", "callsign", "vehicle_type"])[["percentage_of_group"]].mean().reset_index()

    return sim_averages


def make_SIMULATION_stacked_callsign_util_plot(call_df):
    sim_averages = prep_util_df_from_call_df(call_df)

    fig = px.bar(
        sim_averages,
        x="percentage_of_group",
        y="callsign_group",
        color="callsign",
        height=300
    )

    # Update axis labels and legend title
    fig.update_layout(
        yaxis=dict(
            title="Callsign Group",
            tickmode="linear",  # Ensures ticks appear at regular intervals
            dtick=1  # Set tick spacing to 1 unit
        ),
        xaxis=dict(
            title="Utilisation % within Callsign Group"
        ),
        legend_title="Callsign"
    )

    return fig

def create_UTIL_rwc_plot(call_df,
                        real_data_path="../historical_data/historical_jobs_per_month_by_callsign.csv"):

    #############
    # Prep real-world data
    #############
    jobs_by_callsign = pd.read_csv(real_data_path)
    jobs_by_callsign["month"] = pd.to_datetime(jobs_by_callsign["month"], dayfirst=True)

    jobs_by_callsign_long = jobs_by_callsign.melt(id_vars="month").rename(columns={"variable":"callsign", "value":"jobs"})

    all_combinations = pd.MultiIndex.from_product([
        jobs_by_callsign_long["month"].unique(),
        jobs_by_callsign_long["callsign"].unique()
    ], names=["month", "callsign"])

    # Reindex the dataframe to include missing callsigns
    jobs_by_callsign_long = jobs_by_callsign_long.set_index(
        ["month", "callsign"]
    ).reindex(all_combinations, fill_value=0).reset_index()

    jobs_by_callsign_long["callsign_group"] = jobs_by_callsign_long["callsign"].str.extract(r"(\d+)")
    jobs_by_callsign_long = jobs_by_callsign_long[~jobs_by_callsign_long["callsign_group"].isna()]
    jobs_by_callsign_long["vehicle_type"] = jobs_by_callsign_long["callsign"].apply(lambda x: "car" if "CC" in x else "helicopter")

    # Compute total jobs per callsign_group per month
    jobs_by_callsign_long["total_jobs_per_group"] = jobs_by_callsign_long.groupby(["month", "callsign_group"])["jobs"].transform("sum")

    # Compute percentage of calls per row
    jobs_by_callsign_long["percentage_of_group"] = (jobs_by_callsign_long["jobs"] / jobs_by_callsign_long["total_jobs_per_group"]) * 100

    # Handle potential division by zero (if total_jobs_per_group is 0)
    jobs_by_callsign_long["percentage_of_group"] = jobs_by_callsign_long["percentage_of_group"].fillna(0)


    ###############
    # Prep sim data
    ###############

    sim_averages = prep_util_df_from_call_df(call_df)

    print("==utilisation_result_calculation.py - create_UTIL_rwc_plot - sim_averages==")
    print(sim_averages)

    fig = go.Figure()

    # Bar chart (Simulation Averages)
    for idx, vehicle in enumerate(sim_averages["vehicle_type"].unique()):
        filtered_data = sim_averages[sim_averages["vehicle_type"] == vehicle]

        fig.add_trace(go.Bar(
            y=filtered_data["percentage_of_group"],
            x=filtered_data["callsign_group"],
            name=f"Simulated - {vehicle}",
            marker=dict(
            color=list(DAA_COLORSCHEME.values())[idx]),
            width=0.3,
            opacity=0.6,  # Same opacity for consistency
            text=["Simulated:<br>{:.1f}%".format(val) for val in filtered_data["percentage_of_group"]],  # Correct way            textposition="inside",  # Places text just above the x-axis
            insidetextanchor="start",  # Anchors text inside the bottom of the bar
            textfont=dict(
                        color='white'
                    ),
        ))

        fig.update_layout(
            title="<b>Comparison of Allocated Resources by Callsign Group</b><br>Simulation vs Historical Data",
            yaxis_title="Percentage of Jobs in Callsign Group<br>Tasked to Callsign",
            xaxis_title="Callsign Group",
            barmode='group',
            legend_title="Vehicle Type",
            height=600
        )

    for callsign in jobs_by_callsign_long["callsign"].unique():

        filtered_data = jobs_by_callsign_long[
            jobs_by_callsign_long["callsign"] == callsign
        ].groupby(["callsign_group", "vehicle_type", "callsign"])[["percentage_of_group"]].mean().reset_index()

        if filtered_data["callsign_group"].values[0] in ["70", "71"]:

            expected_x = float(filtered_data["callsign_group"].values[0])
            y_value = filtered_data['percentage_of_group'].values[0]
            expected_y =y_value - 1  # Position for the line

            if filtered_data["vehicle_type"].values[0] == "car":
                x_start = expected_x - 0.4
                x_end = expected_x
            else:
                x_start = expected_x
                x_end = expected_x + 0.4

            # Add dashed line
            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[expected_y, expected_y],
                    mode="lines",
                    name=f"Expected Level - {callsign}",
                    showlegend=False,
                    hoverinfo="all",
                    line=dict(dash='dash', color=DAA_COLORSCHEME['charcoal'])
                )
            )

            # Add text annotation above the line
            fig.add_trace(
                go.Scatter(
                    x=[(x_start + x_end) / 2],  # Center the text horizontally
                    y=[expected_y + 5],  # Slightly above the line
                    text=[f"Historical:<br>{y_value:.1f}%"],
                    mode="text",
                    textfont=dict(
                        color='black'
                    ),
                    showlegend=False  # Don't show in legend
                )
            )

    min_x = min(sim_averages["callsign_group"].astype('int').values)
    max_x = max(sim_averages["callsign_group"].astype('int').values)

    tick_vals = list(range(min_x, max_x + 1))  # Tick positions at integer values

    fig.update_layout(
        xaxis=dict(
            titlefont = dict(size=20),
            tickfont = dict(size=25),
            tickmode='array',
            tickvals=tick_vals,  # Ensure ticks are at integer positions
            range=[min_x - 0.5, max_x + 0.5]  # Extend range to start 0.5 units earlier
        ),
        yaxis=dict(ticksuffix="%", titlefont = dict(size = 15), tickfont = dict(size=20),
                     range=[0, 100]),

        legend = dict(font = dict(size = 15)),
        legend_title = dict(font = dict(size = 20))
    )

    return fig
