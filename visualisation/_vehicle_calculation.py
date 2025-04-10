"""
File containing all calculations and visualisations relating to the vehicles.

[] Simultaneous usage of different callsign groups
[] Total available hours
[] Servicing overrun
[] Instances of being unable to lift
[] Resource allocation hierarchies

Covers variation within the simulation, and comparison with real world data.
"""

import _processing_functions
from datetime import datetime
import pandas as pd
import re

def calculate_available_hours(params_df,
        rota_path="../actual_data/HEMS_ROTA.csv",
        # service_data_path="../data/service_dates.csv"
        ):
    """
    Version of a function to calculate the number of hours a resource is available for use
    across the duration of the simulation, based on the rota used during the period, accounting
    for time during the simulation that uses the summer rota and time that uses the winter rota.

    NOTE: Servicing is not taken into account.

    Warm up duration is taken into account.
    """

    warm_up_end = _processing_functions.get_param("warm_up_end_date", params_df)
    warm_up_end = datetime.strptime(warm_up_end, "%Y-%m-%d %H:%M:%S")

    sim_end = _processing_functions.get_param("sim_end_date", params_df)
    sim_end = datetime.strptime(sim_end, "%Y-%m-%d %H:%M:%S")

    hems_rota = pd.read_csv(rota_path)
    # service_dates = pd.read_csv(service_data_path)

    date_range = pd.date_range(start=warm_up_end.date(),
                            end=sim_end.date(),
                            freq='D')
    df = pd.DataFrame({'date': date_range})

    summer_start = datetime.strptime(_processing_functions.get_param("summer_start_date", params_df), "%Y-%m-%d")
    winter_start = datetime.strptime(_processing_functions.get_param("winter_start_date", params_df), "%Y-%m-%d")

    summer_start_month_day = pd.to_datetime(summer_start).strftime('%m-%d')
    winter_start_month_day = pd.to_datetime(winter_start).strftime('%m-%d')

    def is_summer(date):
            year = date.year
            summer_start_date = pd.to_datetime(f"{year}-{summer_start_month_day}")
            winter_start_date = pd.to_datetime(f"{year}-{winter_start_month_day}")
            return summer_start_date <= date < winter_start_date

    df['is_summer'] = df['date'].apply(is_summer)

    # Fill with a blank column for each callsign
    for callsign in hems_rota.callsign:
        df[callsign] = 0

    def update_availability(df, rota):
        df = df.copy()

        for _, row in rota.iterrows():
            callsign = row['callsign']
            summer_start, winter_start = row['summer_start'], row['winter_start']
            summer_end, winter_end = row['summer_end'], row['winter_end']

            for i in range(len(df)):
                is_summer = df.at[i, 'is_summer']
                start_time = summer_start if is_summer else winter_start
                end_time = summer_end if is_summer else winter_end

                # Handle end time past midnight
                if end_time < start_time:
                    duration = (24 - start_time) + end_time
                    df.at[i, callsign] = duration

                    if i + 1 < len(df):  # Add extra hours to next day
                        df.at[i + 1, callsign] = df.at[i + 1, callsign] + end_time
                else:
                    df.at[i, callsign] = end_time - start_time

        # Adjust for simulation start time
        first_day = df.iloc[0]
        for callsign in rota['callsign']:
            if first_day[callsign] > 0:
                first_day[callsign] = max(0, first_day[callsign] - warm_up_end.hour)
        df.iloc[0] = first_day

        # Adjust for simulation end time
        last_day = df.iloc[-1]
        for callsign in rota['callsign']:
            if last_day[callsign] > 0:
                last_day[callsign] = min(last_day[callsign], sim_end.hour)
        df.iloc[-1] = last_day

        return df

    daily_available_hours = update_availability(df, rota=hems_rota)

    total_avail_hours = daily_available_hours.drop(columns=['is_summer']).sum(axis=0, numeric_only=True)

    total_avail_hours = pd.DataFrame(total_avail_hours)

    total_avail_hours.index.name = "callsign"

    # TODO: There is a mismatch here and need to investigate further where it's actually occurring
    # Fixing here for now, but will need to remove this if it's sorted elsewhere
    # total_avail_hours.index = total_avail_hours.index.str.replace("CC", "C")

    total_avail_hours.columns = ["total_available_hours_in_sim"]

    total_avail_hours = total_avail_hours.reset_index()
    total_avail_hours["callsign_group"] = total_avail_hours["callsign"].apply(lambda x: re.sub('\D', '', x))

    total_avail_minutes = total_avail_hours.copy()
    total_avail_minutes['total_available_hours_in_sim'] = total_avail_minutes['total_available_hours_in_sim'] * 60
    total_avail_minutes = total_avail_minutes.rename(columns={'total_available_hours_in_sim': 'total_available_minutes_in_sim'})

    return (daily_available_hours, total_avail_hours, total_avail_minutes)


def calculate_available_hours_v2(params_df,
    rota_data,#=pd.read_csv("../actual_data/HEMS_ROTA.csv"),
    service_data,#=pd.read_csv("../data/service_dates.csv"),
    callsign_data,
    output_by_month=False,
    long_format_df=False
):
    """
    Version of a function to calculate the number of hours a resource is available for use
    across the duration of the simulation, based on the rota used during the period, accounting
    for time during the simulation that uses the summer rota and time that uses the winter rota.

    Servicing is also taken into account.

    Warm up duration is taken into account.
    """
    # Convert data into DataFrames
    warm_up_end = _processing_functions.get_param("warm_up_end_date", params_df)
    warm_up_end = datetime.strptime(warm_up_end, "%Y-%m-%d %H:%M:%S")

    sim_end = _processing_functions.get_param("sim_end_date", params_df)
    sim_end = datetime.strptime(sim_end, "%Y-%m-%d %H:%M:%S")

#     hems_rota = pd.read_csv(rota_path)
    # service_dates = pd.read_csv(service_data_path)

    date_range = pd.date_range(start=warm_up_end.date(),
                            end=sim_end.date(),
                            freq='D')
    daily_df = pd.DataFrame({'date': date_range})

#     daily_df = pd.DataFrame(daily_data)
    rota_df = pd.DataFrame(rota_data)
    service_df = pd.DataFrame(service_data)

    callsign_df = pd.DataFrame(callsign_data)

    rota_df = rota_df.merge(callsign_df, on="callsign")
    service_df = service_df.merge(callsign_df, on="registration")

    # Convert date columns to datetime format
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    service_df['service_start_date'] = pd.to_datetime(service_df['service_start_date'])
    service_df['service_end_date'] = pd.to_datetime(service_df['service_end_date'])

    # Initialize dictionary to store results
    theoretical_availability = {}

    # Iterate over each row in the daily dataset
    for index, row in daily_df.iterrows():
        current_date = row['date']

        # Store theoretical available time for each resource
        day_data = {}

        for _, rota in rota_df.iterrows():
            callsign = rota['callsign']

            # Determine summer or winter schedule
            is_summer = current_date.month in range(3, 11)
            start_hour = rota['summer_start'] if is_summer else rota['winter_start']
            end_hour = rota['summer_end'] if is_summer else rota['winter_end']

            # Handle cases where the shift ends after midnight
            if end_hour < start_hour:
                daily_available_time = (24 - start_hour) + end_hour
            else:
                daily_available_time = end_hour - start_hour

            total_available_time = daily_available_time * 60  # Convert to minutes

            # Adjust for servicing periods
            service_downtime = 0
            for _, service in service_df[service_df['callsign'] == callsign].iterrows():
                if service['service_start_date'] <= current_date <= service['service_end_date']:
                    service_downtime = daily_available_time * 60
                    break

            # Final available time after accounting for servicing
            day_data[callsign] = total_available_time - service_downtime

        theoretical_availability[current_date.strftime('%Y-%m-%d')] = day_data

    theoretical_availability_df = pd.DataFrame(theoretical_availability).T
    theoretical_availability_df.index.name = "month"
    theoretical_availability_df = theoretical_availability_df.reset_index()
    # theoretical_availability_df = theoretical_availability_df.add_prefix("theoretical_availability_")

    theoretical_availability_df.fillna(0.0)

    theoretical_availability_df_long = (
        theoretical_availability_df
        .melt(id_vars="month")
        .rename(columns={"value":"theoretical_availability", "variable": "callsign"})
        )

    theoretical_availability_df_long['theoretical_availability'] = theoretical_availability_df_long['theoretical_availability'].astype('float')

    daily_available_minutes = theoretical_availability_df_long.copy()

    print("==Daily Available Minutes==")
    print(daily_available_minutes)

    total_avail_minutes = daily_available_minutes.groupby('callsign')[['theoretical_availability']].sum(numeric_only=True).reset_index().rename(columns={'theoretical_availability':'total_available_minutes_in_sim'})

    total_avail_minutes["callsign_group"] = total_avail_minutes["callsign"].apply(lambda x: re.sub('\D', '', x))

    if long_format_df:
        theoretical_availability_df_long
    else:
        return theoretical_availability_df, total_avail_minutes

def resource_allocation_outcomes(event_log_df):
    n_runs = len(event_log_df["run_number"].unique())
    resource_allocation_outcomes_df = (
        (event_log_df[event_log_df['event_type']=="resource_preferred_outcome"]
         .groupby(['time_type'])[['time_type']].count()/n_runs)
         .round(0).astype('int')
         .rename(columns={'time_type': 'Count'})
         .reset_index().rename(columns={'time_type': 'Resource Allocation Attempt Outcome'})
    )
    print("==_vehicle_calculation.py - resource_allocation_outcomes==")
    print(resource_allocation_outcomes_df)
    return resource_allocation_outcomes_df


def resource_allocation_outcomes_run_variation(event_log_df):
    n_runs = len(event_log_df["run_number"].unique())
    return (
        (event_log_df[event_log_df['event_type']=="resource_preferred_outcome"]
        .groupby(['time_type', 'run_number'])[['time_type']].count()/n_runs)
        .round(0).astype('int').rename(columns={'time_type': 'Count'})
        .reset_index().rename(columns={'time_type': 'Resource Allocation Attempt Outcome', 'run_number': "Run"})
    )

def get_perc_unattended_string(event_log_df):
    """
    Alternative to display_UNTATTENDED_calls_per_run

    This approach looks at instances where the resource allocation attempt outcome was
    'no resource in group available'
    """
    df = resource_allocation_outcomes(event_log_df)
    print("==get_perc_unattended_string - resource_allocation_outcomes==")
    print(df)
    try:
        num_unattendable = df[df["Resource Allocation Attempt Outcome"].str.contains("No HEMS resource available")]['Count'].sum()
        print(f"==get_perc_unattended_string - num_unattended: {num_unattendable}==")
    except:
        "Error"

    total_calls = df['Count'].sum()
    print(f"==get_perc_unattended_string - total calls: {total_calls}==")
    try:
        perc_unattendable = num_unattendable/total_calls

        if perc_unattendable < 0.01:
            return f"{num_unattendable} of {total_calls} (< 0.1%)"
        else:
            return f"{num_unattendable} of {total_calls} ({perc_unattendable:.1%})"
    except:
        return "Error"
