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

# def calculate_available_hours(params_df,
#         rota_path="../actual_data/HEMS_ROTA.csv",
#         # service_data_path="../data/service_dates.csv"
#         ):
#     """
#     Version of a function to calculate the number of hours a resource is available for use
#     across the duration of the simulation, based on the rota used during the period, accounting
#     for time during the simulation that uses the summer rota and time that uses the winter rota.

#     NOTE: Servicing is not taken into account.

#     Warm up duration is taken into account.
#     """

#     warm_up_end = _processing_functions.get_param("warm_up_end_date", params_df)
#     warm_up_end = datetime.strptime(warm_up_end, "%Y-%m-%d %H:%M:%S")

#     sim_end = _processing_functions.get_param("sim_end_date", params_df)
#     sim_end = datetime.strptime(sim_end, "%Y-%m-%d %H:%M:%S")

#     hems_rota = pd.read_csv(rota_path)
#     # service_dates = pd.read_csv(service_data_path)

#     date_range = pd.date_range(start=warm_up_end.date(),
#                             end=sim_end.date(),
#                             freq='D')
#     df = pd.DataFrame({'date': date_range})

#     summer_start = datetime.strptime(_processing_functions.get_param("summer_start_date", params_df), "%Y-%m-%d")
#     winter_start = datetime.strptime(_processing_functions.get_param("winter_start_date", params_df), "%Y-%m-%d")

#     summer_start_month_day = pd.to_datetime(summer_start).strftime('%m-%d')
#     winter_start_month_day = pd.to_datetime(winter_start).strftime('%m-%d')

#     def is_summer(date):
#             year = date.year
#             summer_start_date = pd.to_datetime(f"{year}-{summer_start_month_day}")
#             winter_start_date = pd.to_datetime(f"{year}-{winter_start_month_day}")
#             return summer_start_date <= date < winter_start_date

#     df['is_summer'] = df['date'].apply(is_summer)

#     # Fill with a blank column for each callsign
#     for callsign in hems_rota.callsign:
#         df[callsign] = 0

#     def update_availability(df, rota):
#         df = df.copy()

#         for _, row in rota.iterrows():
#             callsign = row['callsign']
#             summer_start, winter_start = row['summer_start'], row['winter_start']
#             summer_end, winter_end = row['summer_end'], row['winter_end']

#             for i in range(len(df)):
#                 is_summer = df.at[i, 'is_summer']
#                 start_time = summer_start if is_summer else winter_start
#                 end_time = summer_end if is_summer else winter_end

#                 # Handle end time past midnight
#                 if end_time < start_time:
#                     duration = (24 - start_time) + end_time
#                     df.at[i, callsign] = duration

#                     if i + 1 < len(df):  # Add extra hours to next day
#                         df.at[i + 1, callsign] = df.at[i + 1, callsign] + end_time
#                 else:
#                     df.at[i, callsign] = end_time - start_time

#         # Adjust for simulation start time
#         first_day = df.iloc[0]
#         for callsign in rota['callsign']:
#             if first_day[callsign] > 0:
#                 first_day[callsign] = max(0, first_day[callsign] - warm_up_end.hour)
#         df.iloc[0] = first_day

#         # Adjust for simulation end time
#         last_day = df.iloc[-1]
#         for callsign in rota['callsign']:
#             if last_day[callsign] > 0:
#                 last_day[callsign] = min(last_day[callsign], sim_end.hour)
#         df.iloc[-1] = last_day

#         return df

#     daily_available_hours = update_availability(df, rota=hems_rota)

#     total_avail_hours = daily_available_hours.drop(columns=['is_summer']).sum(axis=0, numeric_only=True)

#     total_avail_hours = pd.DataFrame(total_avail_hours)

#     total_avail_hours.index.name = "callsign"

#     # TODO: There is a mismatch here and need to investigate further where it's actually occurring
#     # Fixing here for now, but will need to remove this if it's sorted elsewhere
#     # total_avail_hours.index = total_avail_hours.index.str.replace("CC", "C")

#     total_avail_hours.columns = ["total_available_hours_in_sim"]

#     total_avail_hours = total_avail_hours.reset_index()
#     total_avail_hours["callsign_group"] = total_avail_hours["callsign"].apply(lambda x: re.sub('\D', '', x))

#     total_avail_minutes = total_avail_hours.copy()
#     total_avail_minutes['total_available_hours_in_sim'] = total_avail_minutes['total_available_hours_in_sim'] * 60
#     total_avail_minutes = total_avail_minutes.rename(columns={'total_available_hours_in_sim': 'total_available_minutes_in_sim'})

#     return (daily_available_hours, total_avail_hours, total_avail_minutes)


def calculate_available_hours_v2(params_df,
    rota_data,#=pd.read_csv("../actual_data/HEMS_ROTA.csv"),
    service_data,#=pd.read_csv("../data/service_dates.csv"),
    callsign_data,
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

    date_range = pd.date_range(start=warm_up_end.date(),
                            end=sim_end.date(),
                            freq='D')
    daily_df = pd.DataFrame({'date': date_range})

    rota_df = pd.DataFrame(rota_data)
    service_df = pd.DataFrame(service_data)

    callsign_df = pd.DataFrame(callsign_data)

    rota_df = rota_df.merge(callsign_df, on="callsign")
    service_df = service_df.merge(callsign_df, on="registration")

    # Convert date columns to datetime format
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    service_df['service_start_date'] = pd.to_datetime(service_df['service_start_date'])
    service_df['service_end_date'] = pd.to_datetime(service_df['service_end_date'])

    def is_summer(date_obj):
        return current_date.month in [4,5,6,7,8,9]

    # Initialize columns in df_availability for each unique callsign
    for callsign in rota_df['callsign'].unique():
        daily_df[callsign] = 0 # Initialize with 0 minutes

    daily_df = daily_df.set_index('date')

    # Iterate through each date in our availability dataframe
    for date_idx, current_date in enumerate(daily_df.index):
        is_current_date_summer = is_summer(current_date)

        # Iterate through each resource's rota entry
        for _, rota_entry in rota_df.iterrows():
            callsign = rota_entry['callsign']
            start_hour_col = 'summer_start' if is_current_date_summer else 'winter_start'
            end_hour_col = 'summer_end' if is_current_date_summer else 'winter_end'

            start_hour = rota_entry[start_hour_col]
            end_hour = rota_entry[end_hour_col]

            # --- Calculate minutes for the current_date ---
            minutes_for_callsign_on_date = 0

            # Scenario 1: Shift is fully within one day (e.g., 7:00 to 19:00)
            if start_hour < end_hour:
                # Check if this shift is active on current_date (it always is in this logic,
                # as we are calculating for the current_date based on its rota)
                minutes_for_callsign_on_date = (end_hour - start_hour) * 60
            # Scenario 2: Shift spans midnight (e.g., 19:00 to 02:00)
            elif start_hour > end_hour:
                # Part 1: Minutes from start_hour to midnight on current_date
                minutes_today = (24 - start_hour) * 60
                minutes_for_callsign_on_date += minutes_today

                # Part 2: Minutes from midnight to end_hour on the *next* day
                # These minutes need to be added to the *next day's* total for this callsign
                if date_idx + 1 < len(daily_df): # Ensure there is a next day in our df
                    next_date = daily_df.index[date_idx + 1]
                    minutes_on_next_day = end_hour * 60
                    daily_df.loc[next_date, callsign] = daily_df.loc[next_date, callsign] + minutes_on_next_day

            daily_df.loc[current_date, callsign] += minutes_for_callsign_on_date

    theoretical_availability_df = daily_df
    theoretical_availability_df.index.name = "month"
    theoretical_availability_df = theoretical_availability_df.reset_index()

    theoretical_availability_df.fillna(0.0)

    theoretical_availability_df.to_csv("data/daily_availability.csv", index=False)

    theoretical_availability_df["ms"] = theoretical_availability_df["month"].dt.strftime('%Y-%m-01')
    theoretical_availability_df.groupby('ms').sum(numeric_only=True).to_csv("data/monthly_availability.csv")
    theoretical_availability_df.drop(columns=["ms"], inplace=True)

    theoretical_availability_df_long = (
        theoretical_availability_df
        .melt(id_vars="month")
        .rename(columns={"value":"theoretical_availability", "variable": "callsign"})
        )

    theoretical_availability_df_long['theoretical_availability'] = theoretical_availability_df_long['theoretical_availability'].astype('float')

    daily_available_minutes = theoretical_availability_df_long.copy()

    # print("==Daily Available Minutes==")
    # print(daily_available_minutes)

    total_avail_minutes = daily_available_minutes.groupby('callsign')[['theoretical_availability']].sum(numeric_only=True).reset_index().rename(columns={'theoretical_availability':'total_available_minutes_in_sim'})

    total_avail_minutes["callsign_group"] = total_avail_minutes["callsign"].apply(lambda x: re.sub('\D', '', x))

    total_avail_minutes.to_csv("data/daily_availability_summary.csv", index=False)

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
    # print("==_vehicle_calculation.py - resource_allocation_outcomes==")
    # print(resource_allocation_outcomes_df)
    return resource_allocation_outcomes_df


def resource_allocation_outcomes_run_variation(event_log_df):
    n_runs = len(event_log_df["run_number"].unique())
    return (
        (event_log_df[event_log_df['event_type']=="resource_preferred_outcome"]
        .groupby(['time_type', 'run_number'])[['time_type']].count()/n_runs)
        .round(0).astype('int').rename(columns={'time_type': 'Count'})
        .reset_index().rename(columns={'time_type': 'Resource Allocation Attempt Outcome', 'run_number': "Run"})
    )

# def get_perc_unattended_string(event_log_df):
#     """
#     Alternative to display_UNTATTENDED_calls_per_run

#     This approach looks at instances where the resource allocation attempt outcome was
#     'no resource in group available'
#     """
#     df = resource_allocation_outcomes(event_log_df)
#     print("==get_perc_unattended_string - resource_allocation_outcomes==")
#     print(df)
#     try:
#         num_unattendable = df[df["Resource Allocation Attempt Outcome"].str.contains("No HEMS resource available")]['Count'].sum()
#         print(f"==get_perc_unattended_string - num_unattended: {num_unattendable}==")
#     except:
#         "Error"

#     total_calls = df['Count'].sum()
#     print(f"==get_perc_unattended_string - total calls: {total_calls}==")
#     try:
#         perc_unattendable = num_unattendable/total_calls

#         if perc_unattendable < 0.01:
#             return f"{num_unattendable} of {total_calls} (< 0.1%)"
#         else:
#             return f"{num_unattendable} of {total_calls} ({perc_unattendable:.1%})"
#     except:
#         return "Error"


def get_perc_unattended_string(event_log_df):
    """
    Alternative to display_UNTATTENDED_calls_per_run

    This approach looks at instances where the resource_request_outcome
    was 'no resource available'
    """
    # event_log_df = pd.read_csv("data/run_results.csv")
    try:
        num_unattendable = len(event_log_df[
            (event_log_df["event_type"] == "resource_request_outcome") &
            (event_log_df["time_type"] == "No Resource Available")
            ])

        # print(f"==get_perc_unattended_string - num_unattended: {num_unattendable}==")
    except:
        "Error"

    total_calls = len(event_log_df[
            (event_log_df["event_type"] == "resource_request_outcome")
            ])

    # print(f"==get_perc_unattended_string - total calls: {total_calls}==")

    try:
        perc_unattendable = num_unattendable/total_calls

        if perc_unattendable < 0.01:
            return f"{num_unattendable} of {total_calls} (< 0.1%)"
        else:
            return f"{num_unattendable} of {total_calls} ({perc_unattendable:.1%})"
    except:
        return "Error"


def get_perc_unattended_string_normalised(event_log_df, params_df="data/run_params_used.csv"):
    """
    Alternative to display_UNTATTENDED_calls_per_run

    This approach looks at instances where the resource_request_outcome
    was 'no resource available'
    """
    # event_log_df = pd.read_csv("data/run_results.csv")
    try:
        num_unattendable = len(event_log_df[
            (event_log_df["event_type"] == "resource_request_outcome") &
            (event_log_df["time_type"] == "No Resource Available")
            ])

        # print(f"==get_perc_unattended_string - num_unattended: {num_unattendable}==")
    except:
        "Error"

    total_calls = len(event_log_df[
            (event_log_df["event_type"] == "resource_request_outcome")
            ])

    # print(f"==get_perc_unattended_string - total calls: {total_calls}==")

    try:
        num_runs = len(event_log_df["run_number"].unique()) # More reliable than taking max run number if zero indexed
        sim_duration_mins = float(_processing_functions.get_param("sim_duration", pd.read_csv(params_df)))
        sim_duration_days = sim_duration_mins / 24 / 60

        perc_unattendable = num_unattendable/total_calls

        if perc_unattendable < 0.01:
            return f"{(num_unattendable/num_runs):.0f} of {(total_calls/num_runs):.0f} (< 0.1%)", f"This equates to around {((num_unattendable/num_runs/sim_duration_days)*365):.0f} of {((total_calls/num_runs/sim_duration_days)*365):.0f} calls per year"
        else:
            return f"{(num_unattendable/num_runs):.0f} of {(total_calls/num_runs):.0f} ({perc_unattendable:.1%})", f"This equates to around {((num_unattendable/num_runs/sim_duration_days)*365):.0f} of {((total_calls/num_runs/sim_duration_days)*365):.0f} calls per year"
    except:
        return "Error"
