"""
File containing all calculations and visualisations relating to the vehicles.

- Simultaneous usage of different callsign groups
- Total available hours
- Servicing overrun
- Instances of being unable to lift
- Resource allocation hierarchies

Covers variation within the simulation, and comparison with real world data.
"""

import _processing_functions
from datetime import datetime
import pandas as pd

def calculate_available_hours(params_df, rota_path="../data/hems_rota_used.csv"):
    warm_up_end = _processing_functions.get_param("warm_up_end_date", params_df)
    warm_up_end = datetime.strptime(warm_up_end, "%Y-%m-%d %H:%M:%S")
    sim_end = _processing_functions.get_param("sim_end_date", params_df)
    sim_end = datetime.strptime(sim_end, "%Y-%m-%d %H:%M:%S")

    hems_rota = pd.read_csv(rota_path)


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

    return update_availability(df, hems_rota)
