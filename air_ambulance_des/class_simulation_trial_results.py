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

===

File containing all calculations and visualisations arelating to the patient and job outcomes/
results.

- Stand-down rates
- Patient outcomes

Covers variation within the simulation, and comparison with real world data.

===

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

===

File containing all calculations and visualisations relating to the vehicles.

[] Simultaneous usage of different callsign groups
[] Total available hours
[] Servicing overrun
[] Instances of being unable to lift
[] Resource allocation hierarchies

Covers variation within the simulation, and comparison with real world data.

===

File containing all calculations and visualisations arelating to the patients.

[] Split by AMPDS card
[] Patient Ages
[] Patient Genders
[] Patient Ages by AMPDS
[] Patient Genders by AMPDS

Covers variation within the simulation, and comparison with real world data.


"""

import pandas as pd
import numpy as np
import re

import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

import gc

import os
import sys

import datetime
from calendar import monthrange, day_name
import itertools
import textwrap

from scipy.stats import ks_2samp

import streamlit as st

from air_ambulance_des._processing_functions import (
    graceful_methods,
    get_param,
    fill_missing_values,
    calculate_time_difference,
)

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from air_ambulance_des.utils import (
    COLORSCHEME,
    q10,
    q90,
    q25,
    q75,
    format_sigfigs,
    iconMetricContainer,
)


@graceful_methods
class TrialResults:
    """
    Manages results from a model trial
    """

    def __init__(
        self,
        simulation_inputs,
        run_results: pd.DataFrame,
        daily_availability_data: pd.DataFrame,
        historical_data=None,
    ):
        self.run_results = run_results

        self.resource_use_events_only_df = self.run_results[
            self.run_results["event_type"].isin(["resource_use", "resource_use_end"])
        ].copy()

        self.historical_data = historical_data

        self.n_runs = len(self.run_results["run_number"].unique())

        self.simulation_inputs = simulation_inputs

        self.params_df = self.simulation_inputs.params_df

        self.call_df = None
        self.hourly_calls_per_run = None
        self.daily_calls_per_run = None

        self.theoretical_availability_df_long = None
        self.theoretical_availability_df = None
        self.total_avail_minutes = None

        # Placeholders for event duration dataframes
        self.simulation_event_duration_df = None
        self.simulation_event_duration_df_summary = None

        # Placeholders for utilisation dataframes
        # self.utilisation_model_df = None
        self.resource_use_wide = None
        self.utilisation_df_overall = None
        self.utilisation_df_per_run = None
        self.utilisation_df_per_run_by_csg = None

        # Placeholders for vehicle dataframes
        self.resource_allocation_outcomes_df = None

        self.missed_jobs_per_run_breakdown = None
        self.missed_jobs_per_run_care_cat_summary = None

        self.sim_averages_utilisation = None

        self.daily_availability_df = daily_availability_data

        self.event_counts_df = None
        self.event_counts_long = None

        # Set fixed order for months
        self.month_order = (
            pd.date_range("2000-01-01", periods=12, freq="MS").strftime("%B").tolist()
        )

        # Create a lookup dict to ensure formatting of weekday is consistent across simulated
        # and historical datasets
        self.day_dict = {
            "Mon": "Monday",
            "Tue": "Tuesday",
            "Wed": "Wednesday",
            "Thu": "Thursday",
            "Fri": "Friday",
            "Sat": "Saturday",
            "Sun": "Sunday",
        }

        # Create a list to ensure days of the week are displayed in the correct order in plots
        self.day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        # Add extra useful columns to run results
        self.enhance_run_results()
        # Add extra useful columns to the daily availability data
        self.update_daily_availability_df()
        # Create self.call_df
        self.make_job_count_df()
        # Create self.hourly_calls_per_run
        self.get_hourly_calls_per_run()
        # Create self.daily_calls_per_run
        self.get_daily_calls_per_run()
        # Create simulation event duration breakdown and summary
        self.create_simulation_event_duration_df()
        self.summarise_event_times()
        # Create event_counts_df and event_counts_long
        self.get_event_counts()
        # Create theoretical available resource time dataframe
        self.calculate_available_hours(
            summer_start=int(
                self.simulation_inputs.rota_start_end_months_df[
                    self.simulation_inputs.rota_start_end_months_df["what"]
                    == "summer_start_month"
                ]["month"].values[0]
            ),
            summer_end=int(
                self.simulation_inputs.rota_start_end_months_df[
                    self.simulation_inputs.rota_start_end_months_df["what"]
                    == "summer_end_month"
                ]["month"].values[0]
            ),
        )

        self.make_utilisation_model_dataframe()

        self.get_missed_call_df()
        self.prep_util_df_from_call_df()

    def enhance_run_results(self):
        self.run_results["timestamp_dt"] = pd.to_datetime(
            self.run_results["timestamp_dt"], format="ISO8601"
        )

        # Extract month from datetime
        self.run_results["month"] = self.run_results["timestamp_dt"].dt.strftime(
            "%B"
        )  # e.g., 'January', 'February'

        self.run_results["month"] = pd.Categorical(
            self.run_results["month"], categories=self.month_order, ordered=True
        )

        try:
            self.run_results = self.run_results.drop(columns=["Unnamed: 0"])
        except KeyError:
            pass

    def resource_allocation_outcomes(self):
        self.resource_allocation_outcomes_df = (
            (
                self.run_results[
                    self.run_results["event_type"] == "resource_preferred_outcome"
                ]
                .groupby(["time_type"])[["time_type"]]
                .count()
                / self.n_runs
            )
            .round(0)
            .astype("int")
            .rename(columns={"time_type": "Count"})
            .reset_index()
            .rename(columns={"time_type": "Resource Allocation Attempt Outcome"})
        ).copy()

    def update_daily_availability_df(self):
        self.daily_availability_df = self.daily_availability_df.melt(
            id_vars="month"
        ).rename(
            columns={
                "value": "theoretical_availability",
                "variable": "callsign",
            }
        )

    def calculate_available_hours(self, summer_start, summer_end):
        """
        Version of a function to calculate the number of hours a resource is available for use
        across the duration of the simulation, based on the rota used during the period, accounting
        for time during the simulation that uses the summer rota and time that uses the winter rota.

        Servicing is also taken into account.

        Warm up duration is taken into account.
        """
        # Convert data into DataFrames
        warm_up_end = get_param("warm_up_end_date", self.params_df)
        warm_up_end = datetime.datetime.strptime(warm_up_end, "%Y-%m-%d %H:%M:%S")

        sim_end = get_param("sim_end_date", self.params_df)
        sim_end = datetime.datetime.strptime(sim_end, "%Y-%m-%d %H:%M:%S")

        date_range = pd.date_range(
            start=warm_up_end.date(), end=sim_end.date(), freq="D"
        )
        daily_df = pd.DataFrame({"date": date_range})

        rota_df = self.simulation_inputs.rota_df
        service_df = self.simulation_inputs.service_dates_df

        callsign_df = self.simulation_inputs.callsign_registration_lookup_df

        rota_df = rota_df.merge(callsign_df, on="callsign")
        service_df = service_df.merge(callsign_df, on="registration")

        # Convert date columns to datetime format
        daily_df["date"] = pd.to_datetime(daily_df["date"])

        service_df["service_start_date"] = pd.to_datetime(
            service_df["service_start_date"]
        )
        service_df["service_end_date"] = pd.to_datetime(service_df["service_end_date"])

        def is_summer(date_obj):
            # return date_obj.month in [4,5,6,7,8,9]
            return date_obj.month in [i for i in range(summer_start, summer_end + 1)]

        # Initialize columns in df_availability for each unique callsign
        for callsign in rota_df["callsign"].unique():
            daily_df[callsign] = 0  # Initialize with 0 minutes

        daily_df = daily_df.set_index("date")

        # Iterate through each date in our availability dataframe
        for date_idx, current_date in enumerate(daily_df.index):
            is_current_date_summer = is_summer(current_date)

            # Iterate through each resource's rota entry
            for _, rota_entry in rota_df.iterrows():
                callsign = rota_entry["callsign"]
                start_hour_col = (
                    "summer_start" if is_current_date_summer else "winter_start"
                )
                end_hour_col = "summer_end" if is_current_date_summer else "winter_end"

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
                    if date_idx + 1 < len(
                        daily_df
                    ):  # Ensure there is a next day in our df
                        next_date = daily_df.index[date_idx + 1]
                        minutes_on_next_day = end_hour * 60

                        daily_df.loc[next_date, callsign] = (
                            daily_df.loc[next_date, callsign] + minutes_on_next_day
                        )

                daily_df.loc[current_date, callsign] += minutes_for_callsign_on_date

        self.theoretical_availability_df = daily_df
        self.theoretical_availability_df.index.name = "month"
        self.theoretical_availability_df = (
            self.theoretical_availability_df.reset_index()
        )

        self.theoretical_availability_df.fillna(0.0)

        self.theoretical_availability_df.to_csv(
            "data/daily_availability.csv", index=False
        )

        self.theoretical_availability_df["ms"] = self.theoretical_availability_df[
            "month"
        ].dt.strftime("%Y-%m-01")

        self.theoretical_availability_df.groupby("ms").sum(numeric_only=True).to_csv(
            "data/monthly_availability.csv"
        )

        self.theoretical_availability_df.drop(columns=["ms"], inplace=True)

        self.theoretical_availability_df_long = self.theoretical_availability_df.melt(
            id_vars="month"
        ).rename(columns={"value": "theoretical_availability", "variable": "callsign"})

        self.theoretical_availability_df_long["theoretical_availability"] = (
            self.theoretical_availability_df_long["theoretical_availability"].astype(
                "float"
            )
        )

        daily_available_minutes = self.theoretical_availability_df_long.copy()

        # print("==Daily Available Minutes==")
        # print(daily_available_minutes)

        self.total_avail_minutes = (
            daily_available_minutes.groupby("callsign")[["theoretical_availability"]]
            .sum(numeric_only=True)
            .reset_index()
            .rename(
                columns={"theoretical_availability": "total_available_minutes_in_sim"}
            )
        )

        self.total_avail_minutes["callsign_group"] = self.total_avail_minutes[
            "callsign"
        ].apply(lambda x: re.sub("\D", "", x))

        self.total_avail_minutes.to_csv(
            "data/daily_availability_summary.csv", index=False
        )

    def make_utilisation_model_dataframe(self):
        # Restrict to only events in the event log where resource use was starting or ending
        resource_use_only = self.run_results[
            self.run_results["event_type"].isin(["resource_use", "resource_use_end"])
        ].copy()

        # Pivot to wide-format dataframe with one row per patient/call
        # and columns for start and end types
        self.resource_use_wide = (
            resource_use_only[
                [
                    "P_ID",
                    "run_number",
                    "event_type",
                    "timestamp_dt",
                    "callsign_group",
                    "vehicle_type",
                    "callsign",
                ]
            ]
            .pivot(
                index=[
                    "P_ID",
                    "run_number",
                    "callsign_group",
                    "vehicle_type",
                    "callsign",
                ],
                columns="event_type",
                values="timestamp_dt",
            )
            .reset_index()
        )

        del resource_use_only

        # If utilisation end date is missing then set to end of model
        # as we can assume this is a call that didn't finish before the model did
        self.resource_use_wide = fill_missing_values(
            self.resource_use_wide,
            "resource_use_end",
            get_param("sim_end_date", self.params_df),
        )

        # If utilisation start time is missing, then set to start of model + warm-up time (if relevant)
        # as can assume this is a call that started before the warm-up period elapsed but finished
        # after the warm-up period elapsed
        # TODO: need to add in a check to ensure this only happens for calls at the end of the model,
        # not due to errors elsewhere that could fail to assign a resource end time
        self.resource_use_wide = fill_missing_values(
            self.resource_use_wide,
            "resource_use",
            get_param("warm_up_end_date", self.params_df),
        )

        # Calculate number of minutes the attending resource was in use on each call
        self.resource_use_wide["resource_use_duration"] = calculate_time_difference(
            self.resource_use_wide,
            "resource_use",
            "resource_use_end",
            unit="minutes",
        )

        # ============================================================ #
        # Calculage per-run utilisation,
        # stratified by callsign and vehicle type (car/helicopter)
        # ============================================================ #
        self.utilisation_df_per_run = self.resource_use_wide.groupby(
            ["run_number", "vehicle_type", "callsign"]
        )[["resource_use_duration"]].sum()

        # Join with df of how long each resource was available for in the sim
        # We will for now assume this is the same across each run
        self.utilisation_df_per_run = self.utilisation_df_per_run.reset_index(
            drop=False
        ).merge(self.total_avail_minutes, on="callsign", how="left")

        self.utilisation_df_per_run["perc_time_in_use"] = (
            self.utilisation_df_per_run["resource_use_duration"].astype(float)
            /
            # float(_processing_functions.get_param("sim_duration", params_df))
            self.utilisation_df_per_run["total_available_minutes_in_sim"].astype(float)
        )

        # Add column of nicely-formatted values to make printing values more streamlined
        self.utilisation_df_per_run["PRINT_perc"] = self.utilisation_df_per_run[
            "perc_time_in_use"
        ].apply(lambda x: f"{x:.1%}")

        # ============================================================ #
        # Calculage averge utilisation across simulation,
        # stratified by callsign group
        # ============================================================ #

        self.utilisation_df_per_run_by_csg = self.resource_use_wide.groupby(
            ["callsign_group"]
        )[["resource_use_duration"]].sum()

        self.utilisation_df_per_run_by_csg["resource_use_duration"] = (
            self.utilisation_df_per_run_by_csg["resource_use_duration"] / self.n_runs
        )

        self.utilisation_df_per_run_by_csg = (
            self.utilisation_df_per_run_by_csg.reset_index()
        )

        total_avail_minutes_per_csg = (
            self.total_avail_minutes.groupby("callsign_group")
            .head(1)
            .drop(columns="callsign")
        )

        total_avail_minutes_per_csg["callsign_group"] = total_avail_minutes_per_csg[
            "callsign_group"
        ].astype("float")

        self.utilisation_df_per_run_by_csg = self.utilisation_df_per_run_by_csg.merge(
            total_avail_minutes_per_csg, on="callsign_group", how="left"
        )

        self.utilisation_df_per_run_by_csg["perc_time_in_use"] = (
            self.utilisation_df_per_run_by_csg["resource_use_duration"].astype(float)
            /
            # float(_processing_functions.get_param("sim_duration", params_df))
            self.utilisation_df_per_run_by_csg["total_available_minutes_in_sim"].astype(
                float
            )
        )

        self.utilisation_df_per_run_by_csg["PRINT_perc"] = (
            self.utilisation_df_per_run_by_csg["perc_time_in_use"].apply(
                lambda x: f"{x:.1%}"
            )
        )

        # ============================================================ #
        # Calculage averge utilisation across simulation,
        # stratified by callsign and vehicle type (car/helicopter)
        # ============================================================ #
        self.utilisation_df_overall = self.utilisation_df_per_run.groupby(
            ["callsign", "vehicle_type"]
        )[["resource_use_duration"]].sum()

        self.utilisation_df_overall["resource_use_duration"] = (
            self.utilisation_df_overall["resource_use_duration"] / self.n_runs
        )

        self.utilisation_df_overall = self.utilisation_df_overall.reset_index(
            drop=False
        ).merge(self.total_avail_minutes, on="callsign", how="left")

        self.utilisation_df_overall["perc_time_in_use"] = (
            self.utilisation_df_overall["resource_use_duration"].astype(float)
            /
            # float(_processing_functions.get_param("sim_duration", params_df))
            self.utilisation_df_overall["total_available_minutes_in_sim"].astype(float)
        )

        # Add column of nicely-formatted values to make printing values more streamlined
        self.utilisation_df_overall["PRINT_perc"] = self.utilisation_df_overall[
            "perc_time_in_use"
        ].apply(lambda x: f"{x:.1%}")

    def make_job_count_df(self):
        """
        Given the event log produced by running the model, create a dataframe with one row per
        patient, but all pertinent information about each call added to that row if it would not
        usually be present until a later entry in the log
        """
        # hems_result and outcome columns aren't determined until a later step
        # backfill this per patient/run so we'll have access to it from the row for
        # the patient's arrival
        run_results_bfilled = self.run_results.copy()
        # print("==run results bfilled==")
        # print(run_results_bfilled.head())
        # print(run_results_bfilled.columns)

        if "P_ID" not in run_results_bfilled.columns:
            run_results_bfilled = run_results_bfilled.reset_index()

        run_results_bfilled["hems_result"] = run_results_bfilled.groupby(
            ["P_ID", "run_number"]
        ).hems_result.bfill()
        run_results_bfilled["outcome"] = run_results_bfilled.groupby(
            ["P_ID", "run_number"]
        ).outcome.bfill()
        # same for various things around allocated resource
        run_results_bfilled["vehicle_type"] = run_results_bfilled.groupby(
            ["P_ID", "run_number"]
        ).vehicle_type.bfill()
        run_results_bfilled["callsign"] = run_results_bfilled.groupby(
            ["P_ID", "run_number"]
        ).callsign.bfill()
        run_results_bfilled["registration"] = run_results_bfilled.groupby(
            ["P_ID", "run_number"]
        ).registration.bfill()

        # TODO - see what we can do about any instances where these columns remain NA
        # Think this is likely to relate to instances where there was no resource available?
        # Would be good to populate these columns with a relevant indicator if that's the case

        # Reduce down to just the 'arrival' row for each patient, giving us one row per patient
        # per run
        self.call_df = run_results_bfilled[
            run_results_bfilled["time_type"] == "arrival"
        ].drop(columns=["time_type", "event_type"])

        self.call_df.to_csv("data/call_df.csv", index=False)

        self.call_df["timestamp_dt"] = pd.to_datetime(
            self.call_df["timestamp_dt"], format="ISO8601"
        )
        self.call_df["month_start"] = (
            self.call_df["timestamp_dt"].dt.to_period("M").dt.to_timestamp()
        )

        self.call_df["day_date"] = pd.to_datetime(self.call_df["timestamp_dt"]).dt.date

    def get_daily_calls_per_run(self):
        # Calculate the number of calls per day across a full run
        self.daily_calls_per_run = (
            self.call_df.groupby(["day", "run_number"])[["P_ID"]]
            .count()
            .reset_index()
            .rename(columns={"P_ID": "count"})
        )

        # Ensure the days in the simulated dataset are formatted the same as the historical dataset
        self.daily_calls_per_run["day"] = self.daily_calls_per_run["day"].apply(
            lambda x: self.day_dict[x]
        )

    def get_calls_per_run(self):
        """
        Returns a series of the calls per simulation run
        """
        return self.call_df.groupby("run_number")[["P_ID"]].count().reset_index()

    def get_AVERAGE_calls_per_run(self):
        """
        Returns a count of the calls per run, averaged across all runs
        """
        calls_per_run = self.get_calls_per_run(self.call_df)
        return calls_per_run.mean()["P_ID"].round(2)

    def get_UNATTENDED_calls_per_run(self):
        """
        Returns a count of the unattended calls per run

        This is done by looking for any instances where no callsign was assigned, indicating that
        no resource was sent
        """
        return (
            self.call_df[self.call_df["callsign"].isna()]
            .groupby("run_number")[["P_ID"]]
            .count()
            .reset_index()
        )

    def get_AVERAGE_UNATTENDED_calls_per_run(self):
        """
        Returns a count of the calls per run, averaged across all runs

        This is done by looking for any instances where no callsign was assigned, indicating that
        no resource was sent
        """
        unattended_calls_per_run = self.get_UNATTENDED_calls_per_run(self.call_df)
        return unattended_calls_per_run.mean()["P_ID"].round(2)

    def display_UNTATTENDED_calls_per_run(self):
        """
        Alternative to get_perc_unattended_string(), using a different approach, allowing for
        robustness testing

        Here, this is done by looking for any instances where no callsign was assigned, indicating that
        no resource was sent
        """
        total_calls = self.get_AVERAGE_calls_per_run(self.call_df)
        unattended_calls = self.get_AVERAGE_UNATTENDED_calls_per_run(self.call_df)

        return f"{unattended_calls:.0f} of {total_calls:.0f} ({(unattended_calls / total_calls):.1%})"

    def get_hourly_calls_per_run(self):
        self.hourly_calls_per_run = (
            self.call_df.groupby(["hour", "run_number"])[["P_ID"]]
            .count()
            .reset_index()
            .rename(columns={"P_ID": "count"})
        )

    def PLOT_hourly_call_counts(
        self,
        box_plot=False,
        average_per_month=False,
        bar_colour="teal",
        title="Calls Per Hour",
        use_poppins=False,
        error_bar_colour="charcoal",
        show_error_bars_bar=True,
        show_historical=True,
    ) -> Figure:
        """
        Produces an interactive plot showing the number of calls that were received per hour in
        the simulation

        This can be compared with the processed historical data used to inform the simulation
        """

        fig = go.Figure()

        if show_historical:
            jobs_per_hour_historic = (
                self.historical_data.historical_monthly_totals_by_hour_of_day.copy()
            )

            jobs_per_hour_historic["month"] = pd.to_datetime(
                jobs_per_hour_historic["month"], format="ISO8601"
            )
            jobs_per_hour_historic["year_numeric"] = jobs_per_hour_historic[
                "month"
            ].apply(lambda x: x.year)
            jobs_per_hour_historic["month_numeric"] = jobs_per_hour_historic[
                "month"
            ].apply(lambda x: x.month)
            jobs_per_hour_historic_long = jobs_per_hour_historic.melt(
                id_vars=["month", "month_numeric", "year_numeric"]
            )
            # jobs_per_hour_historic_long["hour"] = jobs_per_hour_historic_long['variable'].str.extract(r"(\d+)\s")
            jobs_per_hour_historic_long.rename(
                columns={"variable": "hour"}, inplace=True
            )
            jobs_per_hour_historic_long["hour"] = jobs_per_hour_historic_long[
                "hour"
            ].astype("int")
            jobs_per_hour_historic_long = jobs_per_hour_historic_long[
                ~jobs_per_hour_historic_long["value"].isna()
            ]

            if not average_per_month:
                jobs_per_hour_historic_long["value"] = jobs_per_hour_historic_long[
                    "value"
                ] * (float(get_param("sim_duration", self.params_df)) / 60 / 24 / 30)

            jobs_per_hour_historic_agg = (
                jobs_per_hour_historic_long.groupby(["hour"])["value"].agg(
                    ["min", "max", q10, q90]
                )
            ).reset_index()

            fig.add_trace(
                go.Bar(
                    x=jobs_per_hour_historic_agg["hour"],
                    y=jobs_per_hour_historic_agg["max"]
                    - jobs_per_hour_historic_agg["min"],  # The range
                    base=jobs_per_hour_historic_agg["min"],  # Starts from the minimum
                    name="Historical Range",
                    marker_color="rgba(100, 100, 255, 0.2)",  # Light blue with transparency
                    hoverinfo="skip",  # Hide hover info for clarity
                    showlegend=True,
                    width=1.0,  # Wider bars to make them contiguous
                    offsetgroup="historical",  # Grouping ensures alignment
                )
            )

            fig.add_trace(
                go.Bar(
                    x=jobs_per_hour_historic_agg["hour"],
                    y=jobs_per_hour_historic_agg["q90"]
                    - jobs_per_hour_historic_agg["q10"],  # The range
                    base=jobs_per_hour_historic_agg["q10"],  # Starts from the minimum
                    name="Historical 80% Range",
                    marker_color="rgba(100, 100, 255, 0.3)",  # Light blue with transparency
                    hoverinfo="skip",  # Hide hover info for clarity
                    showlegend=True,
                    width=1.0,  # Wider bars to make them contiguous
                    offsetgroup="historical",  # Grouping ensures alignment
                )
            )

            fig.update_layout(
                xaxis=dict(dtick=1),
                barmode="overlay",  # Ensures bars overlay instead of stacking
                title="Comparison of Simulated and Historical Call Counts",
            )

        if box_plot:
            if average_per_month:
                self.hourly_calls_per_run["average_per_day"] = (
                    self.hourly_calls_per_run["count"]
                    / (float(get_param("sim_duration", self.params_df)) / 60 / 24 / 30)
                )
                y_column = "average_per_day"
                y_label = "Average Monthly Calls Per Hour Across Simulation<br>Averaged Across Simulation Runs"
            else:
                y_column = "count"
                y_label = "Total Calls Per Hour Across Simulation<br>Averaged Across Simulation Runs"

            # Add box plot trace
            fig.add_trace(
                go.Box(
                    x=self.hourly_calls_per_run["hour"],
                    y=self.hourly_calls_per_run[y_column],
                    name="Simulated Mean",
                    width=0.4,
                    marker=dict(color=COLORSCHEME[bar_colour]),
                    showlegend=True,
                    boxpoints="outliers",  # Show all data points
                )
            )

            # Update layout
            fig.update_layout(
                title=title,
                xaxis=dict(title="Hour", dtick=1),
                yaxis=dict(title=y_label),
            )

        else:
            # Create required dataframe for simulation output display
            aggregated_data = (
                self.hourly_calls_per_run.groupby("hour")
                .agg(
                    mean_count=("count", "mean"),
                    # std_count=("count", "std")
                    se_count=(
                        "count",
                        lambda x: x.std() / np.sqrt(len(x)),
                    ),  # Standard Error
                )
                .reset_index()
            )

            if show_error_bars_bar:
                # error_y = aggregated_data["std_count"]
                error_y = aggregated_data["se_count"]
            else:
                error_y = None

            if average_per_month:
                aggregated_data["mean_count"] = aggregated_data["mean_count"] / (
                    float(get_param("sim_duration", self.params_df)) / 60 / 24 / 30
                )
                # aggregated_data['std_count'] = aggregated_data['std_count'] / (float(_processing_functions.get_param("sim_duration", params_df))/60/24/ 30)
                aggregated_data["se_count"] = aggregated_data["se_count"] / (
                    float(get_param("sim_duration", self.params_df)) / 60 / 24 / 30
                )

                fig.add_trace(
                    go.Bar(
                        x=aggregated_data["hour"],
                        y=aggregated_data["mean_count"],
                        name="Simulated Mean",
                        marker=dict(
                            color=COLORSCHEME[bar_colour]
                        ),  # Use your color scheme
                        error_y=dict(type="data", array=error_y, visible=True)
                        if error_y is not None
                        else None,
                        width=0.4,  # Narrower bars in front
                        offsetgroup="simulated",
                    )
                )

                fig.update_layout(
                    xaxis=dict(dtick=1),
                    barmode="overlay",  # Ensures bars overlay instead of stacking
                    title=title,
                    yaxis_title="Average Monthly Calls Per Hour Across Simulation<br>Averaged Across Simulation Runs",
                    xaxis_title="Hour",
                )

            else:
                fig.add_trace(
                    go.Bar(
                        x=aggregated_data["hour"],
                        y=aggregated_data["mean_count"],
                        name="Simulated Mean",
                        marker=dict(
                            color=COLORSCHEME[bar_colour]
                        ),  # Use your color scheme
                        error_y=dict(type="data", array=error_y, visible=True)
                        if error_y is not None
                        else None,
                        width=0.4,  # Narrower bars in front
                        offsetgroup="simulated",
                    )
                )

                fig.update_layout(
                    xaxis=dict(dtick=1),
                    barmode="overlay",  # Ensures bars overlay instead of stacking
                    title=title,
                    yaxis_title="Total Calls Per Hour Across Simulation",
                    xaxis_title="Hour",
                )

        if not box_plot:
            fig = fig.update_traces(error_y_color=COLORSCHEME[error_bar_colour])

        if use_poppins:
            return fig.update_layout(
                font=dict(family="Poppins", size=18, color="black")
            )
        else:
            return fig

    ######
    # TODO: Update to pull historical data from hist data class
    #####
    def PLOT_monthly_calls(
        self,
        show_individual_runs=False,
        use_poppins=False,
        show_historical=False,
        show_historical_individual_years=False,
        job_count_col="total_jobs",
    ) -> Figure:
        call_counts_monthly = (
            self.call_df.groupby(["run_number", "month_start"])[["P_ID"]]
            .count()
            .reset_index()
            .rename(columns={"P_ID": "monthly_calls"})
        )

        # Identify first and last month in the dataset
        first_month = call_counts_monthly["month_start"].min()
        last_month = call_counts_monthly["month_start"].max()

        # Filter out the first and last month
        call_counts_monthly = call_counts_monthly[
            (call_counts_monthly["month_start"] != first_month)
            & (call_counts_monthly["month_start"] != last_month)
        ]

        # Compute mean of number of patients, standard deviation, and total number of monthly calls

        summary = (
            call_counts_monthly.groupby("month_start")["monthly_calls"]
            .agg(["mean", "std", "count", "max", "min", q10, q90])
            .reset_index()
        )

        # Create the plot
        fig = px.line(
            summary,
            x="month_start",
            y="mean",
            markers=True,
            labels={"mean": "Average Calls Per Month", "month_start": "Month"},
            title="Number of Monthly Calls Received in Simulation",
            color_discrete_sequence=[COLORSCHEME["navy"]],
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
                        x=run_data["month_start"],
                        y=run_data["monthly_calls"],
                        mode="lines",
                        line=dict(color="gray", width=2, dash="dot"),
                        opacity=0.6,
                        name=f"Simulation Run {run}",
                        showlegend=True,
                    )
                )

        # Add full range as a shaded region
        fig.add_traces(
            [
                go.Scatter(
                    x=summary["month_start"],
                    y=summary["max"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    x=summary["month_start"],
                    y=summary["min"],
                    mode="lines",
                    fill="tonexty",
                    line=dict(width=0),
                    fillcolor="rgba(0, 176, 185, 0.15)",
                    showlegend=True,
                    name="Full Range Across Simulation Runs",
                ),
            ]
        )

        # Add 10th-90th percentile interval as a shaded region
        fig.add_traces(
            [
                go.Scatter(
                    x=summary["month_start"],
                    y=summary["q90"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    x=summary["month_start"],
                    y=summary["q10"],
                    mode="lines",
                    fill="tonexty",
                    line=dict(width=0),
                    fillcolor="rgba(0, 176, 185, 0.3)",
                    showlegend=True,
                    name="80% Range Across Simulation Runs",
                ),
            ]
        )

        # Increase upper y limit to be sightly bigger than the max number of calls observed in a month
        # Ensure lower y limit is 0
        fig = fig.update_yaxes(
            {"range": (0, call_counts_monthly["monthly_calls"].max() * 1.1)}
        )

        if show_historical:
            # Convert to datetime
            # (using 'parse_dates=True' in read_csv isn't reliably doing that, so make it explicit here)
            historical_jobs_per_month = (
                self.historical_data.historical_monthly_totals_all_calls.copy()
            )

            historical_summary = (
                historical_jobs_per_month.groupby("Month_Numeric")[job_count_col]
                .agg(["max", "min"])
                .reset_index()
                .rename(columns={"max": "historic_max", "min": "historic_min"})
            )

            call_counts_monthly["Month_Numeric"] = call_counts_monthly[
                "month_start"
            ].apply(lambda x: x.month)

            # historical_jobs_per_month["New_Date"] = (
            #     historical_jobs_per_month["Month"]
            #     .apply(lambda x: datetime.date(year=first_month.year,day=1,month=x.month))
            #     )

            if (historical_jobs_per_month[job_count_col].max() * 1.1) > (
                call_counts_monthly["monthly_calls"].max() * 1.1
            ):
                fig = fig.update_yaxes(
                    {"range": (0, historical_jobs_per_month[job_count_col].max() * 1.1)}
                )

            if show_historical_individual_years:
                for idx, year in enumerate(
                    historical_jobs_per_month["Year_Numeric"].unique()
                ):
                    # Filter the data for the current year
                    year_data = historical_jobs_per_month[
                        historical_jobs_per_month["Year_Numeric"] == year
                    ]

                    new_df = call_counts_monthly.drop_duplicates("month_start").merge(
                        year_data, on="Month_Numeric", how="left"
                    )

                    # Add the trace for the current year
                    fig.add_trace(
                        go.Scatter(
                            x=new_df["month_start"],
                            y=new_df[job_count_col],
                            mode="lines+markers",
                            opacity=0.7,
                            name=str(year),  # Using the year as the trace name
                            line=dict(
                                color=list(COLORSCHEME.values())[idx], dash="dash"
                            ),  # Default to gray if no specific color found
                        )
                    )
            else:
                # Add a filled range showing the entire historical range
                call_counts_monthly = call_counts_monthly.merge(
                    historical_summary, on="Month_Numeric", how="left"
                )

                # Add a filled range (shaded area) for the historical range
                # print("==_job_count_calculation plot_monthly_calls(): call_counts_monthly")
                # print(call_counts_monthly)

                # Ensure we only have one row per month to avoid issues with filling the historical range
                call_counts_historical_plotting_min_max = call_counts_monthly[
                    ["month_start", "historic_max", "historic_min"]
                ].drop_duplicates()

                fig.add_trace(
                    go.Scatter(
                        x=call_counts_historical_plotting_min_max["month_start"],
                        y=call_counts_historical_plotting_min_max["historic_max"],
                        mode="lines",
                        showlegend=False,
                        line=dict(
                            color="rgba(0,0,0,0)"
                        ),  # Invisible line (just the area)
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=call_counts_historical_plotting_min_max["month_start"],
                        y=call_counts_historical_plotting_min_max["historic_min"],
                        mode="lines",
                        name="Historical Range",
                        line=dict(
                            color="rgba(0,0,0,0)"
                        ),  # Invisible line (just the area)
                        fill="tonexty",  # Fill the area between this and the next trace
                        fillcolor="rgba(255, 164, 0, 0.15)",  # Semi-transparent fill color
                    )
                )

        # Adjust font to match DAA style
        if use_poppins:
            return fig.update_layout(
                font=dict(family="Poppins", size=18, color="black")
            )
        else:
            return fig

    def count_weekdays_in_month(self, year, month):
        """Returns a dictionary with the count of each weekday in a given month."""
        weekday_counts = {day: 0 for day in day_name}

        _, num_days = monthrange(year, month)  # Get total days in month
        for day in range(1, num_days + 1):
            weekday = day_name[datetime.datetime(year, month, day).weekday()]
            weekday_counts[weekday] += 1

        return weekday_counts

    def compute_average_calls(self, df):
        """Computes the average calls received per day of the week for each month."""
        results = []
        for _, row in df.iterrows():
            year, month = row["month"].year, row["month"].month
            weekday_counts = self.count_weekdays_in_month(year, month)

            averages = {day: row[day] / weekday_counts[day] for day in weekday_counts}
            averages["month"] = row["month"].strftime("%Y-%m")
            results.append(averages)

        return pd.DataFrame(results)

    ######
    # TODO: Update to pull historical data from hist data class
    #####
    def PLOT_daily_call_counts(
        self,
        box_plot=False,
        average_per_month=False,
        bar_colour="teal",
        title="Calls Per Day",
        use_poppins=False,
        error_bar_colour="charcoal",
        show_error_bars_bar=True,
        show_historical=True,
    ) -> Figure:
        # Create a blank figure to build on
        fig = go.Figure()

        ###########
        # Add historical data if option selected
        ###########
        if show_historical:
            jobs_per_day_historic = (
                self.historical_data.historical_monthly_totals_by_day_of_week.copy()
            )

            # print("===========jobs_per_day_historic============")
            # print(jobs_per_day_historic)
            # Compute the average calls per day
            jobs_per_day_historic = self.compute_average_calls(jobs_per_day_historic)
            jobs_per_day_historic["month"] = pd.to_datetime(
                jobs_per_day_historic["month"], format="ISO8601"
            )

            jobs_per_day_historic["year_numeric"] = jobs_per_day_historic[
                "month"
            ].apply(lambda x: x.year)
            jobs_per_day_historic["month_numeric"] = jobs_per_day_historic[
                "month"
            ].apply(lambda x: x.month)
            # print("======== jobs_per_day_historic - updated ========")
            # print(jobs_per_day_historic)

            jobs_per_day_historic_long = jobs_per_day_historic.melt(
                id_vars=["month", "month_numeric", "year_numeric"]
            )
            # jobs_per_day_historic_long["hour"] = jobs_per_day_historic_long['variable'].str.extract(r"(\d+)\s")
            jobs_per_day_historic_long.rename(columns={"variable": "day"}, inplace=True)
            # jobs_per_day_historic_long["hour"] = jobs_per_day_historic_long["hour"].astype('int')
            jobs_per_day_historic_long = jobs_per_day_historic_long[
                ~jobs_per_day_historic_long["value"].isna()
            ]

            if not average_per_month:
                jobs_per_day_historic_long["value"] = jobs_per_day_historic_long[
                    "value"
                ] * (float(get_param("sim_duration", self.params_df)) / 60 / 24 / 7)

            jobs_per_day_historic_agg = (
                jobs_per_day_historic_long.groupby(["day"])["value"].agg(
                    ["min", "max", q10, q90]
                )
            ).reset_index()

            fig.add_trace(
                go.Bar(
                    x=jobs_per_day_historic_agg["day"],
                    y=jobs_per_day_historic_agg["max"]
                    - jobs_per_day_historic_agg["min"],  # The range
                    base=jobs_per_day_historic_agg["min"],  # Starts from the minimum
                    name="Historical Range",
                    marker_color="rgba(100, 100, 255, 0.2)",  # Light blue with transparency
                    hoverinfo="skip",  # Hide hover info for clarity
                    showlegend=True,
                    width=1.0,  # Wider bars to make them contiguous
                    offsetgroup="historical",  # Grouping ensures alignment
                )
            )

            fig.add_trace(
                go.Bar(
                    x=jobs_per_day_historic_agg["day"],
                    y=jobs_per_day_historic_agg["q90"]
                    - jobs_per_day_historic_agg["q10"],  # The range
                    base=jobs_per_day_historic_agg["q10"],  # Starts from the minimum
                    name="Historical 80% Range",
                    marker_color="rgba(100, 100, 255, 0.3)",  # Light blue with transparency
                    hoverinfo="skip",  # Hide hover info for clarity
                    showlegend=True,
                    width=1.0,  # Wider bars to make them contiguous
                    offsetgroup="historical",  # Grouping ensures alignment
                )
            )

            fig.update_layout(
                xaxis=dict(dtick=1),
                barmode="overlay",  # Ensures bars overlay instead of stacking
                title="Comparison of Simulated and Historical Call Counts",
            )

        ################################
        # Add in the actual data
        ################################

        if box_plot:
            if average_per_month:
                self.daily_calls_per_run["average_per_month"] = (
                    self.daily_calls_per_run["count"]
                    / (float(get_param("sim_duration", self.params_df)) / 60 / 24 / 7)
                )
                y_column = "average_per_month"
                y_label = "Average Monthly Calls Per Day Across Simulation<br>Averaged Across Simulation Runs"
            else:
                y_column = "count"
                y_label = "Total Calls Per Hour Day Simulation<br>Averaged Across Simulation Runs"

            # Add box plot trace
            fig.add_trace(
                go.Box(
                    x=self.daily_calls_per_run["day"],
                    y=self.daily_calls_per_run[y_column],
                    name="Simulated Mean",
                    width=0.4,
                    marker=dict(color=COLORSCHEME[bar_colour]),
                    showlegend=True,
                    boxpoints="outliers",  # Show all data points
                )
            )

            # Update layout
            fig.update_layout(
                title=title, xaxis=dict(title="Day", dtick=1), yaxis=dict(title=y_label)
            )

        else:
            # Create required dataframe for simulation output display
            aggregated_data = (
                self.daily_calls_per_run.groupby("day")
                .agg(
                    mean_count=("count", "mean"),
                    # std_count=("count", "std")
                    se_count=(
                        "count",
                        lambda x: x.std() / np.sqrt(len(x)),
                    ),  # Standard Error
                )
                .reset_index()
            )

            if show_error_bars_bar:
                # error_y = aggregated_data["std_count"]
                error_y = aggregated_data["se_count"]
            else:
                error_y = None

            # Add the bar trace if plotting averages across
            if average_per_month:
                aggregated_data["mean_count"] = aggregated_data["mean_count"] / (
                    float(get_param("sim_duration", self.params_df)) / 60 / 24 / 7
                )
                # aggregated_data['std_count'] = (
                #     aggregated_data['std_count'] /
                #     (float(_processing_functions.get_param("sim_duration", params_df))/ 60 / 24/ 7)
                #     )

                aggregated_data["se_count"] = aggregated_data["se_count"] / (
                    float(get_param("sim_duration", self.params_df)) / 60 / 24 / 7
                )

                fig.add_trace(
                    go.Bar(
                        x=aggregated_data["day"],
                        y=aggregated_data["mean_count"],
                        name="Simulated Mean",
                        marker=dict(
                            color=COLORSCHEME[bar_colour]
                        ),  # Use your color scheme
                        error_y=dict(type="data", array=error_y, visible=True)
                        if error_y is not None
                        else None,
                        width=0.4,  # Narrower bars in front
                        offsetgroup="simulated",
                    )
                )

                fig.update_layout(
                    xaxis=dict(dtick=1),
                    barmode="overlay",  # Ensures bars overlay instead of stacking
                    title=title,
                    yaxis_title="Average Monthly Calls Per Day Across Simulation<br>Averaged Across Simulation Runs",
                    xaxis_title="Day",
                )

            # Add the bar trace if plotting total calls over the course of the simulation
            else:
                fig.add_trace(
                    go.Bar(
                        x=aggregated_data["day"],
                        y=aggregated_data["mean_count"],
                        name="Simulated Mean",
                        marker=dict(
                            color=COLORSCHEME[bar_colour]
                        ),  # Use your color scheme
                        error_y=dict(type="data", array=error_y, visible=True)
                        if error_y is not None
                        else None,
                        width=0.4,  # Narrower bars in front
                        offsetgroup="simulated",
                    )
                )

                fig.update_layout(
                    xaxis=dict(dtick=1),
                    barmode="overlay",  # Ensures bars overlay instead of stacking
                    title=title,
                    yaxis_title="Total Calls Per Day Across Simulation",
                    xaxis_title="Day",
                )

        if not box_plot:
            fig = fig.update_traces(error_y_color=COLORSCHEME[error_bar_colour])

        fig.update_xaxes(categoryorder="array", categoryarray=self.day_order)

        if use_poppins:
            return fig.update_layout(
                font=dict(family="Poppins", size=18, color="black")
            )
        else:
            return fig

    def PLOT_missed_jobs(
        self,
        show_proportions_per_hour=False,
        by_quarter=False,
    ) -> Figure:
        simulated_df_resource_preferred_outcome = self.run_results[
            self.run_results["event_type"] == "resource_preferred_outcome"
        ].copy()

        simulated_df_resource_preferred_outcome["outcome_simplified"] = (
            simulated_df_resource_preferred_outcome["time_type"].apply(
                lambda x: "No HEMS available"
                if "No HEMS resource available" in x
                else "HEMS (helo or car) available and sent"
            )
        )

        # Set up historical dataframe
        if not by_quarter:
            historical_df = (
                self.historical_data.historical_missed_calls_by_hour_df.copy()
            )

        else:
            historical_df = self.historical_data.historical_missed_calls_by_quarter_and_hour_df.copy()

        # Regardless of whether grouping by quarter or not, do some basic wrangling
        historical_df.rename(
            columns={"callsign_group_simplified": "outcome_simplified"},
            inplace=True,
        )
        historical_df["what"] = "Historical"

        # Now branch off for distinctive data wrangling
        if not by_quarter:
            simulated_df_counts = (
                simulated_df_resource_preferred_outcome.groupby(
                    ["outcome_simplified", "hour"]
                )[["P_ID"]]
                .count()
                .reset_index()
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
                    facet_row_spacing=0.2,
                    labels={
                        "outcome_simplified": "Job Outcome",
                        "count": "Count of Jobs",
                        "hour": "Hour",
                    },
                )

                # Allow each y-axis to be independent
                fig.update_yaxes(matches=None)

                # Move facet row labels above each subplot, aligned left
                fig.for_each_annotation(
                    lambda a: a.update(
                        text=a.text.split("=")[-1],  # remove 'what='
                        x=0,  # align left
                        xanchor="left",
                        y=a.y + 0.35,  # move label above the plot
                        yanchor="top",
                        textangle=0,  # horizontal
                        font=dict(size=24),
                    )
                )

                # Ensure x axis tick labels appear on both facets
                fig.for_each_xaxis(
                    lambda xaxis: xaxis.update(
                        showticklabels=True, tickmode="linear", tick0=0, dtick=1
                    )
                )

                # Increase top margin to prevent overlap
                fig.update_layout(margin=dict(t=100))

                return fig
            else:
                # Compute proportions within each hour + source
                df_prop = (
                    full_df.groupby(["hour", "what"])
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
                    labels={"proportion": "Proportion", "what": ""},
                )

                fig.update_yaxes(range=[0, 1], matches="y")  # consistent y-axis
                fig.for_each_annotation(
                    lambda a: a.update(text=a.text.split("=")[-1])
                )  # clean facet labels

                fig.update_layout(
                    hovermode="x unified",
                    legend_title_text="Outcome",
                )

                fig.update_layout(
                    legend=dict(
                        orientation="h",  # horizontal layout
                        yanchor="bottom",
                        y=1.12,  # a bit above the plot
                        xanchor="center",
                        x=0.5,  # center aligned
                    )
                )

                # Increase spacing below title
                fig.update_layout(margin=dict(t=150))

                fig.for_each_xaxis(
                    lambda axis: axis.update(
                        # Rotate labels
                        tickangle=90,
                        # Force display of both labels even on narrow screens
                        showticklabels=True,
                        tickmode="linear",
                        tick0=0,
                        dtick=1,
                    )
                )

                return fig
        # if by_quarter
        else:
            simulated_df_resource_preferred_outcome.rename(
                columns={"qtr": "quarter"}, inplace=True
            )

            simulated_df_counts = (
                simulated_df_resource_preferred_outcome.groupby(
                    ["outcome_simplified", "quarter", "hour"]
                )[["P_ID"]]
                .count()
                .reset_index()
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
                    labels={
                        "outcome_simplified": "Job Outcome",
                        "count": "Count of Jobs",
                        "hour": "Hour",
                    },
                )

                # Allow each y-axis to be independent
                fig.update_yaxes(matches=None)

                fig.for_each_xaxis(
                    lambda xaxis: xaxis.update(
                        showticklabels=True, tickmode="linear", tick0=0, dtick=1
                    )
                )

                # Increase top margin to prevent overlap
                fig.update_layout(margin=dict(t=100))

                fig.update_layout(
                    legend=dict(
                        orientation="h",  # horizontal layout
                        yanchor="bottom",
                        y=1.12,  # a bit above the plot
                        xanchor="center",
                        x=0.5,  # center aligned
                    )
                )

                return fig

            else:
                # Step 1: Compute proportions within each hour + source
                df_prop = (
                    full_df.groupby(["quarter", "hour", "what"])
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
                    labels={"proportion": "Proportion", "what": ""},
                )

                fig.update_yaxes(range=[0, 1], matches="y")  # consistent y-axis
                fig.for_each_annotation(
                    lambda a: a.update(text=a.text.split("=")[-1])
                )  # clean facet labels

                fig.update_layout(
                    hovermode="x unified",
                    legend_title_text="Outcome",
                )

                fig.update_layout(
                    legend=dict(
                        orientation="h",  # horizontal layout
                        yanchor="bottom",
                        y=1.12,  # a bit above the plot
                        xanchor="center",
                        x=0.5,  # center aligned
                    )
                )

                fig.update_layout(margin=dict(t=150))

                fig.for_each_xaxis(
                    lambda axis: axis.update(
                        tickangle=90,
                        # showticklabels=True,
                        tickmode="linear",
                        tick0=0,
                        dtick=1,
                    )
                )

                return fig

    def PLOT_job_count_heatmap(
        self, normalise_per_day=False, simulated_days=None
    ) -> Figure:
        run_results = self.run_results[
            self.run_results["event_type"] == "resource_use"
        ][["P_ID", "run_number", "hour", "time_type"]].copy()

        # Unique values
        hours = list(range(24))  # 0–23 hours
        callsigns = run_results["time_type"].unique()

        # Cartesian product of all combinations to ensure we have a value on the x axis
        # for the hours with no data in
        full_index = pd.MultiIndex.from_product(
            [callsigns, hours], names=["time_type", "hour"]
        )

        # # # Group by callsign and hour, and count the number of occurrences
        heatmap_data = (
            run_results.groupby(["time_type", "hour"])
            .size()
            .reindex(full_index, fill_value=0)
            .reset_index(name="count")
        )

        # Normalise by number of runs
        heatmap_data["count"] = heatmap_data["count"] / len(
            run_results["run_number"].unique()
        )

        if normalise_per_day and simulated_days is not None:
            heatmap_data["count"] = heatmap_data["count"] / simulated_days

        # Create pivot table for heatmap matrix
        pivot = heatmap_data.pivot(
            index="time_type", columns="hour", values="count"
        ).fillna(0)

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[str(h) for h in pivot.columns],
                y=pivot.index,
                colorscale="Blues",
                colorbar=dict(title="Count"),
                showscale=True,
                xgap=1,  # Gap between columns (simulates vertical borders)
                ygap=1,  # Gap between rows (simulates horizontal borders)
            )
        )

        if normalise_per_day and simulated_days:
            title = "Heatmap of Average Hourly Simulated Calls Responded to by Hour and Callsign"
        else:
            title = "Heatmap of Total Hourly Simulated Calls Responded to by Hour and Callsign (Run Average)"

        fig.update_layout(
            title=title,
            xaxis_title="Hour of Day",
            yaxis_title="Callsign",
            height=600,
            xaxis=dict(dtick=1),
        )

        return fig

    def PLOT_job_count_heatmap_monthly(
        self, normalise_per_day=False, simulated_days=None
    ) -> Figure:
        run_results = self.run_results[
            self.run_results["event_type"] == "resource_use"
        ][["P_ID", "run_number", "hour", "timestamp_dt", "time_type", "month"]].copy()

        # Prepare full grid of (month, hour, callsign)
        hours = list(range(24))
        months = self.month_order
        callsigns = sorted(run_results["time_type"].unique())

        full_index = pd.MultiIndex.from_product(
            [months, hours, callsigns], names=["month", "hour", "time_type"]
        )

        grouped = (
            run_results.groupby(["month", "hour", "time_type"], observed=False)
            .size()
            .reindex(full_index, fill_value=0)
            .reset_index(name="count")
        )

        # Normalize
        grouped["count"] = grouped["count"] / run_results["run_number"].nunique()
        if normalise_per_day and simulated_days:
            grouped["count"] = grouped["count"] / simulated_days

        # Pivot to 2D arrays for each callsign
        subplot_data = {
            callsign: grouped[grouped["time_type"] == callsign]
            .pivot(index="month", columns="hour", values="count")
            .reindex(index=self.month_order)  # Ensure full month order
            .fillna(0)
            for callsign in callsigns
        }

        # Create subplots
        fig = make_subplots(
            rows=len(callsigns),
            cols=1,
            shared_xaxes=True,
            shared_yaxes=False,
            subplot_titles=[f"Callsign: {c}" for c in callsigns],
            vertical_spacing=0.1,
        )

        for i, callsign in enumerate(callsigns, start=1):
            z = subplot_data[callsign].values
            x = [str(h) for h in subplot_data[callsign].columns]
            y = subplot_data[callsign].index.tolist()

            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=x,
                    y=y,
                    colorscale="Blues",
                    colorbar=dict(title="Count") if i == 1 else None,
                    showscale=(i == 1),
                    xgap=1,
                    ygap=1,
                ),
                row=i,
                col=1,
            )

        if normalise_per_day and simulated_days:
            title = "Average Hourly Job Count by Month and Callsign"
        else:
            title = "Total Hourly Job Count Across Simulated Period by Month and Callsign (Run Average)"

        fig.update_layout(
            height=350 * len(callsigns),
            title=title,
            # xaxis_title='Hour of Day',
            yaxis_title="Month",
        )

        fig.update_yaxes(autorange="reversed")

        # Make x-axis labels show on all subplots
        for i in range(1, len(callsigns) + 1):
            fig.update_xaxes(
                showticklabels=True, dtick=1, title_text="Hour of Day", row=i, col=1
            )

        return fig

    def PLOT_jobs_per_callsign(self):
        # Create a count of the number of days in the sim that each resource had that many jobs
        # i.e. how many days did CC70 have 0 jobs, 1 job, 2 jobs, etc.
        df = self.run_results.copy()

        df["date"] = pd.to_datetime(df["timestamp_dt"], format="ISO8601").dt.date
        all_counts = (
            df[df["event_type"] == "resource_use"]
            .groupby(["time_type", "date", "run_number"])["P_ID"]
            .count()
            .reset_index()
        )
        all_counts.rename(columns={"P_ID": "jobs_in_day"}, inplace=True)

        # We must assume any missing day in our initial count df is a 0 count
        # So generate a df with every possible combo
        all_combinations = pd.DataFrame(
            list(
                itertools.product(
                    all_counts["time_type"].unique(),
                    df["date"].unique(),
                    df["run_number"].unique(),
                )
            ),
            columns=["time_type", "date", "run_number"],
        )
        # Join this in
        merged = all_combinations.merge(
            all_counts, on=["time_type", "date", "run_number"], how="left"
        )
        # Fill na values with 0
        merged["jobs_in_day"] = merged["jobs_in_day"].fillna(0).astype(int)
        # Finally transform into pure counts
        sim_count_df = (
            merged.groupby(["time_type", "jobs_in_day"])[["date"]]
            .count()
            .reset_index()
            .rename(columns={"date": "count", "time_type": "callsign"})
        )

        sim_count_df["what"] = "Simulated"

        # Bring in historical data
        jobs_per_day_per_callsign_historical = (
            self.historical_data.historical_jobs_per_day_per_callsign.copy()
        )

        jobs_per_day_per_callsign_historical["what"] = "Historical"

        # Join the two together
        full_df = pd.concat([jobs_per_day_per_callsign_historical, sim_count_df])

        # Plot as histograms
        fig = px.histogram(
            full_df,
            x="jobs_in_day",
            y="count",
            color="what",
            facet_col="callsign",
            facet_row="what",
            histnorm="percent",
            #  barmode="overlay",
            #  opacity=1
        )

        return fig

    def get_care_cat_proportion_table(self):
        historical_counts_simple = (
            self.historical_data.care_cat_by_hour_historic.groupby("care_category")[
                "count"
            ]
            .sum()
            .reset_index()
            .rename(
                columns={
                    "count": "Historic Job Counts",
                    "care_category": "Care Category",
                }
            )
        )

        run_results = self.run_results.copy()

        # Amend care category to reflect the small proportion of regular jobs assumed to have
        # a helicopter benefit
        run_results.loc[
            (run_results["heli_benefit"] == "y") & (run_results["care_cat"] == "REG"),
            "care_cat",
        ] = "REG - helicopter benefit"

        care_cat_counts_sim = (
            run_results[run_results["event_type"] == "patient_helicopter_benefit"][
                ["P_ID", "run_number", "care_cat"]
            ]
            .reset_index()
            .groupby(["care_cat"])
            .size()
            .reset_index(name="count")
        ).copy()

        full_counts = historical_counts_simple.merge(
            care_cat_counts_sim.rename(
                columns={"care_cat": "Care Category", "count": "Simulated Job Counts"}
            ),
            how="outer",
            on="Care Category",
        )

        # Calculate proportions by column
        full_counts = full_counts[
            full_counts["Care Category"] != "Unknown - DAA resource did not attend"
        ].copy()

        full_counts["Historic Percentage"] = (
            full_counts["Historic Job Counts"]
            / full_counts["Historic Job Counts"].sum()
        )
        full_counts["Simulated Percentage"] = (
            full_counts["Simulated Job Counts"]
            / full_counts["Simulated Job Counts"].sum()
        )

        full_counts["Historic Percentage"] = full_counts["Historic Percentage"].apply(
            lambda x: f"{x:.1%}"
        )
        full_counts["Simulated Percentage"] = full_counts["Simulated Percentage"].apply(
            lambda x: f"{x:.1%}"
        )

        full_counts["Care Category"] = pd.Categorical(
            full_counts["Care Category"],
            ["CC", "EC", "REG - helicopter benefit", "REG"],
        )

        return full_counts.sort_values("Care Category")

    def get_care_cat_counts_plot_sim(self, show_proportions=False):
        run_results = self.run_results.copy()

        # Amend care category to reflect the small proportion of regular jobs assumed to have
        # a helicopter benefit
        run_results.loc[
            (run_results["heli_benefit"] == "y") & (run_results["care_cat"] == "REG"),
            "care_cat",
        ] = "REG - helicopter benefit"

        care_cat_by_hour = (
            run_results[run_results["event_type"] == "patient_helicopter_benefit"][
                ["P_ID", "run_number", "care_cat", "hour"]
            ]
            .reset_index()
            .groupby(["hour", "care_cat"])
            .size()
            .reset_index(name="count")
        ).copy()

        # Calculate total per hour
        total_per_hour = care_cat_by_hour.groupby("hour")["count"].transform("sum")
        # Add proportion column
        care_cat_by_hour["proportion"] = care_cat_by_hour["count"] / total_per_hour

        title = "Care Category of calls in simulation by hour of day with EC/CC/Regular - Heli Benefit/Regular"

        if not show_proportions:
            fig = px.bar(
                care_cat_by_hour,
                x="hour",
                y="count",
                color="care_cat",
                title=title,
                category_orders={
                    "care_cat": ["CC", "EC", "REG - helicopter benefit", "REG"]
                },
            )

        # if show_proportions
        else:
            fig = px.bar(
                care_cat_by_hour,
                x="hour",
                y="proportion",
                color="care_cat",
                title=title,
                category_orders={
                    "care_cat": ["CC", "EC", "REG - helicopter benefit", "REG"]
                },
            )

        fig.update_layout(xaxis=dict(dtick=1))

        return fig

    def get_preferred_outcome_by_hour(self, show_proportions=False):
        resource_preferred_outcome_by_hour = (
            self.run_results[
                self.run_results["event_type"] == "resource_preferred_outcome"
            ][["P_ID", "run_number", "care_cat", "time_type", "hour"]]
            .reset_index()
            .groupby(["time_type", "hour"])
            .size()
            .reset_index(name="count")
        ).copy()

        # Calculate total per hour
        total_per_hour = resource_preferred_outcome_by_hour.groupby("hour")[
            "count"
        ].transform("sum")
        # Add proportion column
        resource_preferred_outcome_by_hour["proportion"] = (
            resource_preferred_outcome_by_hour["count"] / total_per_hour
        )

        if not show_proportions:
            fig = px.bar(
                resource_preferred_outcome_by_hour,
                x="hour",
                y="count",
                color="time_type",
            )

        else:
            fig = px.bar(
                resource_preferred_outcome_by_hour,
                x="hour",
                y="proportion",
                color="time_type",
            )

        return fig

    def get_facet_plot_preferred_outcome_by_hour(self):
        resource_preferred_outcome_by_hour = (
            self.run_results[
                self.run_results["event_type"] == "resource_preferred_outcome"
            ][["P_ID", "run_number", "care_cat", "time_type", "hour"]]
            .reset_index()
            .groupby(["time_type", "hour"])
            .size()
            .reset_index(name="count")
        ).copy()

        # Calculate total per hour
        total_per_hour = resource_preferred_outcome_by_hour.groupby("hour")[
            "count"
        ].transform("sum")
        # Add proportion column
        resource_preferred_outcome_by_hour["proportion"] = (
            resource_preferred_outcome_by_hour["count"] / total_per_hour
        )

        resource_preferred_outcome_by_hour["time_type"] = (
            resource_preferred_outcome_by_hour["time_type"].apply(
                lambda x: textwrap.fill(x, width=25).replace("\n", "<br>")
            )
        )

        fig = px.bar(
            resource_preferred_outcome_by_hour,
            x="hour",
            y="proportion",
            facet_col="time_type",
            facet_col_wrap=4,
            height=800,
            facet_col_spacing=0.05,
            facet_row_spacing=0.13,
        )

        return fig

    def plot_patient_outcomes(
        self,
        group_cols="vehicle_type",
        outcome_col="hems_result",
        plot_counts=False,
        return_fig=True,
    ):
        patient_outcomes_df = (
            self.run_results[self.run_results["time_type"] == "HEMS call start"][
                [
                    "P_ID",
                    "run_number",
                    "heli_benefit",
                    "care_cat",
                    "vehicle_type",
                    "hems_result",
                    "outcome",
                ]
            ]
            .reset_index(drop=True)
            .copy()
        )

        def calculate_grouped_proportions(df, group_cols, outcome_col):
            """
            Calculate counts and proportions of an outcome column grouped by one or more columns.

            Parameters:
            df (pd.DataFrame): The input DataFrame.
            group_cols (str or list of str): Column(s) to group by (e.g., 'care_cat', 'vehicle_type').
            outcome_col (str): The name of the outcome column (e.g., 'hems_result').

            Returns:
            pd.DataFrame: A DataFrame with counts and proportions of outcome values per group.
            """
            if isinstance(group_cols, str):
                group_cols = [group_cols]

            count_df = (
                df.value_counts(group_cols + [outcome_col])
                .reset_index()
                .sort_values(group_cols + [outcome_col])
            )
            count_df.rename(columns={0: "count"}, inplace=True)

            # Calculate the total count per group (excluding the outcome column)
            total_per_group = count_df.groupby(group_cols)["count"].transform("sum")
            count_df["proportion"] = count_df["count"] / total_per_group

            return count_df

        patient_outcomes_df_grouped_counts = calculate_grouped_proportions(
            patient_outcomes_df, group_cols, outcome_col
        )

        if return_fig:
            if plot_counts:
                y = "count"
            else:
                y = "proportion"

            fig = px.bar(
                patient_outcomes_df_grouped_counts,
                color=group_cols,
                y=y,
                x=outcome_col,
                barmode="group",
            )

            return fig
        else:
            return patient_outcomes_df_grouped_counts

    def create_simulation_event_duration_df(self):
        order_of_events = [
            "HEMS call start",
            # 'No HEMS available',
            "HEMS allocated to call",
            # 'HEMS stand down before mobile',
            "HEMS mobile",
            # 'HEMS stand down en route',
            "HEMS on scene",
            # 'HEMS landed but no patient contact',
            "HEMS leaving scene",
            # 'HEMS patient treated (not conveyed)',
            "HEMS arrived destination",
            "HEMS clear",
        ]

        self.simulation_event_duration_df = self.run_results[
            self.run_results["time_type"].isin(order_of_events)
        ].copy()

        self.simulation_event_duration_df.time_type = (
            self.simulation_event_duration_df.time_type.astype("category")
        )

        self.simulation_event_duration_df.time_type = (
            self.simulation_event_duration_df.time_type.cat.set_categories(
                order_of_events
            )
        )

        self.simulation_event_duration_df = (
            self.simulation_event_duration_df.sort_values(
                ["run_number", "P_ID", "time_type"]
            )
        )

        # Calculate time difference within each group
        self.simulation_event_duration_df["time_elapsed"] = (
            self.simulation_event_duration_df.groupby(["P_ID", "run_number"])[
                "timestamp_dt"
            ].diff()
        )

        self.simulation_event_duration_df["time_elapsed_minutes"] = (
            self.simulation_event_duration_df["time_elapsed"].apply(
                lambda x: x.total_seconds() / 60.0 if pd.notna(x) else None
            )
        )

    def summarise_event_times(self):
        self.simulation_event_duration_df_summary = (
            self.simulation_event_duration_df.groupby("time_type", observed=False)[
                "time_elapsed_minutes"
            ]
            .agg(["mean", "median", "max", "min"])
            .round(1)
        )

    ##############
    # TODO: Confirm difference between run_results and event_log_df
    #############
    def resource_allocation_outcomes_run_variation(self):
        return (
            (
                self.run_results[
                    self.run_results["event_type"] == "resource_preferred_outcome"
                ]
                .groupby(["time_type", "run_number"])[["time_type"]]
                .count()
                / self.n_runs
            )
            .round(0)
            .astype("int")
            .rename(columns={"time_type": "Count"})
            .reset_index()
            .rename(
                columns={
                    "time_type": "Resource Allocation Attempt Outcome",
                    "run_number": "Run",
                }
            )
        ).copy()

    def get_perc_unattended_string(self):
        """
        Alternative to display_UNTATTENDED_calls_per_run

        This approach looks at instances where the resource_request_outcome
        was 'no resource available'
        """
        try:
            num_unattendable = len(
                self.run_results[
                    (self.run_results["event_type"] == "resource_request_outcome")
                    & (self.run_results["time_type"] == "No Resource Available")
                ]
            )

            # print(f"==get_perc_unattended_string - num_unattended: {num_unattendable}==")
        except:
            "Error"

        total_calls = len(
            self.run_results[
                (self.run_results["event_type"] == "resource_request_outcome")
            ]
        )

        try:
            perc_unattendable = num_unattendable / total_calls

            if perc_unattendable < 0.01:
                return f"{num_unattendable} of {total_calls} (< 0.1%)"
            else:
                return f"{num_unattendable} of {total_calls} ({perc_unattendable:.1%})"
        except:
            return "Error"

    def get_perc_unattended_string_normalised(self):
        """
        Alternative to display_UNTATTENDED_calls_per_run

        This approach looks at instances where the resource_request_outcome
        was 'no resource available'
        """
        try:
            num_unattendable = len(
                self.run_results[
                    (self.run_results["event_type"] == "resource_request_outcome")
                    & (self.run_results["time_type"] == "No Resource Available")
                ]
            )

            # print(f"==get_perc_unattended_string - num_unattended: {num_unattendable}==")
        except:
            "Error"

        total_calls = len(
            self.run_results[
                (self.run_results["event_type"] == "resource_request_outcome")
            ]
        )

        # print(f"==get_perc_unattended_string - total calls: {total_calls}==")

        try:
            perc_unattendable = num_unattendable / total_calls

            sim_duration_days = float(get_param("sim_duration_days", self.params_df))

            total_average_calls_missed_per_year = (
                num_unattendable / self.n_runs / sim_duration_days
            ) * 365
            total_average_calls_received_per_year = (
                total_calls / self.n_runs / sim_duration_days
            ) * 365

            if perc_unattendable < 0.01:
                return (
                    total_average_calls_received_per_year,
                    f"{(num_unattendable / self.n_runs):.0f} of {(total_calls / self.n_runs):.0f} (< 0.1%)",
                    f"This equates to around {total_average_calls_missed_per_year:.0f} of {total_average_calls_received_per_year:.0f} calls per year having no resource available to attend.",
                )
            else:
                return (
                    total_average_calls_received_per_year,
                    f"{(num_unattendable / self.n_runs):.0f} of {(total_calls / self.n_runs):.0f} ({perc_unattendable:.1%})",
                    f"This equates to around {total_average_calls_missed_per_year:.0f} of {total_average_calls_received_per_year:.0f} calls per year having no resource available to attend.",
                )
        except:
            return 2500, "Error", "Error"

    def get_missed_call_df(self):
        # Filter for relevant events

        resource_requests = self.run_results[
            self.run_results["event_type"] == "resource_request_outcome"
        ].copy()

        # Recode care_cat when helicopter benefit applies
        resource_requests["care_cat"] = resource_requests.apply(
            lambda x: "REG - Helicopter Benefit"
            if x["heli_benefit"] == "y" and x["care_cat"] == "REG"
            else x["care_cat"],
            axis=1,
        )

        # Group by care_cat, time_type, and run_number to get jobs per run
        jobs_per_run = (
            resource_requests.groupby(["care_cat", "time_type", "run_number"])
            .size()
            .reset_index(name="jobs")
        )

        sim_duration_days = float(get_param("sim_duration_days", self.params_df))

        # print(jobs_per_run.head())
        # print(jobs_per_run.jobs)
        # print(sim_duration_days)

        self.missed_jobs_per_run_breakdown = jobs_per_run.copy()

        self.missed_jobs_per_run_breakdown["jobs_per_year"] = (
            jobs_per_run["jobs"] / sim_duration_days
        ) * 365

        # Then aggregate to get average, min, and max per group
        self.missed_jobs_per_run_care_cat_summary = (
            jobs_per_run.groupby(["care_cat", "time_type"])
            .agg(
                jobs_average=("jobs", "mean"),
                jobs_min=("jobs", "min"),
                jobs_max=("jobs", "max"),
            )
            .reset_index()
        )

        # Add annualised average per year
        self.missed_jobs_per_run_care_cat_summary["jobs_per_year_average"] = (
            (
                self.missed_jobs_per_run_care_cat_summary["jobs_average"]
                / sim_duration_days
            )
            * 365
        ).round(0)

        self.missed_jobs_per_run_care_cat_summary["jobs_per_year_min"] = (
            (self.missed_jobs_per_run_care_cat_summary["jobs_min"] / sim_duration_days)
            * 365
        ).round(0)

        self.missed_jobs_per_run_care_cat_summary["jobs_per_year_max"] = (
            (
                self.missed_jobs_per_run_care_cat_summary["jobs_max"]
                / float(get_param("sim_duration_days", self.params_df))
            )
            * 365
        ).round(0)

    ###########################
    # TODO: NOT YET CONVERTED TO OO
    ##########################
    def make_SIMULATION_utilisation_variation_plot(
        self,
        historical_results_obj,
        car_colour="blue",
        helicopter_colour="red",
        use_poppins=False,
    ):
        """
        Creates a box plot to visualize the variation in resource utilization
        across all simulation runs, with historical mean and range indicators.

        Parameters
        ----------
        historical_results_obj

        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly box plot.
        """
        utilisation_df_per_run = self.utilisation_df_per_run.reset_index()
        utilisation_df_per_run["vehicle_type"] = utilisation_df_per_run[
            "vehicle_type"
        ].str.title()
        utilisation_df_per_run["callsign"] = utilisation_df_per_run["callsign"].astype(
            str
        )

        sorted_callsigns = sorted(utilisation_df_per_run["callsign"].unique())

        # Force ordering for y-axis alignment
        utilisation_df_per_run["callsign"] = pd.Categorical(
            utilisation_df_per_run["callsign"],
            categories=sorted_callsigns,
            ordered=True,
        )
        utilisation_df_per_run = utilisation_df_per_run.sort_values("callsign")

        fig = px.box(
            utilisation_df_per_run,
            x="perc_time_in_use",
            y="callsign",
            color="vehicle_type",
            title="Variation in Resource Utilisation Across All Simulation Runs",
            labels={
                "callsign": "Callsign",
                "perc_time_in_use": "Average Percentage of Available<br>Time Spent in Use",
                "vehicle_type": "Vehicle Type",
            },
            color_discrete_map={
                "Car": COLORSCHEME.get(car_colour, "blue"),
                "Helicopter": COLORSCHEME.get(helicopter_colour, "red"),
            },
            category_orders={"callsign": sorted_callsigns},
        )

        fig.update_layout(xaxis={"tickformat": ".0%"})

        # Flip y-axis so our numeric y positions match callsign order
        fig.update_yaxes(
            categoryorder="array", categoryarray=sorted_callsigns, autorange="reversed"
        )

        # Add historical data
        historical_utilisation_df_summary = (
            historical_results_obj.historical_utilisation_df_summary.copy()
        )
        historical_utilisation_df_summary.index = (
            historical_utilisation_df_summary.index.astype(str)
        )

        historical_marker_height_fraction = 0.4

        for num_idx, callsign_str in enumerate(sorted_callsigns):
            if callsign_str in historical_utilisation_df_summary.index:
                row = historical_utilisation_df_summary.loc[callsign_str]
                min_val = row["min"] / 100.0
                max_val = row["max"] / 100.0
                mean_val = row["mean"] / 100.0

                y_pos_center = num_idx
                y_pos_start = y_pos_center - historical_marker_height_fraction
                y_pos_end = y_pos_center + historical_marker_height_fraction

                fig.add_shape(
                    type="rect",
                    yref="y",
                    xref="x",
                    y0=y_pos_start,
                    x0=min_val,
                    y1=y_pos_end,
                    x1=max_val,
                    fillcolor=COLORSCHEME.get(
                        "historical_box_fill", "rgba(0,0,0,0.08)"
                    ),
                    line=dict(color="rgba(0,0,0,0)"),
                    layer="below",
                )

                fig.add_shape(
                    type="line",
                    yref="y",
                    xref="x",
                    y0=y_pos_start,
                    x0=mean_val,
                    y1=y_pos_end,
                    x1=mean_val,
                    line=dict(
                        dash="dot",
                        color=COLORSCHEME.get("charcoal", "black"),
                        width=2,
                    ),
                    layer="below",
                )

                fig.add_trace(
                    go.Scatter(
                        x=[mean_val + 0.01],
                        y=[callsign_str],
                        text=[f"Hist. Mean: {mean_val:.0%}"],
                        mode="text",
                        textfont=dict(
                            color=COLORSCHEME.get("charcoal", "black"), size=10
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        if use_poppins:
            fig.update_layout(
                font=dict(
                    family="Poppins",
                    size=18,
                    color=COLORSCHEME.get("charcoal", "black"),
                )
            )
        else:
            fig.update_layout(
                font=dict(
                    family="Arial, sans-serif",
                    size=12,
                    color=COLORSCHEME.get("charcoal", "black"),
                )
            )

        return fig

    def PLOT_SIMULATION_utilisation_summary(
        self,
        historical_results_obj,
        car_colour="blue",
        helicopter_colour="red",
        use_poppins=False,
    ):
        """
        Creates a bar plot to summarize the average resource utilization
        across all simulation runs.

        Parameters
        ----------
        historical_results_obj

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
        - "Car" is mapped to `COLORSCHEME["blue"]`.
        - "Helicopter" is mapped to `COLORSCHEME["red"]`.

        Example
        -------
        >>> fig = make_SIMULATION_utilisation_summary_plot(utilisation_df)
        >>> fig.show()
        """
        utilisation_df_overall = self.utilisation_df_overall.reset_index()
        utilisation_df_overall["vehicle_type"] = utilisation_df_overall[
            "vehicle_type"
        ].str.title()
        utilisation_df_overall["perc_time_formatted"] = utilisation_df_overall[
            "perc_time_in_use"
        ].apply(lambda x: f"Simulated: {x:.1%}")

        # Create base bar chart
        fig = px.bar(
            utilisation_df_overall,
            y="perc_time_in_use",
            x="callsign",
            color="vehicle_type",
            opacity=0.5,
            text="perc_time_formatted",
            title="Average Resource Utilisation Across All Simulation Runs",
            labels={
                "callsign": "Callsign",
                "perc_time_in_use": "Average Percentage of Available<br>Time Spent in Use",
                "vehicle_type": "Vehicle Type",
            },
            color_discrete_map={
                "Car": COLORSCHEME.get(car_colour, "blue"),  # Use .get for safety
                "Helicopter": COLORSCHEME.get(helicopter_colour, "red"),
            },
            # barmode='group' is default when color is used, which is good.
        )

        # Place actual label at the bottom of the bar
        fig.update_traces(textposition="inside", insidetextanchor="start")

        fig.update_layout(
            bargap=0.4,
            yaxis_tickformat=".0%",
            xaxis_type="category",  # Explicitly set x-axis to category
            # Ensure categories are ordered as per the sorted DataFrame
            # If callsigns are purely numeric but should be treated as categories, ensure they are strings
            xaxis={
                "categoryorder": "array",
                "categoryarray": sorted(utilisation_df_overall["callsign"].unique()),
            },
        )

        # Get the unique, sorted callsigns as they will appear on the x-axis
        # This order is now determined by the 'categoryarray' in layout or default sorting.
        x_axis_categories = sorted(utilisation_df_overall["callsign"].unique())

        # Define the width of the historical markers (box and line) relative to category slot
        # A category slot is 1 unit wide (e.g., from -0.5 to 0.5 around the category's integer index).
        # We'll make the markers 80% of this width.
        historical_marker_width_fraction = 0.3  # Half-width, so total width is 0.8

        # Iterate through the callsigns in the order they appear on the axis
        for num_idx, callsign_str in enumerate(x_axis_categories):
            if (
                callsign_str
                in historical_results_obj.historical_utilisation_df_summary.index
            ):
                row = historical_results_obj.historical_utilisation_df_summary.loc[
                    callsign_str
                ]

                min_val = row["min"] / 100.0  # Convert percentage to 0-1 scale
                max_val = row["max"] / 100.0
                mean_val = row["mean"] / 100.0

                # Calculate x-positions for the historical markers
                # num_idx is the integer position of the category (0, 1, 2, ...)
                x_pos_start = num_idx - historical_marker_width_fraction
                x_pos_end = num_idx + historical_marker_width_fraction

                # --- Min/Max shaded rectangle for historical range ---
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="y",  # Refer to data coordinates
                    x0=x_pos_start,
                    y0=min_val,
                    x1=x_pos_end,
                    y1=max_val,
                    fillcolor=COLORSCHEME.get(
                        "historical_box_fill", "rgba(0,0,0,0.08)"
                    ),
                    line=dict(color="rgba(0,0,0,0)"),  # No border for the box
                    layer="below",  # Draw below the main bars
                )

                # --- Mean horizontal line for historical mean ---
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=x_pos_start,
                    y0=mean_val,
                    x1=x_pos_end,
                    y1=mean_val,
                    line=dict(
                        dash="dot",
                        color=COLORSCHEME.get("charcoal", "black"),
                        width=2,
                    ),
                    layer="below",  # Draw below main bars
                )

                # --- Mean value label (text annotation) ---
                # Using go.Scatter for text annotation, positioned at the center of the category
                fig.add_trace(
                    go.Scatter(
                        x=[callsign_str],  # Use the category name for x
                        y=[
                            mean_val + (0.01 if max_val < 0.95 else -0.01)
                        ],  # Adjust y to avoid overlap with top
                        text=[f"Historical: {mean_val * 100:.0f}%"],  # Simpler label
                        mode="text",
                        textfont=dict(
                            color=COLORSCHEME.get("charcoal", "black"), size=10
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        if use_poppins:
            fig.update_layout(
                font=dict(
                    family="Poppins",
                    size=18,
                    color=COLORSCHEME.get("charcoal", "black"),
                )
            )
        else:  # Apply a default font for better appearance
            fig.update_layout(
                font=dict(
                    family="Arial, sans-serif",
                    size=12,
                    color=COLORSCHEME.get("charcoal", "black"),
                )
            )

        return fig

    # TODO: convert to use historical class
    # --- Helper function to display vehicle metric ---
    def display_vehicle_utilisation_metric(
        self,
        historical_data_class,
        st_column,
        callsign_to_display,
        vehicle_type_label,
        icon_unicode,
        current_quarto_string,
    ):
        """
        Displays the utilisation metrics for a given vehicle in a specified Streamlit column.
        Returns the updated quarto_string.
        """
        with st_column:
            with iconMetricContainer(
                key=f"{vehicle_type_label.lower()}_util_{callsign_to_display}",
                icon_unicode=icon_unicode,
                type="symbols",
            ):
                matched_sim = self.utilisation_df_overall[
                    self.utilisation_df_overall["callsign"] == callsign_to_display
                ]

                if not matched_sim.empty:
                    sim_util_fig = matched_sim["PRINT_perc"].values[0]
                    sim_util_display = f"{sim_util_fig}"
                else:
                    sim_util_fig = "N/A"
                    sim_util_display = "N/A"

                current_quarto_string += f"\n\nAverage simulated {callsign_to_display} Utilisation was {sim_util_fig}\n\n"

                st.metric(
                    f"Average Simulated {callsign_to_display} Utilisation",
                    sim_util_display,
                    border=True,
                )

            # Get historical data
            hist_util_value = historical_data_class.RETURN_hist_util_fig(
                callsign_to_display, "mean"
            )

            hist_util_value_display = (
                f"{hist_util_value}%"
                if isinstance(hist_util_value, (int, float))
                else hist_util_value
            )

            hist_util_caption = f"*The historical average utilisation of {callsign_to_display} was {hist_util_value_display}*\n\n"
            current_quarto_string += hist_util_caption
            current_quarto_string += "\n\n---\n\n"
            st.caption(hist_util_caption)

        return current_quarto_string

    def create_callsign_group_split_rwc_plot(
        self,
        historical_data_obj,
        x_is_callsign_group=False,
    ):
        jobs_by_callsign = (
            historical_data_obj.historical_monthly_totals_by_callsign.copy()
        )
        jobs_by_callsign["month"] = pd.to_datetime(
            jobs_by_callsign["month"], format="ISO8601"
        )
        jobs_by_callsign["quarter"] = jobs_by_callsign["month"].dt.quarter
        jobs_by_callsign = jobs_by_callsign.melt(id_vars=["month", "quarter"]).rename(
            columns={"variable": "callsign", "value": "jobs"}
        )
        jobs_by_callsign["callsign_group"] = jobs_by_callsign["callsign"].str.extract(
            r"(\d+)"
        )
        jobs_by_callsign_group_hist = (
            jobs_by_callsign.groupby(["callsign_group", "quarter"])["jobs"]
            .sum()
            .reset_index()
        )
        # Group by quarter and compute total jobs per quarter
        quarter_totals_hist = jobs_by_callsign_group_hist.groupby("quarter")[
            "jobs"
        ].transform("sum")
        # Calculate proportion
        jobs_by_callsign_group_hist["proportion"] = (
            jobs_by_callsign_group_hist["jobs"] / quarter_totals_hist
        )
        jobs_by_callsign_group_hist["what"] = "Historical"

        jobs_by_callsign_sim = self.resource_use_events_only_df[
            ["run_number", "P_ID", "time_type", "qtr"]
        ].copy()
        jobs_by_callsign_sim["callsign_group"] = jobs_by_callsign_sim[
            "time_type"
        ].str.extract(r"(\d+)")
        jobs_by_callsign_group_sim = (
            jobs_by_callsign_sim.groupby(["qtr", "callsign_group"])
            .size()
            .reset_index()
            .rename(columns={"qtr": "quarter", 0: "jobs"})
        )
        # Group by quarter and compute total jobs per quarter
        quarter_totals_sim = jobs_by_callsign_group_sim.groupby("quarter")[
            "jobs"
        ].transform("sum")
        # Calculate proportion
        jobs_by_callsign_group_sim["proportion"] = (
            jobs_by_callsign_group_sim["jobs"] / quarter_totals_sim
        )
        jobs_by_callsign_group_sim["what"] = "Simulated"

        full_df_callsign_group_counts = pd.concat(
            [jobs_by_callsign_group_hist, jobs_by_callsign_group_sim]
        )

        if not x_is_callsign_group:
            fig = px.bar(
                full_df_callsign_group_counts,
                title="Historical vs Simulated Split of Jobs Between Callsign Groups",
                color="callsign_group",
                y="proportion",
                x="what",
                barmode="stack",
                labels={
                    "what": "",
                    "proportion": "Percent of Jobs in Quarter",
                    "callsign_group": "Callsign Group",
                },
                facet_col="quarter",
                text=full_df_callsign_group_counts["proportion"].map(
                    lambda p: f"{p:.0%}"
                ),  # format as percent
            )

            fig.update_layout(yaxis_tickformat=".0%")

            fig.update_traces(
                textposition="inside"
            )  # You can also try 'auto' or 'outside'
            return fig

        else:
            fig = px.bar(
                full_df_callsign_group_counts,
                title="Historical vs Simulated Split of Jobs Between Callsign Groups",
                color="what",
                y="proportion",
                x="callsign_group",
                barmode="group",
                labels={
                    "what": "",
                    "proportion": "Percent of Jobs in Quarter",
                    "callsign_group": "Callsign Group",
                },
                facet_col="quarter",
                text=full_df_callsign_group_counts["proportion"].map(
                    lambda p: f"{p:.0%}"
                ),  # format as percent
            )

            fig.update_layout(yaxis_tickformat=".0%")

            fig.update_traces(
                textposition="inside"
            )  # You can also try 'auto' or 'outside'
            return fig

    def PLOT_UTIL_rwc_plot(self) -> Figure:
        #############
        # Prep real-world data
        #############

        jobs_by_callsign_long = (
            self.historical_data.historical_monthly_totals_by_callsign.melt(
                id_vars="month"
            )
            .rename(columns={"variable": "callsign", "value": "jobs"})
            .copy()
        )

        # print(jobs_by_callsign_long)

        all_combinations = pd.MultiIndex.from_product(
            [
                jobs_by_callsign_long["month"].unique(),
                jobs_by_callsign_long["callsign"].unique(),
            ],
            names=["month", "callsign"],
        )

        # Reindex the dataframe to include missing callsigns
        jobs_by_callsign_long = (
            jobs_by_callsign_long.set_index(["month", "callsign"])
            .reindex(all_combinations, fill_value=0)
            .reset_index()
        )

        jobs_by_callsign_long["callsign_group"] = jobs_by_callsign_long[
            "callsign"
        ].str.extract(r"(\d+)")

        jobs_by_callsign_long = jobs_by_callsign_long[
            ~jobs_by_callsign_long["callsign_group"].isna()
        ]

        jobs_by_callsign_long["vehicle_type"] = jobs_by_callsign_long["callsign"].apply(
            lambda x: "car" if "CC" in x else "helicopter"
        )

        # Compute total jobs per callsign_group per month
        jobs_by_callsign_long["total_jobs_per_group"] = jobs_by_callsign_long.groupby(
            ["month", "callsign_group"]
        )["jobs"].transform("sum")

        # Compute percentage of calls per row
        jobs_by_callsign_long["percentage_of_group"] = (
            jobs_by_callsign_long["jobs"]
            / jobs_by_callsign_long["total_jobs_per_group"]
        ) * 100

        # Handle potential division by zero (if total_jobs_per_group is 0)
        jobs_by_callsign_long["percentage_of_group"] = jobs_by_callsign_long[
            "percentage_of_group"
        ].fillna(0)

        # print(jobs_by_callsign_long)

        fig = go.Figure()

        # Bar chart (Simulation Averages)
        for idx, vehicle in enumerate(
            self.sim_averages_utilisation["vehicle_type"].unique()
        ):
            filtered_data = self.sim_averages_utilisation[
                self.sim_averages_utilisation["vehicle_type"] == vehicle
            ]

            fig.add_trace(
                go.Bar(
                    y=filtered_data["percentage_of_group"],
                    x=filtered_data["callsign_group"],
                    name=f"Simulated - {vehicle}",
                    marker=dict(color=list(COLORSCHEME.values())[idx]),
                    width=0.3,
                    opacity=0.6,  # Same opacity for consistency
                    text=[
                        "Simulated:<br>{:.1f}%".format(val)
                        for val in filtered_data["percentage_of_group"]
                    ],  # Correct way            textposition="inside",  # Places text just above the x-axis
                    insidetextanchor="start",  # Anchors text inside the bottom of the bar
                    textfont=dict(color="white"),
                )
            )

            fig.update_layout(
                title="<b>Comparison of Allocated Resources by Callsign Group</b><br>Simulation vs Historical Data",
                yaxis_title="Percentage of Jobs in Callsign Group<br>Tasked to Callsign",
                xaxis_title="Callsign Group",
                barmode="group",
                legend_title="Vehicle Type",
                height=600,
            )

        for callsign in jobs_by_callsign_long["callsign"].unique():
            filtered_data = (
                jobs_by_callsign_long[jobs_by_callsign_long["callsign"] == callsign]
                .groupby(["callsign_group", "vehicle_type", "callsign"])[
                    ["percentage_of_group"]
                ]
                .mean()
                .reset_index()
            )

            if filtered_data["callsign_group"].values[0] in ["70", "71"]:
                expected_x = float(filtered_data["callsign_group"].values[0])
                y_value = filtered_data["percentage_of_group"].values[0]
                expected_y = y_value - 1  # Position for the line

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
                        line=dict(dash="dash", color=COLORSCHEME["charcoal"]),
                    )
                )

                # Add text annotation above the line
                fig.add_trace(
                    go.Scatter(
                        x=[(x_start + x_end) / 2],  # Center the text horizontally
                        y=[expected_y + 5],  # Slightly above the line
                        text=[f"Historical:<br>{y_value:.1f}%"],
                        mode="text",
                        textfont=dict(color="black"),
                        showlegend=False,  # Don't show in legend
                    )
                )

        min_x = min(
            self.sim_averages_utilisation["callsign_group"].astype("int").values
        )
        max_x = max(
            self.sim_averages_utilisation["callsign_group"].astype("int").values
        )

        tick_vals = list(range(min_x, max_x + 1))  # Tick positions at integer values

        fig.update_layout(
            xaxis=dict(
                title=dict(font=dict(size=20)),
                tickfont=dict(size=25),
                tickmode="array",
                tickvals=tick_vals,  # Ensure ticks are at integer positions
                range=[
                    min_x - 0.5,
                    max_x + 0.5,
                ],  # Extend range to start 0.5 units earlier
            ),
            yaxis=dict(
                ticksuffix="%",
                title=dict(dict(font=dict(size=15))),
                tickfont=dict(size=20),
                range=[0, 100],
            ),
            legend=dict(font=dict(size=15)),
            legend_title=dict(font=dict(size=20)),
        )

        return fig

    def prep_util_df_from_call_df(self):
        self.call_df["timestamp_dt"] = pd.to_datetime(
            self.call_df["timestamp_dt"], format="ISO8601"
        )
        self.call_df["month_start"] = (
            self.call_df["timestamp_dt"].dt.to_period("M").dt.to_timestamp()
        )

        # print("==prep_util_df_from_call_df: call_df==")
        # print(call_df)

        jobs_counts_by_callsign_monthly_sim = self.call_df[
            ~self.call_df["callsign"].isna()
        ].copy()

        # print("==jobs_counts_by_callsign_monthly_sim - prior to aggregation==")
        # print(jobs_counts_by_callsign_monthly_sim)

        jobs_counts_by_callsign_monthly_sim["callsign_group"] = (
            jobs_counts_by_callsign_monthly_sim["callsign"].str.extract(r"(\d+)")
        )

        jobs_counts_by_callsign_monthly_sim = (
            jobs_counts_by_callsign_monthly_sim.groupby(
                [
                    "run_number",
                    "month_start",
                    "callsign",
                    "callsign_group",
                    "vehicle_type",
                ]
            )["P_ID"]
            .count()
            .reset_index()
            .rename(columns={"P_ID": "jobs"})
        )

        # print("==jobs_counts_by_callsign_monthly_sim==")
        # print(jobs_counts_by_callsign_monthly_sim)

        all_combinations = pd.MultiIndex.from_product(
            [
                jobs_counts_by_callsign_monthly_sim["month_start"].unique(),
                jobs_counts_by_callsign_monthly_sim["run_number"].unique(),
                jobs_counts_by_callsign_monthly_sim["callsign"].unique(),
            ],
            names=["month_start", "run_number", "callsign"],
        )

        # Reindex the dataframe to include missing callsigns
        jobs_counts_by_callsign_monthly_sim = (
            jobs_counts_by_callsign_monthly_sim.set_index(
                ["month_start", "run_number", "callsign"]
            )
            .reindex(all_combinations, fill_value=0)
            .reset_index()
        )

        jobs_counts_by_callsign_monthly_sim["callsign_group"] = (
            jobs_counts_by_callsign_monthly_sim["callsign"].str.extract(r"(\d+)")
        )
        jobs_counts_by_callsign_monthly_sim["vehicle_type"] = (
            jobs_counts_by_callsign_monthly_sim["callsign"].apply(
                lambda x: "car" if "C" in x else "helicopter"
            )
        )

        # Compute total jobs per callsign_group per month
        jobs_counts_by_callsign_monthly_sim["total_jobs_per_group"] = (
            jobs_counts_by_callsign_monthly_sim.groupby(
                ["month_start", "callsign_group", "run_number"]
            )["jobs"].transform("sum")
        )

        # Compute percentage of calls per row
        jobs_counts_by_callsign_monthly_sim["percentage_of_group"] = (
            jobs_counts_by_callsign_monthly_sim["jobs"]
            / jobs_counts_by_callsign_monthly_sim["total_jobs_per_group"]
        ) * 100

        # Handle potential division by zero (if total_jobs_per_group is 0)
        jobs_counts_by_callsign_monthly_sim["percentage_of_group"] = (
            jobs_counts_by_callsign_monthly_sim["percentage_of_group"].fillna(0)
        )

        self.sim_averages_utilisation = (
            jobs_counts_by_callsign_monthly_sim.groupby(
                ["callsign_group", "callsign", "vehicle_type"]
            )[["percentage_of_group"]]
            .mean()
            .reset_index()
        )

    def make_SIMULATION_stacked_callsign_util_plot(self):
        fig = px.bar(
            self.sim_averages_utilisation,
            x="percentage_of_group",
            y="callsign_group",
            color="callsign",
            height=300,
        )

        # Update axis labels and legend title
        fig.update_layout(
            yaxis=dict(
                title="Callsign Group",
                tickmode="linear",  # Ensures ticks appear at regular intervals
                dtick=1,  # Set tick spacing to 1 unit
            ),
            xaxis=dict(title="Utilisation % within Callsign Group"),
            legend_title="Callsign",
        )

        return fig

    def make_SIMULATION_utilisation_headline_figure(self, vehicle_type):
        """
        Options:
            - helicopter
            - solo car
            - helicopter backup car
        """

        if vehicle_type == "helicopter":
            return self.utilisation_df_overall[
                self.utilisation_df_overall["vehicle_type"] == "helicopter"
            ].mean(numeric_only=True)["perc_time_in_use"]

        else:
            # assume anything with >= 1 entries in a callsign group is helicopter + backup car
            # NOTE: This assumption may not hold forever! It assumes
            vehicles_per_callsign_group = self.utilisation_df_overall.groupby(
                "callsign_group"
            ).count()[["callsign"]]

            if vehicle_type == "solo car":
                car_only = vehicles_per_callsign_group[
                    vehicles_per_callsign_group["callsign"] == 1
                ].copy()

                return self.utilisation_df_overall[
                    self.utilisation_df_overall["callsign_group"].isin(
                        car_only.reset_index().callsign_group.values
                    )
                ].mean(numeric_only=True)["perc_time_in_use"]

            elif vehicle_type == "helicopter backup car":
                backupcar_only = vehicles_per_callsign_group[
                    vehicles_per_callsign_group["callsign"] == 2
                ].copy()

                cars_only = self.utilisation_df_overall[
                    self.utilisation_df_overall["vehicle_type"] == "car"
                ].copy()

                return cars_only[
                    cars_only["callsign_group"].isin(
                        backupcar_only.reset_index().callsign_group.values
                    )
                ].mean(numeric_only=True)["perc_time_in_use"]

            else:
                print(
                    "Invalid vehicle type entered. Please use 'helicopter', 'solo car' or 'helicopter backup car'"
                )

    def create_event_log(self):
        df = self.run_results[self.run_results["event_type"] == "queue"].copy()

        df["activity_id"] = df.groupby("run_number").cumcount() + 1

        # Duplicate rows and modify them
        df_start = df.copy()

        df_start["lifecycle_id"] = "start"

        df_end = df.copy()
        df_end["lifecycle_id"] = "complete"

        # Shift timestamps for 'end' rows
        df_end["timestamp"] = df_end["timestamp"].shift(-1)
        df_end["timestamp_dt"] = df_end["timestamp_dt"].shift(-1)

        # Combine and sort
        df_combined = pd.concat([df_start, df_end]).sort_index(kind="stable")

        # Drop last 'end' row (since there’s no next row to get a timestamp from)
        df_combined = df_combined[:-1]

        df_combined.to_csv("event_log.csv", index=False)

    def display_resource_use_exploration(
        self,
    ):
        """
        Displays the resource use exploration section including dataframes and plots.
        """

        st.subheader("Resource Use")

        # Accounting for odd bug being seen in streamlit community cloud
        # This check might be more robust if done before calling this function,
        # but keeping it here to match original logic if resource_use_events_only_df is passed directly.
        if "P_ID" not in self.resource_use_events_only_df.columns:
            self.resource_use_events_only_df = (
                self.resource_use_events_only_df.reset_index()
            )

        # The @st.fragment decorator is used to group widgets and outputs
        # that should be treated as a single unit for rerun behavior.
        # If you want this behavior, keep it. Otherwise, it can be removed
        # if the function is called within a fragment in the main app.
        # For this refactoring, we'll keep it to ensure similar behavior.
        @st.fragment
        def resource_use_exploration_plots_fragment():
            run_select_ruep = st.selectbox(
                "Choose the run to show",
                self.resource_use_events_only_df["run_number"].unique(),
                key="ruep_run_select",  # Added a key for uniqueness
            )

            # colour_by_cc_ec = st.toggle("Colour the plot by CC/EC/REG patient benefit",
            #                             value=True, key="ruep_color_toggle") # Added a key

            show_outline = st.toggle(
                "Show an outline to help debug overlapping calls",
                value=False,
                key="ruep_outline_toggle",
            )  # Added a key

            with st.expander("Click here to see the timings of resource use"):
                st.dataframe(
                    self.resource_use_events_only_df[
                        self.resource_use_events_only_df["run_number"]
                        == run_select_ruep
                    ]
                )

                st.dataframe(
                    self.resource_use_events_only_df[
                        self.resource_use_events_only_df["run_number"]
                        == run_select_ruep
                    ][["callsign", "callsign_group", "registration"]].value_counts()
                )

                st.dataframe(
                    self.resource_use_events_only_df[
                        self.resource_use_events_only_df["run_number"]
                        == run_select_ruep
                    ][["P_ID", "time_type", "timestamp_dt", "event_type"]]
                    .melt(
                        id_vars=["P_ID", "time_type", "event_type"],
                        value_vars="timestamp_dt",
                    )
                    .drop_duplicates()
                )

                resource_use_wide = (
                    self.resource_use_events_only_df[
                        self.resource_use_events_only_df["run_number"]
                        == run_select_ruep
                    ][
                        [
                            "P_ID",
                            "time_type",
                            "timestamp_dt",
                            "event_type",
                            "registration",
                            "care_cat",
                        ]
                    ]
                    .drop_duplicates()
                    .pivot(
                        columns="event_type",
                        index=["P_ID", "time_type", "registration", "care_cat"],
                        values="timestamp_dt",
                    )
                    .reset_index()
                ).copy()

                # get the number of resources and assign them a value
                resources = resource_use_wide.time_type.unique()
                resources = np.concatenate([resources, ["No Resource Available"]])
                resource_dict = {
                    resource: index for index, resource in enumerate(resources)
                }

                missed_job_events = self.run_results[
                    (
                        self.run_results["run_number"] == run_select_ruep
                    )  # Filter by selected run first
                    & (self.run_results["event_type"] == "resource_request_outcome")
                    & (self.run_results["time_type"] == "No Resource Available")
                ].copy()  # Use .copy() to avoid SettingWithCopyWarning if further modifications are made

                # Check if 'P_ID' is in columns, if not, reset_index (bug handling from original)
                if (
                    "P_ID" not in missed_job_events.columns
                    and not missed_job_events.empty
                ):
                    missed_job_events = missed_job_events.reset_index()

                missed_job_events = missed_job_events[
                    [
                        "P_ID",
                        "time_type",
                        "timestamp_dt",
                        "event_type",
                        "registration",
                        "care_cat",
                    ]
                ].drop_duplicates()
                missed_job_events["event_type"] = "resource_use"

                missed_job_events_end = missed_job_events.copy()
                missed_job_events_end["event_type"] = "resource_use_end"
                missed_job_events_end["timestamp_dt"] = pd.to_datetime(
                    missed_job_events_end["timestamp_dt"]
                ) + datetime.timedelta(minutes=5)

                missed_job_events_full = pd.concat(
                    [missed_job_events, missed_job_events_end]
                )
                missed_job_events_full["registration"] = (
                    "No Resource Available"  # Explicitly set for these events
                )

                if not missed_job_events_full.empty:
                    missed_job_events_full_wide = missed_job_events_full.pivot(
                        columns="event_type",
                        index=["P_ID", "time_type", "registration", "care_cat"],
                        values="timestamp_dt",
                    ).reset_index()
                    resource_use_wide = pd.concat(
                        [resource_use_wide, missed_job_events_full_wide]
                    ).reset_index(drop=True)
                else:
                    # Ensure columns match if missed_job_events_full_wide is empty
                    # This might need more robust handling based on expected columns
                    pass

                resource_use_wide["y_pos"] = resource_use_wide["time_type"].map(
                    resource_dict
                )

                resource_use_wide["resource_use_end"] = pd.to_datetime(
                    resource_use_wide["resource_use_end"]
                )
                resource_use_wide["resource_use"] = pd.to_datetime(
                    resource_use_wide["resource_use"]
                )

                resource_use_wide["duration"] = (
                    resource_use_wide["resource_use_end"]
                    - resource_use_wide["resource_use"]
                )
                resource_use_wide["duration_seconds"] = (
                    resource_use_wide["resource_use_end"]
                    - resource_use_wide["resource_use"]
                ).dt.total_seconds() * 1000
                resource_use_wide["duration_minutes"] = (
                    resource_use_wide["duration_seconds"] / 1000 / 60
                )
                resource_use_wide["duration_minutes"] = resource_use_wide[
                    "duration_minutes"
                ].round(1)

                resource_use_wide["callsign_group"] = resource_use_wide[
                    "time_type"
                ].str.extract(r"(\d+)")  # Added r for raw string

                resource_use_wide = resource_use_wide.sort_values(
                    ["callsign_group", "time_type"]
                )

                st.dataframe(resource_use_wide)

                service_schedule = self.simulation_inputs.service_dates_df.merge(
                    self.simulation_inputs.callsign_registration_lookup_df,
                    on="registration",
                )  # Specify merge key if different

                service_schedule["service_end_date"] = pd.to_datetime(
                    service_schedule["service_end_date"]
                )

                service_schedule["service_start_date"] = pd.to_datetime(
                    service_schedule["service_start_date"]
                )

                service_schedule["duration_seconds"] = (
                    (
                        service_schedule["service_end_date"]
                        - service_schedule["service_start_date"]
                    )
                    + datetime.timedelta(days=1)
                ).dt.total_seconds() * 1000

                service_schedule["duration_days"] = (
                    (service_schedule["duration_seconds"] / 1000) / 60 / 60 / 24
                )

                # Map y_pos using the comprehensive resource_dict
                service_schedule["y_pos"] = service_schedule["callsign"].map(
                    resource_dict
                )
                # Filter out entries that couldn't be mapped if necessary, or handle NaNs
                service_schedule = service_schedule.dropna(subset=["y_pos"])

                # Ensure resource_use_wide is not empty before accessing .min()/.max()
                if (
                    not resource_use_wide.empty
                    and "resource_use" in resource_use_wide.columns
                ):
                    min_date = resource_use_wide.resource_use.min()
                    max_date = resource_use_wide.resource_use.max()
                    service_schedule = service_schedule[
                        (service_schedule["service_start_date"] <= max_date)
                        & (service_schedule["service_end_date"] >= min_date)
                    ]
                else:  # Handle case where resource_use_wide might be empty or missing column
                    service_schedule = pd.DataFrame(
                        columns=service_schedule.columns
                    )  # Empty df with same columns

                st.dataframe(service_schedule)

            # Create figure
            resource_use_fig = go.Figure()

            # Add horizontal bars using actual datetime values
            # Ensure unique callsigns are taken from the sorted resource_use_wide for consistent y-axis order
            unique_time_types_sorted = resource_use_wide.time_type.unique()

            for idx, callsign in enumerate(unique_time_types_sorted):
                callsign_df = resource_use_wide[
                    resource_use_wide["time_type"] == callsign
                ]
                service_schedule_df = service_schedule[
                    service_schedule["callsign"] == callsign
                ]

                # Add in hatched boxes showing the servicing periods
                if not service_schedule_df.empty:
                    resource_use_fig.add_trace(
                        go.Bar(
                            x=service_schedule_df["duration_seconds"],
                            y=service_schedule_df["y_pos"],
                            base=service_schedule_df["service_start_date"],
                            orientation="h",
                            width=0.6,
                            marker_pattern_shape="x",
                            marker=dict(
                                color="rgba(63, 63, 63, 0.30)",
                                line=dict(color="black", width=1),
                            ),
                            name=f"Servicing = {callsign}",
                            customdata=service_schedule_df[
                                [
                                    "callsign",
                                    "duration_days",
                                    "service_start_date",
                                    "service_end_date",
                                    "registration",
                                ]
                            ],
                            hovertemplate="Servicing %{customdata[0]} (registration %{customdata[4]}) lasting %{customdata[1]:.1f} days<br>(%{customdata[2]|%a %-e %b %Y} to %{customdata[3]|%a %-e %b %Y})<extra></extra>",
                        )
                    )

                # if colour_by_cc_ec: # Logic for this needs COLORSCHEME and potentially cc_ec_reg_colour_lookup
                #     if not callsign_df.empty and 'care_cat' in callsign_df.columns:
                #         cc_ec_status = callsign_df["care_cat"].values[0] # This might need adjustment if multiple care_cat per callsign
                #         # cc_ec_reg_colour_lookup would also be needed here
                #         # marker_val = dict(color=list(COLORSCHEME.values())[cc_ec_reg_colour_lookup[cc_ec_status]])
                #     else:
                #         marker_val = dict(color=list(COLORSCHEME.values())[idx % len(COLORSCHEME)])
                # else: # Fallback or default coloring
                if show_outline:
                    marker_val = dict(
                        color=list(COLORSCHEME.values())[
                            idx % len(COLORSCHEME)
                        ],  # Use modulo for safety
                        line=dict(color="#FFA400", width=0.2),
                    )
                else:
                    marker_val = dict(
                        color=list(COLORSCHEME.values())[idx % len(COLORSCHEME)]
                    )  # Use modulo

                # Add in boxes showing the duration of individual calls
                if not callsign_df.empty:
                    resource_use_fig.add_trace(
                        go.Bar(
                            x=callsign_df["duration_seconds"],
                            y=callsign_df["y_pos"],
                            base=callsign_df["resource_use"],
                            orientation="h",
                            width=0.4,
                            marker=marker_val,
                            name=callsign,
                            customdata=callsign_df[
                                [
                                    "resource_use",
                                    "resource_use_end",
                                    "time_type",
                                    "duration_minutes",
                                    "registration",
                                    "care_cat",
                                ]
                            ],
                            hovertemplate="Response to %{customdata[5]} call from %{customdata[2]}<br>(registration %{customdata[4]}) lasting %{customdata[3]:.1f} minutes<br>(%{customdata[0]|%a %-e %b %Y %H:%M} to %{customdata[1]|%a %-e %b %Y %H:%M})<extra></extra>",
                        )
                    )

            # Layout tweaks
            resource_use_fig.update_layout(
                title_text="Resource Use Over Time",  # Changed from title
                barmode="overlay",
                xaxis=dict(
                    title_text="Time",  # Changed from title
                    type="date",
                ),
                yaxis=dict(
                    title_text="Callsign",  # Changed from title
                    tickmode="array",
                    tickvals=list(resource_dict.values()),
                    ticktext=list(
                        resource_dict.keys()
                    ),  # These should be the sorted unique callsigns
                    autorange="reversed",
                ),
                showlegend=True,
                height=700,
            )

            resource_use_fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(
                                count=1, label="1m", step="month", stepmode="backward"
                            ),
                            dict(
                                count=6, label="6m", step="month", stepmode="backward"
                            ),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
            )
            # Ensure the output directory exists
            # import os
            # os.makedirs("app/fig_outputs", exist_ok=True)
            # resource_use_fig.write_html("app/fig_outputs/resource_use_fig.html",full_html=False, include_plotlyjs='cdn')

            st.plotly_chart(
                resource_use_fig,
                use_container_width=True,  # Added for better responsiveness
            )

        # Call the fragment function to render its content
        resource_use_exploration_plots_fragment()

    def get_total_times_model(self, get_summary=False):
        if get_summary:
            utilisation_by_vehicle_summary = (
                self.utilisation_df_overall.groupby("vehicle_type")[
                    "resource_use_duration"
                ]
                .agg(["mean", "median", "min", "max", q10, q90])
                .round(1)
            )
            return utilisation_by_vehicle_summary

        else:
            return self.utilisation_df_overall

    def plot_historical_job_duration_vs_simulation_overall(
        self,
        use_poppins=True,
        write_to_html=False,
        html_output_filepath="fig_job_durations_historical.html",
        violin=False,
    ):
        fig = go.Figure()

        # print(self.historical_data.historical_job_durations_breakdown)
        historical_activity_times_overall = (
            self.historical_data.historical_job_durations_breakdown[
                self.historical_data.historical_job_durations_breakdown["name"]
                == "total_duration"
            ].copy()
        )

        historical_activity_times_overall["what"] = "Historical"

        self.resource_use_wide["what"] = "Simulated"

        # Force 'Simulated' to always appear first (left) and 'Historical' second (right)
        historical_activity_times_overall.rename(
            columns={"value": "resource_use_duration"}, inplace=True
        )

        full_activity_duration_df = pd.concat(
            [historical_activity_times_overall, self.resource_use_wide]
        )

        full_activity_duration_df["what"] = pd.Categorical(
            full_activity_duration_df["what"],
            categories=["Simulated", "Historical"],
            ordered=True,
        )

        if violin:
            fig = px.violin(
                full_activity_duration_df,
                x="vehicle_type",
                y="resource_use_duration",
                color="what",
                category_orders={"what": ["Simulated", "Historical"]},
            )
        else:
            fig = px.box(
                full_activity_duration_df,
                x="vehicle_type",
                y="resource_use_duration",
                color="what",
                category_orders={"what": ["Simulated", "Historical"]},
            )

        fig.update_layout(title="Resource Utilisation Duration vs Historical Averages")

        if write_to_html:
            fig.write_html(
                html_output_filepath, full_html=False, include_plotlyjs="cdn"
            )

        # Adjust font to match DAA style
        if use_poppins:
            fig.update_layout(font=dict(family="Poppins", size=18, color="black"))

        return fig

    def plot_total_times(self, by_run=False):
        if not by_run:
            fig = px.box(
                self.utilisation_df_overall,
                x="resource_use_duration",
                y="vehicle_type",
                color_discrete_sequence=list(COLORSCHEME.values()),
                labels={
                    "resource_use_duration": "Resource Use Duration (minutes)",
                    "vehicle_type": "Vehicle Type",
                },
            )
        else:
            fig = px.box(
                self.utilisation_df_overall,
                x="resource_use_duration",
                y="vehicle_type",
                color="run_number",
                color_discrete_sequence=list(COLORSCHEME.values()),
                labels={
                    "resource_use_duration": "Resource Use Duration (minutes)",
                    "vehicle_type": "Vehicle Type",
                    "run_number": "Run Number",
                },
            )

        return fig

    def plot_total_times_by_hems_or_pt_outcome(
        self, y, color, column_of_interest="hems_result", show_group_averages=True
    ):
        resource_use_wide = (
            self.resource_use_events_only_df[
                [
                    "P_ID",
                    "run_number",
                    "event_type",
                    "timestamp_dt",
                    "callsign_group",
                    "vehicle_type",
                    "callsign",
                    column_of_interest,
                ]
            ]
            .pivot(
                index=[
                    "P_ID",
                    "run_number",
                    "callsign_group",
                    "vehicle_type",
                    "callsign",
                    column_of_interest,
                ],
                columns="event_type",
                values="timestamp_dt",
            )
            .reset_index()
        ).copy()

        # If utilisation start time is missing, then set to start of model + warm-up time (if relevant)
        # as can assume this is a call that started before the warm-up period elapsed but finished
        # after the warm-up period elapsed
        # TODO: need to add in a check to ensure this only happens for calls at the end of the model,
        # not due to errors elsewhere that could fail to assign a resource end time
        resource_use_wide = fill_missing_values(
            resource_use_wide,
            "resource_use",
            get_param("warm_up_end_date", self.params_df),
        )

        # Calculate number of minutes the attending resource was in use on each call
        resource_use_wide["resource_use_duration"] = calculate_time_difference(
            resource_use_wide, "resource_use", "resource_use_end", unit="minutes"
        )

        # Calculate average duration per HEMS result
        mean_durations = (
            resource_use_wide.groupby(y)["resource_use_duration"]
            .mean()
            .sort_values(ascending=True)
        )

        # Create sorted list of HEMS results
        sorted_results = mean_durations.index.tolist()

        fig = px.box(
            resource_use_wide,
            x="resource_use_duration",
            y=y,
            color=color,
            color_discrete_sequence=list(COLORSCHEME.values()),
            category_orders={y: sorted_results},
            labels={
                "resource_use_duration": "Resource Use Duration (minutes)",
                "vehicle_type": "Vehicle Type",
                "hems_result": "HEMS Result",
                "outcome": "Patient Outcome",
                "callsign": "Callsign",
                "callsign_group": "Callsign Group",
            },
            height=900,
        )

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

    def calculate_ks_for_job_durations(
        self, historical_data_series, simulated_data_series, what="cars"
    ):
        statistic, p_value = ks_2samp(historical_data_series, simulated_data_series)

        if p_value > 0.05:
            st.success(f"""
There is no statistically significant difference between
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
                st.info(
                    ks_text_string_sig
                    + f"""Although the difference is
                        statistically significant, the actual magnitude
                        of the difference (D = {format_sigfigs(statistic, 3)}) is small.
                        This suggests the simulation's total job duration pattern is reasonably
                        close to reality.
                        """
                )

            elif statistic < 0.2:
                st.warning(
                    ks_text_string_sig
                    + f"""The KS statistic (D = {format_sigfigs(statistic, 3)})
                        indicates a moderate difference in
                        distribution. You may want to review the simulation model to
                        ensure it adequately reflects real-world variability.
                        """
                )

            else:
                st.error(
                    ks_text_string_sig
                    + f"""The KS statistic (D = {format_sigfigs(statistic, 3)})
                    suggests a large difference in overall job duration patterns.
                    The simulation may not accurately reflect historical
                    patterns and may need adjustment.
                    """
                )

    def PLOT_time_breakdown(self):
        job_times = [
            "time_allocation",
            "time_mobile",
            "time_to_scene",
            "time_on_scene",
            "time_to_hospital",
            "time_to_clear",
        ]

        run_results = self.run_results[self.run_results["event_type"].isin(job_times)][
            ["P_ID", "run_number", "time_type", "event_type", "vehicle_type"]
        ].copy()
        run_results["time_type"] = run_results["time_type"].astype("float")

        self.historical_data.historical_job_durations_breakdown["what"] = "Historical"
        run_results["what"] = "Simulated"

        full_job_duration_breakdown_df = pd.concat(
            [
                run_results.rename(
                    columns={"time_type": "value", "event_type": "name"}
                ).drop(columns=["P_ID", "run_number"]),
                self.historical_data.historical_job_durations_breakdown[
                    self.historical_data.historical_job_durations_breakdown["name"]
                    != "total_duration"
                ].drop(columns=["callsign", "job_identifier"]),
            ]
        )

        full_job_duration_breakdown_df["what"] = pd.Categorical(
            full_job_duration_breakdown_df["what"],
            categories=["Simulated", "Historical"],
            ordered=True,
        )

        full_job_duration_breakdown_df["name"] = (
            full_job_duration_breakdown_df["name"].str.replace("_", " ").str.title()
        )

        fig = px.box(
            full_job_duration_breakdown_df,
            y="value",
            x="name",
            color="what",
            facet_row="vehicle_type",
            category_orders={"what": ["Simulated", "Historical"]},
            labels={
                "value": "Duration (minutes)",
                # "vehicle_type": "Vehicle Type",
                "what": "Time Type (Historical Data vs Simulated Data)",
                "name": "Job Stage",
            },
            title="Comparison of Job Stage Durations by Vehicle Type",
            facet_row_spacing=0.2,
        )

        # Remove default facet titles
        fig.layout.annotations = [
            anno
            for anno in fig.layout.annotations
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
                xref="paper",
                yref="paper",
                x=0.5,
                y=row_heights[i] + 0.02,  # slightly above the subplot
                showarrow=False,
                font=dict(size=14, color="black"),
                xanchor="center",
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

    def RETURN_prediction_cc_patients_sent_ec_resource(self) -> tuple:
        counts_df = (
            self.run_results[self.run_results["event_type"] == "resource_use"][
                ["run_number", "hems_res_category", "care_cat"]
            ]
            .value_counts()
            .reset_index()
        ).copy()

        counts_df_summary = (
            counts_df.groupby(["hems_res_category", "care_cat"])["count"]
            .agg(["mean", "min", "max"])
            .reset_index()
        )

        row_of_interest = counts_df_summary[
            (counts_df_summary["hems_res_category"] != "CC")
            & (counts_df_summary["care_cat"] == "CC")
        ]

        run_duration_days = float(get_param("sim_duration_days", self.params_df))

        return (
            (row_of_interest["mean"].values[0] / run_duration_days) * 365,
            (row_of_interest["min"].values[0] / run_duration_days) * 365,
            (row_of_interest["max"].values[0] / run_duration_days) * 365,
        )

    def RETURN_prediction_heli_benefit_patients_sent_car(self) -> tuple:
        counts_df = (
            self.run_results[self.run_results["event_type"] == "resource_use"][
                ["run_number", "vehicle_type", "heli_benefit"]
            ]
            .value_counts()
            .reset_index()
        ).copy()

        counts_df_summary = (
            counts_df.groupby(["vehicle_type", "heli_benefit"])["count"]
            .agg(["mean", "min", "max"])
            .reset_index()
        )

        row_of_interest = counts_df_summary[
            (counts_df_summary["vehicle_type"] == "car")
            & (counts_df_summary["heli_benefit"] == "y")
        ]

        run_duration_days = float(get_param("sim_duration_days", self.params_df))

        return (
            (row_of_interest["mean"].values[0] / run_duration_days) * 365,
            (row_of_interest["min"].values[0] / run_duration_days) * 365,
            (row_of_interest["max"].values[0] / run_duration_days) * 365,
        )

    def plot_missed_calls_boxplot(
        self,
        historical_results_obj,
        what="breakdown",
        historical_yearly_missed_calls_estimate=None,
    ):
        self.missed_jobs_per_run_breakdown["what"] = "Simulation"

        historical_results_obj.SIM_hist_missed_jobs_care_cat_breakdown["what"] = (
            "Historical (Simulated with Historical Rotas)"
        )

        full_df = pd.concat(
            [
                self.missed_jobs_per_run_breakdown,
                historical_results_obj.SIM_hist_missed_jobs_care_cat_breakdown,
            ]
        )

        full_df_no_resource_avail = full_df[
            full_df["time_type"] == "No Resource Available"
        ]

        if what == "breakdown":
            category_order = ["CC", "EC", "REG - Helicopter Benefit", "REG"]

            fig = px.box(
                full_df_no_resource_avail,
                x="jobs_per_year",
                y="care_cat",
                color="what",
                points="all",  # or "suspectedoutliers"
                boxmode="group",
                height=800,
                labels={
                    "jobs_per_year": "Estimated Average Jobs per Year",
                    "care_cat": "Care Category",
                    "what": "Simulation Results vs Simulated Historical Data",
                },
                category_orders={"care_cat": category_order},
            )

            fig.update_layout(
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
            )

        if what == "summary":
            full_df_no_resource_avail_per_run = (
                full_df_no_resource_avail.groupby(["run_number", "what"])[
                    ["jobs_per_year"]
                ]
                .sum()
                .reset_index()
            )

            # Compute data bounds for x-axis
            x_min = full_df_no_resource_avail_per_run["jobs_per_year"].min()
            x_max = full_df_no_resource_avail_per_run["jobs_per_year"].max()
            padding = 0.20 * (x_max - x_min)
            x_range = [x_min - padding, x_max + padding]

            fig = px.box(
                full_df_no_resource_avail_per_run,
                x="jobs_per_year",
                y="what",
                color="what",
                points="all",
                boxmode="group",
                height=400,
            )

            # Update x-axis range
            fig.update_layout(xaxis_range=x_range, showlegend=False)

            if historical_yearly_missed_calls_estimate is not None:
                # Add the dotted vertical line
                fig.add_vline(
                    x=historical_yearly_missed_calls_estimate,
                    line_dash="dot",
                    line_color="black",
                    line_width=2,
                    annotation_text=f"Historical Estimate: {historical_yearly_missed_calls_estimate:.0f}",
                    annotation_position="top",
                )

            # Step 1: Compute Q1 and Q3
            q_df = (
                full_df_no_resource_avail_per_run.groupby("what")["jobs_per_year"]
                .quantile([0.25, 0.5, 0.75])
                .unstack()
                .reset_index()
                .rename(columns={0.25: "q1", 0.5: "median", 0.75: "q3"})
            )

            # Step 2: Calculate IQR and upper whisker cap
            q_df["iqr"] = q_df["q3"] - q_df["q1"]
            q_df["upper_whisker_cap"] = q_df["q3"] + 1.5 * q_df["iqr"]

            # Step 3: Find the max non-outlier per group
            max_non_outliers = full_df_no_resource_avail_per_run.merge(
                q_df[["what", "upper_whisker_cap"]], on="what"
            )
            max_non_outliers = (
                max_non_outliers[
                    max_non_outliers["jobs_per_year"]
                    <= max_non_outliers["upper_whisker_cap"]
                ]
                .groupby("what")["jobs_per_year"]
                .max()
                .reset_index()
                .rename(columns={"jobs_per_year": "max_non_outlier"})
            )

            # Step 4: Merge with median data
            annot_df = pd.merge(q_df[["what", "median"]], max_non_outliers, on="what")

            # Step 5: Add annotations just to the right of the whisker
            for _, row in annot_df.iterrows():
                fig.add_annotation(
                    x=row["max_non_outlier"] + padding * 0.1,
                    y=row["what"],
                    text=f"Median: {row['median']:.0f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=0,
                    font=dict(size=12, color="gray"),
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1,
                )

        return fig

    def RETURN_missed_jobs_fig(self, care_category, what="average"):
        row = self.missed_jobs_per_run_care_cat_summary[
            (self.missed_jobs_per_run_care_cat_summary["care_cat"] == care_category)
            & (
                self.missed_jobs_per_run_care_cat_summary["time_type"]
                == "No Resource Available"
            )
        ]
        if what == "average":
            return row["jobs_per_year_average"].values[0]
        elif what == "min":
            return row["jobs_per_year_min"].values[0]
        elif what == "max":
            return row["jobs_per_year_max"].values[0]

    def PLOT_days_with_job_count_hist_ks(self):
        daily_call_counts = (
            self.call_df.groupby(["run_number", "day_date"])["P_ID"]
            .agg("count")
            .reset_index()
            .rename(columns={"P_ID": "Calls per Day"})
        )

        # Create histogram with two traces
        call_count_hist = go.Figure()

        # Simulated data
        call_count_hist.add_trace(
            go.Histogram(
                x=daily_call_counts["Calls per Day"],
                name="Simulated",
                histnorm="percent",
                xbins=dict(  # bins used for histogram
                    start=0.0,
                    end=max(daily_call_counts["Calls per Day"]) + 1,
                    size=1.0,
                ),
                opacity=0.75,
            )
        )

        # Historical data
        call_count_hist.add_trace(
            go.Histogram(
                x=self.historical_data.historical_daily_calls_breakdown["calls_in_day"],
                xbins=dict(  # bins used for histogram
                    start=0.0,
                    end=max(
                        self.historical_data.historical_daily_calls_breakdown[
                            "calls_in_day"
                        ]
                    )
                    + 1,
                    size=1.0,
                ),
                name="Historical",
                histnorm="percent",
                opacity=0.75,
            )
        )

        # Update layout
        call_count_hist.update_layout(
            title="Distribution of Jobs Per Day: Simulated vs Historical",
            barmode="overlay",
            bargap=0.03,
            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        )

        # Save and display
        call_count_hist.write_html(
            "app/fig_outputs/daily_calls_dist_histogram.html",
            full_html=False,
            include_plotlyjs="cdn",
        )

        call_count_hist.update_layout(
            font=dict(family="Poppins", size=18, color="black")
        )

        st.plotly_chart(call_count_hist)

        st.caption("""
This plot looks at the number of days across all repeats of the simulation where each given number of calls was observed (i.e. on how many days was one call received, two calls, three calls, and so on).
                        """)

        statistic, p_value = ks_2samp(
            daily_call_counts["Calls per Day"],
            self.historical_data.historical_daily_calls_breakdown["calls_in_day"],
        )

        if p_value > 0.05:
            st.success(f"""There is no statistically significant difference between
                                the distributions of call data from historical data and the
                                simulation (p = {format_sigfigs(p_value)})

                                This means that the pattern of calls produced by the simulation
                                matches the pattern seen in the real-world data —
                                for example, the frequency or variability of daily calls
                                is sufficiently similar to what has been observed historically.
                                """)
        else:
            ks_text_string_sig = f"""
There is a statistically significant difference between the
distributions of call data from historical data and
the simulation (p = {format_sigfigs(p_value)}).

This means that the pattern of calls produced by the simulation
does not match the pattern seen in the real-world data —
for example, the frequency or variability of daily calls
may be different.

The simulation may need to be adjusted to better
reflect the patterns of demand observed historically.

"""

            if statistic < 0.1:
                st.info(
                    ks_text_string_sig
                    + f"""Although the difference is
                                statistically significant, the actual magnitude
                                of the difference (D = {format_sigfigs(statistic)}) is small.
                                This suggests the simulation's call volume pattern is reasonably
                                close to reality.
                                """
                )

            elif statistic < 0.2:
                st.warning(
                    ks_text_string_sig
                    + f"""The KS statistic (D = {format_sigfigs(statistic)})
                                indicates a moderate difference in
                                distribution. You may want to review the simulation model to
                                ensure it adequately reflects real-world variability.
                                """
                )

            else:
                st.error(
                    ks_text_string_sig
                    + f"""The KS statistic (D = {format_sigfigs(statistic)})
                                suggests a large difference in call volume patterns.
                                The simulation may not accurately reflect historical
                                demand and may need adjustment.
                                """
                )

    def PLOT_daily_availability(self):
        return px.bar(
            self.daily_availability_df,
            x="month",
            y="theoretical_availability",
            facet_row="callsign",
        )

    def PLOT_events_over_time(self, runs=None):
        events_over_time_df = self.run_results[
            self.run_results["run_number"].isin(runs)
        ].copy()

        # Fix to deal with odd community cloud indexing bug
        if "P_ID" not in events_over_time_df.columns:
            events_over_time_df = events_over_time_df.reset_index()

        events_over_time_df["time_type"] = events_over_time_df["time_type"].astype(
            "str"
        )

        fig = px.scatter(
            events_over_time_df,
            x="timestamp_dt",
            y="time_type",
            # facet_row="run_number",
            # showlegend=False,
            color="time_type",
            height=800,
            title="Events Over Time - By Run",
        )

        fig.update_traces(marker=dict(size=3, opacity=0.5))

        fig.update_layout(
            yaxis_title="",  # Remove y-axis label
            yaxis_type="category",
            showlegend=False,
        )
        # Remove facet labels
        fig.for_each_annotation(lambda x: x.update(text=""))

        return fig

    def PLOT_cumulative_arrivals_per_run(self):
        return px.line(
            self.run_results[self.run_results["time_type"] == "arrival"],
            x="timestamp_dt",
            y="P_ID",
            color="run_number",
            height=800,
            title="Cumulative Arrivals Per Run",
        )

    def get_event_counts(self):
        self.event_counts_df = (
            pd.DataFrame(self.run_results[["run_number", "time_type"]].value_counts())
            .reset_index()
            .pivot(index="run_number", columns="time_type", values="count")
        )
        self.event_counts_long = self.event_counts_df.reset_index(drop=False).melt(
            id_vars="run_number"
        )

    def PLOT_event_funnel_plot(self, hems_events, run_select):
        return px.funnel(
            self.event_counts_long[
                (self.event_counts_long["time_type"].isin(hems_events))
                & (self.event_counts_long["run_number"].isin(run_select))
            ],
            facet_col="run_number",
            x="value",
            y="time_type",
            category_orders={"time_type": hems_events[::-1]},
        )

    def PLOT_per_patient_events(self, patient_df):
        fig = px.scatter(patient_df, x="timestamp_dt", y="time_type", color="time_type")

        fig.update_layout(yaxis_type="category")

        return fig

    def PLOT_outcome_variation_across_day(self, y_col):
        hourly_hems_result_counts = (
            self.run_results[self.run_results["time_type"] == "HEMS call start"]
            .groupby(["hems_result", "hour"])
            .size()
            .reset_index(name="count")
        ).copy()

        total_per_group = hourly_hems_result_counts.groupby("hour")["count"].transform(
            "sum"
        )

        hourly_hems_result_counts["proportion"] = (
            hourly_hems_result_counts["count"] / total_per_group
        )

        return px.bar(
            hourly_hems_result_counts,
            x="hour",
            y=y_col,
            color="hems_result",
        )
