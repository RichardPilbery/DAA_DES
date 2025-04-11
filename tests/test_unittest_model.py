"""Unit testing for the Discrete-Event Simulation (DES) Model.

These check specific parts of the simulation and code, ensuring they work
correctly and as expected.

Many tests inspired by list here:
https://github.com/pythonhealthdatascience/rap_template_python_des/blob/main/tests/test_unittest_model.py

Planned tests are listed below with a [].
Implemented tests are listed below with a [x].

## General Checks

[] Results dataframe is longer when model is run for longer


## Multiple Runs

[] Results dataframe is longer when model conducts more runs
[] Results differ across multiple runs
[] Arrivals differ across multiple runs
[] Test running the model sequentially and in parallel produce identical results when seeds set

## Seeds

[] Model behaves consistently across repeated runs when provided with a seed

## Warm-up period impact

[] Data is not present in results dataframe for warm-up period
[x] Results dataframe is empty if only a warm-up period is provided

## Arrivals

[] All patients who arrive outside of the warm-up period have an entry in the results dataframe
[] Number of arrivals increase if parameter adjusted
[] Number of arrivals decrease if parameter adjusted

## Sensible Resource Use

[] All provided resources get at least some utilisation in a model that
   runs for a sufficient length of time with sufficient demand
[] Utilisation never exceeds 100%
[] Utilisation never drops below 0%
[] No one waits in the model for a resource to become availabile - they leave and are recorded as missed
[] Resources are used in the expected order determined within the model
[x] The same callsign is never sent on two jobs at once
[x] Resources belonging to the same callsign group don't get sent on jobs at the same time
[] Changing helicopter type results in different unavailability results being generated
[] Resources don't leave on service and never return

## Activity during inactive periods

[x] Resources do not respond during times they are meant to be off shift
[] Resources aren't used during their service interval (determined by reg, not callsign)
[] Calls do not generate activity if they arrive during times the resource is meant to be off shift
[] Inactive periods correctly change across seasons if set to do so

## Expected responses of metrics under different conditions

[] Utilisation is higher when resource is reduced but demand kept consistent
[] 'Missed' calls are higher when resource is reduced but demand kept consistent

[] Utilisation is lower when resource is reduced but demand decreases
[] 'Missed' calls are lower when resource is reduced but demand decreases

[] Utilisation is higher when resource is kept consistent but demand increases
[] 'Missed' calls are higher when resource is kept consistent but demand increases

[] Utilisation is lower when resource is kept consistent but demand decreases
[] 'Missed' calls are lower when resource is kept consistent but demand decreases

## Failure to run under nonsensical conditions

[] Model does not run with a negative number of resources
[] Model does not run with negative average demand parameter

"""

import pandas as pd
import pytest
from datetime import datetime

# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from des_parallel_process import parallelProcessJoblib, collateRunResults

def test_warmup_only():
   """
   Ensures no results are recorded during the warm-up phase.

   This is tested by running the simulation model with only a warm-up period,
   and then checking that results are all zero or empty.
   """
   parallelProcessJoblib(
      total_runs=5,
      sim_duration=0,
      warm_up_time=24*7*60,
      sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
      amb_data=False
      )

   collateRunResults()

   results = pd.read_csv("data/run_results.csv")

   assert len(results) == 0, "Results seem to have been generated during the warm-up period"


def test_simultaneous_allocation_same_resource_group():
      parallelProcessJoblib(
         total_runs=2,
         sim_duration= 60 * 24 * 7 * 10,
         warm_up_time=0,
         sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
         amb_data=False
      )

      collateRunResults()

      results = pd.read_csv("data/run_results.csv")

      resource_use_start_and_end = results[results["event_type"].isin(["resource_use","resource_use_end"])][['P_ID','run_number','event_type','callsign','callsign_group','timestamp_dt']]

      resource_use_start = resource_use_start_and_end[resource_use_start_and_end["event_type"] == "resource_use"].rename(columns={'timestamp_dt':'resource_use_start'}).drop(columns="event_type")
      resource_use_end = resource_use_start_and_end[resource_use_start_and_end["event_type"] == "resource_use_end"].rename(columns={'timestamp_dt':'resource_use_end'}).drop(columns="event_type")

      resource_use_wide = resource_use_start.merge(resource_use_end, how="outer", on=["P_ID","run_number","callsign", "callsign_group"]).sort_values(["run_number", "P_ID"])

      resource_use_wide['resource_use_start'] = pd.to_datetime(resource_use_wide['resource_use_start'])
      resource_use_wide['resource_use_end'] = pd.to_datetime(resource_use_wide['resource_use_end'])

      callsign_groups = resource_use_wide["callsign_group"].unique()

      all_overlaps = []

      for callsign_group in callsign_groups:

         single_callsign = resource_use_wide[resource_use_wide["callsign_group"]==callsign_group]

         # Sort by group and start time
         df_sorted = single_callsign.sort_values(by=["callsign_group", "resource_use_start"])

         # Shift end times within each group to compare with the next start
         df_sorted["prev_end"] = df_sorted.groupby("callsign_group")["resource_use_end"].shift()

         # Find overlaps
         df_sorted["overlap"] = df_sorted["resource_use_start"] < df_sorted["prev_end"]

         # Filter to overlapping rows
         overlaps = df_sorted[df_sorted["overlap"]]

         print(f"Callsign Group {callsign_group} - jobs: {len(single_callsign)}")
         print(f"Callsign Group {callsign_group} - overlaps: {len(overlaps)}")

         all_overlaps.append(overlaps)

      all_overlaps_df = pd.concat(all_overlaps)

      assert len(all_overlaps_df) == 0, "Instances found of resources from the same callsign group being sent on two or more jobs at once"


def test_simultaneous_allocation_same_resource():
      parallelProcessJoblib(
         total_runs=2,
         sim_duration= 60 * 24 * 7 * 10,
         warm_up_time=0,
         sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
         amb_data=False
      )

      collateRunResults()

      results = pd.read_csv("data/run_results.csv")

      resource_use_start_and_end = results[results["event_type"].isin(["resource_use","resource_use_end"])][['P_ID','run_number','event_type','callsign','callsign_group','timestamp_dt']]

      resource_use_start = resource_use_start_and_end[resource_use_start_and_end["event_type"] == "resource_use"].rename(columns={'timestamp_dt':'resource_use_start'}).drop(columns="event_type")
      resource_use_end = resource_use_start_and_end[resource_use_start_and_end["event_type"] == "resource_use_end"].rename(columns={'timestamp_dt':'resource_use_end'}).drop(columns="event_type")

      resource_use_wide = resource_use_start.merge(resource_use_end, how="outer", on=["P_ID","run_number","callsign", "callsign_group"]).sort_values(["run_number", "P_ID"])

      resource_use_wide['resource_use_start'] = pd.to_datetime(resource_use_wide['resource_use_start'])
      resource_use_wide['resource_use_end'] = pd.to_datetime(resource_use_wide['resource_use_end'])

      callsigns = resource_use_wide["callsign"].unique()

      all_overlaps = []

      for callsign in callsigns:

         single_callsign = resource_use_wide[resource_use_wide["callsign_group"]==callsign]

         # Sort by group and start time
         df_sorted = single_callsign.sort_values(by=["callsign_group", "resource_use_start"])

         # Shift end times within each group to compare with the next start
         df_sorted["prev_end"] = df_sorted.groupby("callsign_group")["resource_use_end"].shift()

         # Find overlaps
         df_sorted["overlap"] = df_sorted["resource_use_start"] < df_sorted["prev_end"]

         # Filter to overlapping rows
         overlaps = df_sorted[df_sorted["overlap"]]

         print(f"Callsign {callsign} - instances: {len(single_callsign)}")
         print(f"Callsign {callsign} - overlaps: {len(overlaps)}")

         all_overlaps.append(overlaps)

      all_overlaps_df = pd.concat(all_overlaps)

      assert len(all_overlaps_df) == 0, "Instances found of resources being sent on two or more jobs at once"


def test_no_response_during_off_shift_times():
      parallelProcessJoblib(
         total_runs=2,
         sim_duration= 60 * 24 * 7 * 10,
         warm_up_time=0,
         sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
         amb_data=False
      )

      collateRunResults()

      results = pd.read_csv("data/run_results.csv")

      resource_use_start = (
          results[results["event_type"] == "resource_use"]
          .rename(columns={'timestamp_dt':'resource_use_start'})
          [['P_ID','run_number','callsign','resource_use_start', 'day', 'hour','month','qtr']]
          )

      resource_use_start['resource_use_start'] = pd.to_datetime(resource_use_start['resource_use_start'])

      hems_rota_df = pd.read_csv("tests/HEMS_ROTA_test.csv")

      def normalize_hour_range(start, end):
         """Handles overnight hours by mapping to a 0–47 scale (so 2am next day = 26)"""
         if end <= start:
            end += 24
         return start, end

      # Make a copy of the rota and normalize hours
      rota_df = hems_rota_df.copy()
      rota_df[['summer_start', 'summer_end']] = rota_df[['summer_start', 'summer_end']].apply(
         lambda col: pd.to_numeric(col, errors='coerce'))

      rota_df[['winter_start', 'winter_end']] = rota_df[['winter_start', 'winter_end']].apply(
         lambda col: pd.to_numeric(col, errors='coerce'))

      # Group by callsign, take min start and max end, but account for overnight shifts
      def combine_shifts(group):
         summer_starts, summer_ends = zip(*[normalize_hour_range(s, e) for s, e in zip(group['summer_start'], group['summer_end'])])
         winter_starts, winter_ends = zip(*[normalize_hour_range(s, e) for s, e in zip(group['winter_start'], group['winter_end'])])

         return pd.Series({
            'summer_start': min(summer_starts),
            'summer_end': max(summer_ends),
            'winter_start': min(winter_starts),
            'winter_end': max(winter_ends),
         })

      rota_simplified = rota_df.groupby('callsign').apply(combine_shifts, include_groups=False).reset_index()

      merged_df = pd.merge(resource_use_start, rota_simplified, on='callsign', how='left')

      def is_summer(month):
         return 4 <= month <= 9

      def check_if_available(row):
         if is_summer(row['month']):
            start = row['summer_start']
            end = row['summer_end']
         else:
            start = row['winter_start']
            end = row['winter_end']

         hour = row['hour']
         hour_extended = hour if hour >= start else hour + 24  # extend into next day if needed

         return start <= hour_extended < end

      # Apply the function to determine if the call is offline
      # check_if_available(...) returns True if the resource is available for that call.
      # Applying ~ in front of that means:
         # “Store True in is_offline when the resource is NOT available.”
      # So:
      # True from check_if_available ➝ False in is_offline
      # False from check_if_available ➝ True in is_offline
      merged_df['is_offline'] = ~merged_df.apply(check_if_available, axis=1)

      # Filter the DataFrame to get only the offline calls
      offline_calls = merged_df[merged_df['is_offline']]

      # Check there are no offline calls
      assert len(offline_calls)==0, "Calls appear to have had a response initiated outside of rota'd hours"


      # Add several test cases that should fail to the dataframe and rerun to ensure that
      # the test is actually written correctly as well!

      additional_rows = pd.DataFrame(
         [{
            'P_ID': 99999,
            'run_number': 1,
            'callsign': 'H70',
            'resource_use_start': "2024-01-01 04:00:00",
            'day': 	'Mon',
            'hour': 4,
            'month': 1,
            'qtr': 1
         },
         {
            'P_ID': 99998,
            'run_number': 1,
            'callsign': 'CC71',
            'resource_use_start': "2024-01-01 04:00:00",
            'day': 	'Mon',
            'hour': 4,
            'month': 1,
            'qtr': 1
         },
         {
            'P_ID': 99997,
            'run_number': 1,
            'callsign': 'CC71',
            'resource_use_start': "2024-01-01 22:00:00",
            'day': 	'Mon',
            'hour': 22,
            'month': 1,
            'qtr': 1
         },
         {
            'P_ID': 99996,
            'run_number': 1,
            'callsign': 'CC72',
            'resource_use_start': "2024-01-01 07:00:00",
            'day': 	'Mon',
            'hour': 7,
            'month': 1,
            'qtr': 1
         },
         # Also add an extra row that should pass
         {
            'P_ID': 99995,
            'run_number': 1,
            'callsign': 'CC72',
            'resource_use_start': "2024-01-01 09:00:00",
            'day': 	'Mon',
            'hour': 9,
            'month': 1,
            'qtr': 1
         }
         ]
      )

      resource_use_start = pd.concat([resource_use_start, additional_rows])

      merged_df = pd.merge(resource_use_start, rota_simplified, on='callsign', how='left')
      merged_df['is_offline'] = ~merged_df.apply(check_if_available, axis=1)

      # Filter the DataFrame to get only the offline calls
      offline_calls = merged_df[merged_df['is_offline']]

      assert len(offline_calls) == (len(additional_rows) - 1), "The function for testing resources being allocated out of rota'd hours is not behaving correctly"
