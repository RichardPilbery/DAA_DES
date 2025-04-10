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
[] Resources belonging to the same callsign group don't get sent on jobs at the same time
[] Changing helicopter type results in different unavailability results being generated

## Activity during inactive periods

[] Resources do not respond during times they are meant to be off shift
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
      sim_duration=24*7*60,
      warm_up_time=24*7*60,
      sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
      amb_data=False
      )

   collateRunResults()

   results = pd.read_csv("data/run_results.csv")

   assert len(results) == 0


def test_simultaneous_allocation_same_resource_group():
      parallelProcessJoblib(
         total_runs=5,
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

         print(f"Callsign Group {callsign_group} - instances: {len(single_callsign)}")
         print(f"Callsign Group {callsign_group} - overlaps: {len(overlaps)}")

         assert len(overlaps) == 0
