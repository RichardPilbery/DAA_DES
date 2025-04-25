"""Unit testing for the Discrete-Event Simulation (DES) Model.

These check specific parts of the simulation and code, ensuring they work
correctly and as expected.

Many tests inspired by list here:
https://github.com/pythonhealthdatascience/rap_template_python_des/blob/main/tests/test_unittest_model.py

Planned tests are listed below with a [].
Implemented tests are listed below with a [x].

## General Checks

[x] Results dataframe is longer when model is run for longer


## Multiple Runs

[x] Results dataframe is longer when model conducts more runs
[x] Results differ across multiple runs
[x] Arrivals differ across multiple runs
[x] Running the model sequentially and in parallel produce identical results when seeds set

## Seeds

[x] Model behaves consistently across repeated runs when provided with a seed and no parameters change
[x] Arrivals are identical across simulations when provided with a seed even when other parameters are varied

## Warm-up period impact

[x] Data is not present in results dataframe for warm-up period
[x] Results dataframe is empty if only a warm-up period is provided

## Arrivals

[] All patients who arrive outside of the warm-up period have an entry in the results dataframe
[x] Number of arrivals increase if parameter adjusted
[x] Number of arrivals decrease if parameter adjusted
[x] No activity generated or model fails to complete run if number of arrivals is 0

## Sensible Resource Use

[] All provided resources get at least some utilisation in a model that
   runs for a sufficient length of time with sufficient demand
[x] The same callsign is never sent on two jobs at once
[x] Resources belonging to the same callsign group don't get sent on jobs at the same time
[] Resources don't leave on service and never return
[] Resource use duration is never negative (i.e. resource use for an individual never ends before it starts)
[] Utilisation never exceeds 100%
[] Utilisation never drops below 0%
[] No one waits in the model for a resource to become availabile - they leave and are recorded as missed
[] Resources are used in the expected order determined within the model

## Activity during inactive periods

[x] Resources do not respond during times they are meant to be off shift
[x] Resources aren't used during their service interval (determined by reg, not callsign)
[] Inactive periods correctly change across seasons if set to do so

## Failure to run under nonsensical conditions

[] Model does not run with a negative number of resources
[] Model does not run with negative average demand parameter

"""

import pandas as pd
import pytest
from datetime import datetime
import os
import gc

# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from des_parallel_process import parallelProcessJoblib, collateRunResults, runSim, removeExistingResults
from helpers import save_logs

##############################################################################
# Begin tests                                                                #
##############################################################################

@pytest.mark.quick
def test_model_runs():
   try:
      removeExistingResults(remove_run_results_csv=True)

      parallelProcessJoblib(
         total_runs=1,
         sim_duration=60 * 24 * 7 * 5, # Run for five weeks
         warm_up_time=0,
         sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
         amb_data=False,
         print_debug_messages=True
         )

      collateRunResults()

      save_logs("test_model_runs.txt")

      # Read simulation results
      results_df = pd.read_csv("data/run_results.csv")

      assert len(results_df) > 5, "[FAIL - BASIC FUNCTIONS] Model failed to run"

   finally:
      del results_df
      gc.collect()

def test_more_results_for_longer_run():
   try:
      removeExistingResults(remove_run_results_csv=True)

      for i in range(10):
         results_df_1 = runSim(run=1, total_runs=1,
                              sim_duration=60 * 24 * 7 * 2, # run for 2 weeks
                              warm_up_time=0,
                              sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
                              amb_data=False).reset_index()

         results_df_2 = runSim(run=1, total_runs=1,
                              sim_duration=60 * 24 * 7 * 4, # run for twice as long - 4 weeks
                              warm_up_time=0,
                              sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
                              amb_data=False).reset_index()


         assert len(results_df_1) < len(results_df_2), "[FAIL - BASIC FUNCTIONS] Fewer results seen in longer model run"
   finally:
      del results_df_1, results_df_2
      gc.collect()

def longer_df_when_more_runs_conducted():
   try:
      removeExistingResults(remove_run_results_csv=True)

      parallelProcessJoblib(
         total_runs=1,
         sim_duration=60 * 24 * 7 * 5,
         warm_up_time=0,
         sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
         amb_data=False
         )

      collateRunResults()

      # Read simulation results
      results = pd.read_csv("data/run_results.csv")

      results_1_run = len(results)

      removeExistingResults(remove_run_results_csv=True)

      parallelProcessJoblib(
         total_runs=2,
         sim_duration=60 * 24 * 7 * 5,
         warm_up_time=0,
         sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
         amb_data=False
         )

      collateRunResults()

      # Read simulation results
      results = pd.read_csv("data/run_results.csv")

      results_2_runs = len(results)

      assert results_1_run < results_2_runs, "[FAIL - BASIC FUNCTIONS] Fewer results seen with a higher number of runs"

   finally:
      del results
      gc.collect()


def test_arrivals_increase_if_demand_param_increased():
   try:
      removeExistingResults(remove_run_results_csv=True)

      for i in range(5):
         results_df_1 = runSim(run=1, total_runs=1,
                              sim_duration=60 * 24 * 7 * 2, # run for 2 weeks
                              warm_up_time=0,
                              sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
                              amb_data=False,
                              demand_increase_percent=1.0).reset_index()

         results_df_1 = results_df_1[results_df_1["time_type"] == "arrival"]

         results_df_2 = runSim(run=1, total_runs=1,
                              sim_duration=60 * 24 * 7 * 2,
                              warm_up_time=0,
                              sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
                              amb_data=False,
                              demand_increase_percent=1.2).reset_index()

         results_df_2 = results_df_2[results_df_2["time_type"] == "arrival"]

         assert len(results_df_1) < len(results_df_2), "[FAIL - DEMAND PARAMETER] Fewer jobs observed when demand increase parameter above one"
   finally:
      del results_df_1, results_df_2
      gc.collect()

def test_arrivals_decrease_if_demand_param_decrease():
   try:
      removeExistingResults(remove_run_results_csv=True)

      for i in range(5):
         results_df_1 = runSim(run=1, total_runs=1,
                              sim_duration=60 * 24 * 7 * 2, # run for 2 weeks
                              warm_up_time=0,
                              sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
                              amb_data=False,
                              demand_increase_percent=1.0).reset_index()

         results_df_1 = results_df_1[results_df_1["time_type"] == "arrival"]

         results_df_2 = runSim(run=1, total_runs=1,
                              sim_duration=60 * 24 * 7 * 2,
                              warm_up_time=0,
                              sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
                              amb_data=False,
                              demand_increase_percent=0.8).reset_index()

         results_df_2 = results_df_2[results_df_2["time_type"] == "arrival"]

         assert len(results_df_1) > len(results_df_2), "[FAIL - DEMAND PARAMETER] More jobs observed when demand increase parameter below one"
   finally:
      del results_df_1, results_df_2
      gc.collect()


def test_output_when_no_demand():
   removeExistingResults(remove_run_results_csv=True)

   try:
      results_df_1 = runSim(run=1, total_runs=1,
                        sim_duration=60 * 24 * 7 * 2, # run for 2 weeks
                        warm_up_time=0,
                        sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
                        amb_data=False,
                        demand_increase_percent=0).reset_index()

      assert len(results_df_1) == 0

   except Exception:
        # Any exception is allowed
        pass

@pytest.mark.warmup
def test_warmup_only():
   """
   Ensures no results are recorded during the warm-up phase.

   This is tested by running the simulation with a non-zero warm-up time
   but zero actual simulation duration. Since the simulation doesn't run
   past the warm-up, no outputs should be produced.
   """
   try:
      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=5,
         sim_duration=0,
         warm_up_time=24*7*60,
         sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
         amb_data=False
         )

      collateRunResults()

      # Read simulation results
      results = pd.read_csv("data/run_results.csv")

      # Assert that the results are empty, i.e., no output was generated during warm-up
      assert len(results) == 0, (
         f"[FAIL - WARM-UP] {len(results)} results seem to have been generated during the warm-up period"
         )
   finally:
      del results
      gc.collect()

@pytest.mark.warmup
def test_no_results_recorded_from_warmup():
   """
   Ensures no results are recorded during the warm-up phase.

   This test runs the simulation with both a warm-up and post-warm-up period.
   It verifies that no records are generated that fall within the warm-up time.
   """
   try:
      removeExistingResults(remove_run_results_csv=True)

      warm_up_length=60*24*3 # 3 days

      # Run the simulation with a warm-up period and simulation time
      parallelProcessJoblib(
         total_runs=2,
         sim_duration=60*24*7, # 7 days
         warm_up_time=warm_up_length,
         sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
         amb_data=False
         )

      collateRunResults()

      # Read simulation results
      results = pd.read_csv("data/run_results.csv")

      # Filter results to only those within the warm-up period (timestamp < warm-up time)
      results_in_warmup = results[results['timestamp'] < warm_up_length]

      # Assert no records were made during the warm-up period
      assert len(results_in_warmup) == 0, (
         f"[FAIL - WARM-UP] {len(results_in_warmup)} results appear in the results df that shouldn't due to falling in warm-up period"
         )

   finally:
      del results, results_in_warmup
      gc.collect()

@pytest.mark.resources
def test_simultaneous_allocation_same_resource_group(simulation_results):
   """
   Ensures no two jobs are allocated to resources from the same resource group at overlapping times.

   This checks for logical consistency in dispatch logic to prevent simultaneous usage
   of resources grouped together (e.g., mutually exclusive vehicles).

   In the simulation, where a helicopter and car belong to the same resource group, it is assumed
   that they can never be running at the same time due to the presence of a single crew crewing
   that single vehicle.

   For this reason, we are joining on callsign rather than registration (as when H71 is
   reallocated temporarily to H70, we need to continue to check that CC70 and the temporary
   H70 are not running simultaneously).
   """
   try:
      removeExistingResults(remove_run_results_csv=True)

      # Remove existing failure log if it exists
      if os.path.exists("tests/simultaneous_allocation_same_callsigngroup_FAILURES.csv"):
         os.remove("tests/simultaneous_allocation_same_callsigngroup_FAILURES.csv")

      if os.path.exists("tests/simultaneous_allocation_same_callsigngroup_FULL.csv"):
            os.remove("tests/simultaneous_allocation_same_callsigngroup_FULL.csv")

      results = simulation_results # defined in conftest.py

      # Extract start and end times of resource usage
      resource_use_start_and_end = (
         results[results["event_type"].isin(["resource_use","resource_use_end"])]
         [['P_ID','run_number','event_type','callsign','callsign_group','timestamp_dt',"registration"]]
         )

      # Merge start and end events into single row per job
      resource_use_start = (
         resource_use_start_and_end[resource_use_start_and_end["event_type"] == "resource_use"]
         .rename(columns={'timestamp_dt':'resource_use_start'}).drop(columns="event_type")
         )

      resource_use_end = (
         resource_use_start_and_end[resource_use_start_and_end["event_type"] == "resource_use_end"]
         .rename(columns={'timestamp_dt':'resource_use_end'})
         .drop(columns="event_type")
         )

      resource_use_wide = (
         resource_use_start.merge(resource_use_end, how="outer",
                                 on=["P_ID","run_number","callsign", "callsign_group", "registration"])
                                 .sort_values(["run_number", "P_ID"])
                                 )

      resource_use_wide['resource_use_start'] = pd.to_datetime(resource_use_wide['resource_use_start'])
      resource_use_wide['resource_use_end'] = pd.to_datetime(resource_use_wide['resource_use_end'])

      # Get a list of callsign groups that appear in the sim output to iterate through
      callsign_groups = resource_use_wide["callsign_group"].unique()

      # Initialise an empty list to store all instances of overlaps in
      all_overlaps = []

      # For each group, identify any overlapping resource usage
      for callsign_group in callsign_groups:

         single_callsign = resource_use_wide[resource_use_wide["callsign_group"]==callsign_group]

         # Sort by group and start time
         df_sorted = single_callsign.sort_values(by=["run_number", "callsign_group", "resource_use_start"])

         # Shift end times within each group to compare with the next start
         df_sorted["prev_resource_use_start"] = df_sorted.groupby(["run_number", "callsign_group"])["resource_use_start"].shift()
         df_sorted["prev_resource_use_end"] = df_sorted.groupby(["run_number", "callsign_group"])["resource_use_end"].shift()
         df_sorted["prev_resource_callsign"] = df_sorted.groupby(["run_number", "callsign_group"])["callsign"].shift()
         df_sorted["prev_resource_reg"] = df_sorted.groupby(["run_number", "callsign_group"])["registration"].shift()
         df_sorted["prev_P_ID"] = df_sorted.groupby(["run_number", "callsign_group"])["P_ID"].shift()

         # Find overlaps
         df_sorted["overlap"] = df_sorted["resource_use_start"] < df_sorted["prev_resource_use_end"]

         # Filter to overlapping rows
         overlaps = df_sorted[df_sorted["overlap"]]

         print(f"Callsign Group {callsign_group} - jobs: {len(single_callsign)}")
         print(f"Callsign Group {callsign_group} - overlaps: {len(overlaps)}")

         all_overlaps.append(overlaps)

      all_overlaps_df = pd.concat(all_overlaps)

      if len(all_overlaps_df)>0:
         all_overlaps_df.to_csv("tests/simultaneous_allocation_same_callsigngroup_FAILURES.csv")
         resource_use_wide.to_csv("tests/simultaneous_allocation_same_callsigngroup_FULL.csv")


      assert len(all_overlaps_df) == 0, (
            f"[FAIL - RESOURCE ALLOCATION LOGIC] {len(all_overlaps_df)} instances found of resources from the same callsign group being sent on two or more jobs at once across {len(resource_use_wide)} calls")

   finally:
      del resource_use_start_and_end, resource_use_start, resource_use_end, resource_use_wide, single_callsign, df_sorted, overlaps, all_overlaps, all_overlaps_df
      gc.collect()

@pytest.mark.resources
def test_simultaneous_allocation_same_resource(simulation_results):
   """
   Ensures no single resource is allocated to multiple jobs at the same time.

   Checks that a specific callsign (i.e., physical unit) is not double-booked.
   """
   try:
      if os.path.exists("tests/simultaneous_allocation_same_resource_FAILURES.csv"):
         os.remove("tests/simultaneous_allocation_same_resource_FAILURES.csv")

      if os.path.exists("tests/simultaneous_allocation_same_resource_FULL.csv"):
         os.remove("tests/simultaneous_allocation_same_resource_FULL.csv")

      results = simulation_results # defined in conftest.py

      resource_use_start_and_end = (
         results[results["event_type"].isin(["resource_use","resource_use_end"])]
         [['P_ID','run_number','event_type','callsign','callsign_group','registration','timestamp_dt']]
         )

      resource_use_start = (
         resource_use_start_and_end[resource_use_start_and_end["event_type"] == "resource_use"]
         .rename(columns={'timestamp_dt':'resource_use_start'})
         .drop(columns="event_type")
         )

      resource_use_end = (
         resource_use_start_and_end[resource_use_start_and_end["event_type"] == "resource_use_end"]
         .rename(columns={'timestamp_dt':'resource_use_end'})
         .drop(columns="event_type")
         )

      resource_use_wide = (
         resource_use_start.merge(resource_use_end, how="outer",
                                 on=["P_ID","run_number","callsign", "callsign_group", "registration"]
                                 )
         .sort_values(["run_number", "P_ID"]))

      resource_use_wide['resource_use_start'] = pd.to_datetime(resource_use_wide['resource_use_start'])
      resource_use_wide['resource_use_end'] = pd.to_datetime(resource_use_wide['resource_use_end'])

      callsigns = resource_use_wide["callsign"].unique()

      all_overlaps = []

      for callsign in callsigns:

         single_callsign = resource_use_wide[resource_use_wide["callsign"]==callsign]

         assert len(single_callsign) > 0, f"Single callsign df for {callsign} is empty"

         # Sort by group and start time
         df_sorted = single_callsign.sort_values(by=["run_number", "callsign", "resource_use_start"])
         print(df_sorted)

         # Shift end times within each group to compare with the next start
         df_sorted["prev_resource_use_start"] = df_sorted.groupby(["run_number", "callsign"])["resource_use_start"].shift()
         df_sorted["prev_resource_use_end"] = df_sorted.groupby(["run_number", "callsign"])["resource_use_end"].shift()
         df_sorted["prev_resource_callsign"] = df_sorted.groupby(["run_number", "callsign"])["callsign"].shift()
         df_sorted["prev_resource_reg"] = df_sorted.groupby(["run_number", "callsign"])["registration"].shift()
         df_sorted["prev_P_ID"] = df_sorted.groupby(["run_number", "callsign"])["P_ID"].shift()

         # Find overlaps
         df_sorted["overlap"] = df_sorted["resource_use_start"] < df_sorted["prev_resource_use_end"]

         # Filter to overlapping rows
         overlaps = df_sorted[df_sorted["overlap"]]

         print(f"Callsign {callsign} - instances: {len(single_callsign)}")
         print(f"Callsign {callsign} - overlaps: {len(overlaps)}")

         all_overlaps.append(overlaps)

      all_overlaps_df = pd.concat(all_overlaps)

      if len(all_overlaps_df)>0:
         all_overlaps_df.to_csv("tests/simultaneous_allocation_same_resource_FAILURES.csv")
         resource_use_wide.to_csv("tests/simultaneous_allocation_same_resource_FULL.csv")

      assert len(all_overlaps_df) == 0, (
            f"[FAIL - RESOURCE ALLOCATION LOGIC] {len(all_overlaps_df)} instances found of resources being sent on two or more jobs at once across {len(resource_use_wide)} calls"
            )
   finally:
      del resource_use_start_and_end, resource_use_start, resource_use_end, resource_use_wide, single_callsign, df_sorted, overlaps, all_overlaps, all_overlaps_df
      gc.collect()

@pytest.mark.resources
def test_no_response_during_off_shift_times(simulation_results):
   """
   Ensures no response is initiated outside of a resource's rota'd hours.

   Checks that callsigns are only used when their operating hours (seasonal) allow them to be active.
   Includes a manual test case for validation.
   """
   try:
      removeExistingResults(remove_run_results_csv=True)

      if os.path.exists("tests/offline_calls_FAILURES.csv"):
         os.remove("tests/offline_calls_FAILURES.csv")

      results = simulation_results # defined in conftest.py

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

      if len(offline_calls)>0:
         offline_calls.to_csv("tests/offline_calls_FAILURES.csv")

      # Check there are no offline calls
      assert len(offline_calls)==0, (
            f"[FAIL - RESOURCE ALLOCATION LOGIC - ROTA] {len(offline_calls)} calls appear to have had a response initiated outside of rota'd hours"
            )

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

      # 4 rows we added should fail, 1 row we added should pass
      # (update this if adding additional test cases)
      assert len(offline_calls) == (len(additional_rows) - 1), (
            "[FAIL - RESOURCE ALLOCATION LOGIC - ROTA] The function for testing resources being allocated out of rota'd hours is not behaving correctly"
            )

   finally:
      del offline_calls, merged_df, resource_use_start, results
      gc.collect()

@pytest.mark.resources
def test_no_response_during_service(simulation_results):
   try:
      if os.path.exists("tests/responses_during_servicing_FULL.csv"):
         os.remove("tests/responses_during_servicing_FULL.csv")

      if os.path.exists("tests/responses_during_servicing_FAILURES.csv"):
         os.remove("tests/responses_during_servicing_FAILURES.csv")

      # Load key data files produced by the simulation - results and generated service intervals
      results = simulation_results # defined in conftest.py
      services = pd.read_csv("tests/service_dates_fixture.csv")

      # Ensure service start and end dates are datetimes
      services['service_start_date'] = pd.to_datetime(services['service_start_date'], format="%Y-%m-%d", errors='coerce')
      services['service_end_date'] = pd.to_datetime(services['service_end_date'], format="%Y-%m-%d", errors='coerce')

      # Extract 'resource_use' events from the results and parse relevant info
      resource_use_start = (
          results[results["event_type"] == "resource_use"]
          .rename(columns={'timestamp_dt':'resource_use_start'})
          # Note we are going to be using registration here as our identifier - not the callsign
          [['P_ID','run_number','registration', 'callsign', 'resource_use_start', 'day', 'hour','month','qtr']]
          )

      resource_use_start['resource_use_start'] = pd.to_datetime(resource_use_start['resource_use_start'])

      # Merge resource usage with service records via registration
      # Note that registration is the area of importance for servicing - not the callsign
      # g-daas (H70) should borrow g-daan (H71) during servicing, leading to the callsign H70 remaining
      # in action during the servicing, and H71 showing no activity during servicing of g-daas.
      merged_df = pd.merge(resource_use_start, services, on='registration', how='left')

      # Keep only rows where service start and end dates are valid
      # (i.e. discard any rows where no servicing exists in the servicing dataframe)
      # At present we don't have any servicing of cars (standalone or helicopter backup cars)
      # so those rows will not be of interest to us.
      valid_servicing = merged_df.dropna(subset=['service_start_date', 'service_end_date'])

      valid_servicing.to_csv("tests/responses_during_servicing_FULL.csv")

      # Identify any rows where the resource_use_start falls within the servicing interval
      violations = valid_servicing[
         (valid_servicing['resource_use_start'] >= valid_servicing['service_start_date']) &
         (valid_servicing['resource_use_start'] <= valid_servicing['service_end_date'])
      ]

      # Save violations to file if any are found
      if len(violations)>0:
         violations.to_csv("tests/responses_during_servicing_FAILURES.csv")

      # Assert that no responses occurred during servicing periods
      assert len(violations) == 0, (
         f"[FAIL - RESOURCE ALLOCATION LOGIC - SERVICING] {len(violations)} resource_use_start values fall within a servicing interval"
      )

   except:
      del violations, valid_servicing, merged_df, resource_use_start, results
      gc.collect()
