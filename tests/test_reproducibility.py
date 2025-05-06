"""Reproducibility testing for the Discrete-Event Simulation (DES) Model.

These check that results are consistent when random seeds are provided but different when seeds
differ or across multiple runs - while still being reproducible with the same master seed.

Many tests inspired by list here:
https://github.com/pythonhealthdatascience/rap_template_python_des/blob/main/tests/test_unittest_model.py

Planned tests are listed below with a [].
Implemented tests are listed below with a [x].

## Multiple Runs

[x] Results dataframe is longer when model conducts more runs
[x] Results differ across multiple runs
[x] Arrivals differ across multiple runs
[x] Running the model sequentially and in parallel produce identical results when seeds set

## Seeds

[x] Model behaves consistently across repeated runs when provided with a seed and no parameters change
[x] Arrivals are identical across simulations when provided with a seed even when other parameters are varied
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


@pytest.mark.reproducibility
def test_results_differ_across_runs_runSim():
   try:
      removeExistingResults(remove_run_results_csv=True)

      RUN = 1
      TOTAL_RUNS = 1
      SIM_DURATION = 60 * 24 * 7 * 2 # 2 weeks
      WARM_UP_TIME = 0
      SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
      AMB_DATA = False

      # Should only differ in random seed
      results_df_1 = runSim(run=RUN,
                           total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=42).reset_index()

      results_df_2 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=13).reset_index()

      assert not results_df_1.equals(results_df_2), "[FAIL - REPRODUCIBILITY - RESULTS] Results were the same across runs provided with different seeds"

   finally:
      del results_df_1, results_df_2
      gc.collect()

@pytest.mark.reproducibility
def test_results_differ_across_runs_parallelProcessJobLib():
   try:
      removeExistingResults(remove_run_results_csv=True)

      SIM_DURATION = 60 * 24 * 7 * 2 # 2 weeks
      WARM_UP_TIME = 0
      SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
      AMB_DATA = False

      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=2,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=42
         )

      collateRunResults()

      results_df = pd.read_csv("data/run_results.csv")

      results_df_run_1 = results_df[results_df["run_number"]==1]
      results_df_run_2 = results_df[results_df["run_number"]==2]
      assert len(results_df_run_1) > 0, "Results df for run 1 is empty"
      assert len(results_df_run_2) > 0, "Results df for run 2 is empty"

      assert not results_df_run_1.equals(results_df_run_2), "[FAIL - REPRODUCIBILITY - RESULTS] Results for run 1 and run 2 in parallel execution are identical"

   finally:
      del results_df, results_df_run_1, results_df_run_2
      gc.collect()

@pytest.mark.reproducibility
def test_different_seed_gives_different_arrival_pattern_runSim():
   """
   When passing different seeds to the runSim method, check arrivals differ
   """
   # try:
   removeExistingResults(remove_run_results_csv=True)

   RUN = 1
   TOTAL_RUNS = 1
   SIM_DURATION = 60 * 24 * 7 * 2 # 2 weeks
   WARM_UP_TIME = 0
   SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
   AMB_DATA = False

   # Should only differ in random seed
   results_df_1 = runSim(run=RUN,
                        total_runs=TOTAL_RUNS,
                        sim_duration=SIM_DURATION,
                        warm_up_time=WARM_UP_TIME,
                        sim_start_date=SIM_START_DATE,
                        amb_data=AMB_DATA,
                        random_seed=42).reset_index()

   arrivals_df_1 = results_df_1[results_df_1["time_type"]=="arrival"][["P_ID","timestamp_dt"]]

   results_df_2 = runSim(run=RUN,
                           total_runs=TOTAL_RUNS,
                        sim_duration=SIM_DURATION,
                        warm_up_time=WARM_UP_TIME,
                        sim_start_date=SIM_START_DATE,
                        amb_data=AMB_DATA,
                        random_seed=13).reset_index()


   arrivals_df_2 = results_df_2[results_df_2["time_type"]=="arrival"][["P_ID","timestamp_dt"]]

   assert len(arrivals_df_1) > 0, "Arrivals df 1 is empty"
   assert len(arrivals_df_2) > 0, "Arrivals df 2 is empty"

   assert not arrivals_df_1.equals(arrivals_df_2), "[FAIL - REPRODUCIBILITY - ARRIVALS] Arrivals are the same when different random seed provided (runSim function)"

   # finally:
   #    del results_df_1, results_df_2, arrivals_df_1, arrivals_df_2
   #    gc.collect()

@pytest.mark.reproducibility
def test_different_seed_gives_different_arrival_pattern_parallelProcessJoblib():
   """
   When passing different seeds to the parallelProcessJoblib method, check arrivals differ
   """
   SIM_DURATION = 60 * 24 * 7 * 2
   WARM_UP_TIME = 0
   SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
   AMB_DATA = False

   try:
      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=1,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=42
         )

      collateRunResults()

      results_df_1 = pd.read_csv("data/run_results.csv")

      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=1,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=13
         )

      collateRunResults()

      results_df_2 = pd.read_csv("data/run_results.csv")

      arrivals_df_1 = results_df_1[results_df_1["time_type"]=="arrival"][["P_ID","timestamp_dt"]]
      arrivals_df_2 = results_df_2[results_df_2["time_type"]=="arrival"][["P_ID","timestamp_dt"]]
      assert len(arrivals_df_1) > 0, "Arrivals df 1 is empty"
      assert len(arrivals_df_2) > 0, "Arrivals df 2 is empty"

      assert not arrivals_df_1.equals(arrivals_df_2), "[FAIL - REPRODUCIBILITY - ARRIVALS] Arrivals are the same when different random seed provided (parallelProcessJoblib function)"

   finally:
      del results_df_1, results_df_2, arrivals_df_1, arrivals_df_2
      gc.collect()


@pytest.mark.reproducibility
def test_same_seed_gives_consistent_calls_per_day_runSim():
   try:
      removeExistingResults(remove_run_results_csv=True)

      RUN = 1
      TOTAL_RUNS = 1
      SIM_DURATION = 60 * 24 * 7 * 2
      WARM_UP_TIME = 0
      SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
      AMB_DATA = False
      RANDOM_SEED=42

      # Should only differ in random seed
      results_df_1 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=RANDOM_SEED).reset_index()


      results_df_2 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=RANDOM_SEED).reset_index()

      arrivals_df_1 = results_df_1[results_df_1["time_type"]=="arrival"][["P_ID","timestamp_dt"]]
      arrivals_df_2 = results_df_2[results_df_2["time_type"]=="arrival"][["P_ID","timestamp_dt"]]

      assert len(arrivals_df_1) > 0, "Arrivals df 1 is empty"
      assert len(arrivals_df_2) > 0, "Arrivals df 2 is empty"

      arrivals_df_1['day'] = pd.to_datetime(arrivals_df_1["timestamp_dt"]).dt.date
      arrivals_df_2['day'] = pd.to_datetime(arrivals_df_2["timestamp_dt"]).dt.date

      arrivals_df_1 = arrivals_df_1['day'].value_counts(sort=False)
      arrivals_df_2 = arrivals_df_2['day'].value_counts(sort=False)

      arrivals_df_1.to_csv("tests/test_outputs/TEST_OUTPUT_test_same_seed_gives_consistent_calls_per_day_runSim - arrivals_df_1.csv")
      arrivals_df_2.to_csv("tests/test_outputs/TEST_OUTPUT_test_same_seed_gives_consistent_calls_per_day_runSim - arrivals_df_2.csv")


      assert arrivals_df_1.equals(arrivals_df_2), "[FAIL - REPRODUCIBILITY - ARRIVALS] Number of daily arrivals are not the same when same random seed provided and parameters held constant (runSim function)"

   finally:
      del results_df_1, results_df_2, arrivals_df_1, arrivals_df_2
      gc.collect()


@pytest.mark.reproducibility
def test_same_seed_gives_consistent_calls_per_hour_runSim():
   try:
      removeExistingResults(remove_run_results_csv=True)

      RUN = 1
      TOTAL_RUNS = 1
      SIM_DURATION = 60 * 24 * 7 * 2
      WARM_UP_TIME = 0
      SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
      AMB_DATA = False
      RANDOM_SEED=42

      # Should only differ in random seed
      results_df_1 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=RANDOM_SEED).reset_index()


      results_df_2 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=RANDOM_SEED).reset_index()

      arrivals_df_1 = results_df_1[results_df_1["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      arrivals_df_2 = results_df_2[results_df_2["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)

      assert len(arrivals_df_1) > 0, "Arrivals df 1 is empty"
      assert len(arrivals_df_2) > 0, "Arrivals df 2 is empty"

      arrivals_df_1['day'] = pd.to_datetime(arrivals_df_1["timestamp_dt"]).dt.strftime("%Y-%m-%d %H")
      arrivals_df_2['day'] = pd.to_datetime(arrivals_df_2["timestamp_dt"]).dt.strftime("%Y-%m-%d %H")

      arrivals_df_1 = arrivals_df_1['day'].value_counts(sort=False)
      arrivals_df_2 = arrivals_df_2['day'].value_counts(sort=False)

      arrivals_df_1.to_csv("tests/test_outputs/TEST_OUTPUT_test_same_seed_gives_consistent_calls_per_hour_runSim - arrivals_df_1.csv")
      arrivals_df_2.to_csv("tests/test_outputs/TEST_OUTPUT_test_same_seed_gives_consistent_calls_per_hour_runSim - arrivals_df_2.csv")


      assert arrivals_df_1.equals(arrivals_df_2), "[FAIL - REPRODUCIBILITY - ARRIVALS] Number of daily arrivals are not the same when same random seed provided and parameters held constant (runSim function)"

   finally:
      del results_df_1, results_df_2, arrivals_df_1, arrivals_df_2
      gc.collect()



@pytest.mark.reproducibility
def test_same_seed_gives_consistent_arrival_pattern_runSim():
   try:
      removeExistingResults(remove_run_results_csv=True)

      RUN = 1
      TOTAL_RUNS = 1
      SIM_DURATION = 60 * 24 * 7 * 2
      WARM_UP_TIME = 0
      SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
      AMB_DATA = False
      RANDOM_SEED=42

      # Should only differ in random seed
      results_df_1 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=RANDOM_SEED).reset_index()


      results_df_2 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=RANDOM_SEED).reset_index()

      arrivals_df_1 = results_df_1[results_df_1["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      arrivals_df_2 = results_df_2[results_df_2["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      assert len(arrivals_df_1) > 0, "Arrivals df 1 is empty"
      assert len(arrivals_df_2) > 0, "Arrivals df 2 is empty"

      assert arrivals_df_1.equals(arrivals_df_2), "[FAIL - REPRODUCIBILITY - ARRIVALS] Arrivals are not the same when same random seed provided and parameters held constant (runSim function)"

   finally:
      del results_df_1, results_df_2, arrivals_df_1, arrivals_df_2
      gc.collect()

@pytest.mark.reproducibility
def test_different_seed_gives_different_arrival_pattern_runSim():
   try:
      removeExistingResults(remove_run_results_csv=True)

      RUN = 1
      TOTAL_RUNS = 1
      SIM_DURATION = 60 * 24 * 7 * 2
      WARM_UP_TIME = 0
      SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
      AMB_DATA = False

      # Should only differ in random seed
      results_df_1 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=42).reset_index()


      results_df_2 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=13).reset_index()

      arrivals_df_1 = results_df_1[results_df_1["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      arrivals_df_2 = results_df_2[results_df_2["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      assert len(arrivals_df_1) > 0, "Arrivals df 1 is empty"
      assert len(arrivals_df_2) > 0, "Arrivals df 2 is empty"

      assert not arrivals_df_1.equals(arrivals_df_2), "[FAIL - REPRODUCIBILITY - ARRIVALS] Arrivals are the same when different random seed provided and parameters held constant (runSim function)"

   finally:
      del results_df_1, results_df_2, arrivals_df_1, arrivals_df_2
      gc.collect()

@pytest.mark.reproducibility
def test_same_seed_gives_consistent_arrival_pattern_parallelProcessJoblib():
   SIM_DURATION = 60 * 24 * 7 * 2
   WARM_UP_TIME = 0
   SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
   AMB_DATA = False
   RANDOM_SEED=42

   try:
      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=1,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=RANDOM_SEED
         )

      collateRunResults()

      results_df_1 = pd.read_csv("data/run_results.csv")

      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=1,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=RANDOM_SEED
         )

      collateRunResults()

      results_df_2 = pd.read_csv("data/run_results.csv")

      arrivals_df_1 = results_df_1[results_df_1["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      arrivals_df_2 = results_df_2[results_df_2["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      assert len(arrivals_df_1) > 0, "Arrivals df 1 is empty"
      assert len(arrivals_df_2) > 0, "Arrivals df 2 is empty"

      arrivals_df_1.to_csv("tests/test_outputs/TEST_OUTPUT_test_same_seed_gives_consistent_arrival_pattern_parallelProcessJoblib - arrivals_df_1.csv")
      arrivals_df_2.to_csv("tests/test_outputs/TEST_OUTPUT_test_same_seed_gives_consistent_arrival_pattern_parallelProcessJoblib - arrivals_df_2.csv")

      assert arrivals_df_1.equals(arrivals_df_2), "[FAIL - REPRODUCIBILITY - ARRIVALS] Arrivals are not the same when same random seed provided and parameters held constant (parallelProcessJoblib function)"

   finally:
      del results_df_1, results_df_2, arrivals_df_1, arrivals_df_2
      gc.collect()

@pytest.mark.reproducibility
def test_different_seed_gives_different_arrival_pattern_parallelProcessJoblib():
   SIM_DURATION = 60 * 24 * 7 * 2
   WARM_UP_TIME = 0
   SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
   AMB_DATA = False

   try:
      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=1,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=42
         )

      collateRunResults()

      results_df_1 = pd.read_csv("data/run_results.csv")

      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=1,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=13
         )

      collateRunResults()

      results_df_2 = pd.read_csv("data/run_results.csv")

      arrivals_df_1 = results_df_1[results_df_1["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      arrivals_df_2 = results_df_2[results_df_2["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      assert len(arrivals_df_1) > 0, "Arrivals df 1 is empty"
      assert len(arrivals_df_2) > 0, "Arrivals df 2 is empty"

      assert not arrivals_df_1.equals(arrivals_df_2), "[FAIL - REPRODUCIBILITY - ARRIVALS] Arrivals are the same when different random seed provided and parameters held constant (parallelProcessJoblib function)"

   finally:
      del results_df_1, results_df_2, arrivals_df_1, arrivals_df_2
      gc.collect()



@pytest.mark.reproducibility
def test_different_arrival_pattern_across_runs_parallelProcessJoblib():
   SIM_DURATION = 60 * 24 * 7 * 2
   WARM_UP_TIME = 0
   SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
   AMB_DATA = False

   try:
      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=2,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=42
         )

      collateRunResults()

      results_df = pd.read_csv("data/run_results.csv")
      results_df_1 = results_df[results_df["run_number"] == 1]
      results_df_2 = results_df[results_df["run_number"] == 2]

      arrivals_df_1 = results_df_1[results_df_1["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      arrivals_df_2 = results_df_2[results_df_2["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      assert len(arrivals_df_1) > 0, "Arrivals df 1 is empty"
      assert len(arrivals_df_2) > 0, "Arrivals df 2 is empty"

      arrivals_df_1.to_csv("tests/test_outputs/TEST_OUTPUT_test_arrivals_behaviour_parallelProcessJobLib - arrivals_df_1.csv")
      arrivals_df_2.to_csv("tests/test_outputs/TEST_OUTPUT_test_arrivals_behaviour_parallelProcessJobLib - arrivals_df_2.csv")

      assert not arrivals_df_1.equals(arrivals_df_2), "[FAIL - REPRODUCIBILITY - ARRIVALS] Arrivals are the same across different runs across different runs with the parallelProcessJoblib function"

   finally:
      del results_df_1, results_df_2, arrivals_df_1, arrivals_df_2
      gc.collect()


@pytest.mark.reproducibility
def test_same_seed_gives_consistent_arrival_pattern_VARYING_PARAMETERS_runSim():
   try:
      removeExistingResults(remove_run_results_csv=True)

      RUN = 1
      TOTAL_RUNS = 1
      SIM_DURATION = 60 * 24 * 7 * 2
      WARM_UP_TIME = 0
      SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
      AMB_DATA = False
      RANDOM_SEED=42

      # Should only differ in random seed
      results_df_1 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=RANDOM_SEED).reset_index()

      rota = pd.read_csv("tests/rotas_test/HEMS_ROTA_test_one_helicopter_simple.csv")

      rota.to_csv("actual_data/HEMS_ROTA.csv", index=False)

      results_df_2 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=RANDOM_SEED).reset_index()

      arrivals_df_1 = results_df_1[results_df_1["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      arrivals_df_2 = results_df_2[results_df_2["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      assert len(arrivals_df_1) > 0, "Arrivals df 1 is empty"
      assert len(arrivals_df_2) > 0, "Arrivals df 2 is empty"

      assert arrivals_df_1.equals(arrivals_df_2), "[FAIL - REPRODUCIBILITY - ARRIVALS] Arrivals are not the same when same random seed provided and other aspects varied (runSim function)"

   finally:
      default_rota = pd.read_csv("actual_data/HEMS_ROTA_DEFAULT.csv")
      default_rota.to_csv("actual_data/HEMS_ROTA.csv", index=False)
      del results_df_1, results_df_2, arrivals_df_1, arrivals_df_2
      gc.collect()

@pytest.mark.reproducibility
def test_same_seed_gives_consistent_arrival_pattern_VARYING_PARAMETERS_parallelProcessJoblib():
   SIM_DURATION = 60 * 24 * 7 * 2
   WARM_UP_TIME = 0
   SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
   AMB_DATA = False
   RANDOM_SEED=42

   try:
      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=1,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=RANDOM_SEED
         )

      collateRunResults()

      results_df_1 = pd.read_csv("data/run_results.csv")

      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=1,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=RANDOM_SEED
         )

      collateRunResults()

      results_df_2 = pd.read_csv("data/run_results.csv")

      arrivals_df_1 = results_df_1[results_df_1["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      arrivals_df_2 = results_df_2[results_df_2["time_type"]=="arrival"][["P_ID","timestamp_dt"]].reset_index(drop=True)
      assert len(arrivals_df_1) > 0, "Arrivals df 1 is empty"
      assert len(arrivals_df_2) > 0, "Arrivals df 2 is empty"

      assert arrivals_df_1.equals(arrivals_df_2), "[FAIL - REPRODUCIBILITY - ARRIVALS] Arrivals are not the same when same random seed provided and other aspects varied (parallelProcessJoblib function)"

   finally:
      default_rota = pd.read_csv("actual_data/HEMS_ROTA_DEFAULT.csv")
      default_rota.to_csv("actual_data/HEMS_ROTA.csv", index=False)
      del results_df_1, results_df_2, arrivals_df_1, arrivals_df_2
      gc.collect()



@pytest.mark.reproducibility
def test_same_seed_gives_consistent_results_pattern_runSim():
   try:
      removeExistingResults(remove_run_results_csv=True)

      RUN = 1
      TOTAL_RUNS = 1
      SIM_DURATION = 60 * 24 * 7 * 2
      WARM_UP_TIME = 0
      SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
      AMB_DATA = False
      RANDOM_SEED=42

      # Should only differ in random seed
      results_df_1 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=RANDOM_SEED).reset_index()


      results_df_2 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=RANDOM_SEED).reset_index()

      assert len(results_df_1) > 0, "Results df 1 is empty"
      assert len(results_df_2) > 0, "Results df 2 is empty"


      results_df_1.to_csv("tests/test_outputs/TEST_OUTPUT_test_same_seed_gives_consistent_results_pattern_runSim - results_df_1.csv")
      results_df_2.to_csv("tests/test_outputs/TEST_OUTPUT_test_same_seed_gives_consistent_results_pattern_runSim - results_df_2.csv")

      assert results_df_1.equals(results_df_2), "[FAIL - REPRODUCIBILITY - RESULTS] Results are not the same when same random seed provided (runSim function)"

   finally:
      del results_df_1, results_df_2
      gc.collect()

@pytest.mark.reproducibility
def test_same_seed_gives_consistent_results_parallelProcessJoblib():
   SIM_DURATION = 60 * 24 * 7 * 2
   WARM_UP_TIME = 0
   SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
   AMB_DATA = False
   RANDOM_SEED=42

   try:
      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=1,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=RANDOM_SEED
         )

      collateRunResults()

      results_df_1 = pd.read_csv("data/run_results.csv")

      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=1,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=RANDOM_SEED
         )

      collateRunResults()

      results_df_2 = pd.read_csv("data/run_results.csv")

      assert len(results_df_1) > 0, "Results df 1 is empty"
      assert len(results_df_2) > 0, "Results df 2 is empty"

      assert results_df_1.equals(results_df_2), "[FAIL - REPRODUCIBILITY - RESULTS] Results are not the same when same random seed provided (parallelProcessJoblib function)"

   finally:
      del results_df_1, results_df_2
      gc.collect()

@pytest.mark.reproducibility
def test_different_seed_gives_different_results_pattern_runSim():
   try:
      removeExistingResults(remove_run_results_csv=True)

      RUN = 1
      TOTAL_RUNS = 1
      SIM_DURATION = 60 * 24 * 7 * 2
      WARM_UP_TIME = 0
      SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
      AMB_DATA = False

      # Should only differ in random seed
      results_df_1 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=42).reset_index()


      results_df_2 = runSim(run=RUN,
                              total_runs=TOTAL_RUNS,
                           sim_duration=SIM_DURATION,
                           warm_up_time=WARM_UP_TIME,
                           sim_start_date=SIM_START_DATE,
                           amb_data=AMB_DATA,
                           random_seed=13).reset_index()

      assert len(results_df_1) > 0, "Results df 1 is empty"
      assert len(results_df_2) > 0, "Results df 2 is empty"

      assert not results_df_1.equals(results_df_2), "[FAIL - REPRODUCIBILITY - RESULTS] Results are the same when different random seed provided (runSim function)"

   finally:
      del results_df_1, results_df_2
      gc.collect()

@pytest.mark.reproducibility
def test_different_seed_gives_different_results_parallelProcessJoblib():
   SIM_DURATION = 60 * 24 * 7 * 2
   WARM_UP_TIME = 0
   SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
   AMB_DATA = False

   try:
      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=1,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=42
         )

      collateRunResults()

      results_df_1 = pd.read_csv("data/run_results.csv")

      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=1,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=13
         )

      collateRunResults()

      results_df_2 = pd.read_csv("data/run_results.csv")

      assert len(results_df_1) > 0, "Results df 1 is empty"
      assert len(results_df_2) > 0, "Results df 2 is empty"

      assert not results_df_1.equals(results_df_2), "[FAIL - REPRODUCIBILITY - RESULTS] Results are the same when different random seed provided (parallelProcessJoblib function)"

   finally:
      del results_df_1, results_df_2
      gc.collect()

@pytest.mark.reproducibility
def test_different_result_pattern_across_runs_parallelProcessJoblib():
   SIM_DURATION = 60 * 24 * 7 * 2
   WARM_UP_TIME = 0
   SIM_START_DATE = datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S")
   AMB_DATA = False

   try:
      removeExistingResults(remove_run_results_csv=True)

      # Run the simulation with only a warm-up period and no actual simulation time
      parallelProcessJoblib(
         total_runs=2,
         sim_duration=SIM_DURATION,
         warm_up_time=WARM_UP_TIME,
         sim_start_date=SIM_START_DATE,
         amb_data=AMB_DATA,
         master_seed=42
         )

      collateRunResults()

      results_df = pd.read_csv("data/run_results.csv")
      results_df_1 = results_df[results_df["run_number"] == 1].drop(columns="run_number").reset_index(drop=True)
      results_df_2 = results_df[results_df["run_number"] == 2].drop(columns="run_number").reset_index(drop=True)

      assert len(results_df_1) > 0, "Results df 1 is empty"
      assert len(results_df_1) > 0, "Results df 2 is empty"

      results_df_1.to_csv("tests/test_outputs/TEST_OUTPUT_test_result_behaviour_parallelProcessJobLib - results_df_1.csv")
      results_df_2.to_csv("tests/test_outputs/TEST_OUTPUT_test_result_behaviour_parallelProcessJobLib - results_df_2.csv")

      assert not results_df_1.equals(results_df_2), "[FAIL - REPRODUCIBILITY - RESULTS] Results are the same across different runs across different runs with the parallelProcessJoblib function"

   finally:
      del results_df_1, results_df_2
      gc.collect()
