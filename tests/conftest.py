import pandas as pd
from datetime import datetime
import gc
import os
import pytest
# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from des_parallel_process import parallelProcessJoblib, collateRunResults, runSim, removeExistingResults

##############################################################################
# Fixture - single run for tests where input parameters aren't being changed #
##############################################################################
RESULTS_CSV_PATH = "data/run_results.csv"
RESULTS_CSV_PATH_OUT = "tests/run_results_fixture.csv"

@pytest.fixture(scope="session")
def simulation_results():
    """Run the simulation only if needed and return the event dataframe."""
    if not os.path.exists(RESULTS_CSV_PATH) or not os.path.exists(RESULTS_CSV_PATH_OUT):
        print("Generating simulation results...")
        removeExistingResults()

        parallelProcessJoblib(
            total_runs=10,
            sim_duration=60 * 24 * 7 * 52 * 4, # 4 years
            warm_up_time=0,
            sim_start_date=datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
            amb_data=False
        )

        collateRunResults()

        df = pd.read_csv(RESULTS_CSV_PATH)
        df.to_csv(RESULTS_CSV_PATH_OUT)
    else:
        print("Using cached simulation results...")

    df = pd.read_csv(RESULTS_CSV_PATH_OUT)
    yield df

    del df
    gc.collect()

@pytest.fixture(scope="session", autouse=True)
def cleanup_simulation_results():
    """Automatically remove results CSV after all tests are done."""
    yield  # Wait until all tests using this session scope are finished

    if os.path.exists(RESULTS_CSV_PATH):
        try:
            os.remove(RESULTS_CSV_PATH)
            print(f"Removed cached simulation results: {RESULTS_CSV_PATH}")
        except Exception as e:
            print(f"Warning: Failed to remove {RESULTS_CSV_PATH} — {e}")

    if os.path.exists(RESULTS_CSV_PATH_OUT):
        try:
            os.remove(RESULTS_CSV_PATH_OUT)
            print(f"Removed cached simulation results: {RESULTS_CSV_PATH_OUT}")
        except Exception as e:
            print(f"Warning: Failed to remove {RESULTS_CSV_PATH_OUT} — {e}")
