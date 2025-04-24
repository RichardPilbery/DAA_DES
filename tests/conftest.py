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

SERVICE_DATES_CSV_PATH = "data/service_dates.csv"
SERVICE_DATES_CSV_PATH_OUT = "tests/service_dates_fixture.csv"

@pytest.fixture(scope="session")
def simulation_results():
    """Run the simulation only if needed and return the event dataframe."""
    if not os.path.exists(RESULTS_CSV_PATH) or not os.path.exists(RESULTS_CSV_PATH_OUT):
        # Ensure all rotas are using default values
        rota = pd.read_csv("tests/HISTORIC_HEMS_ROTA.csv")
        rota.to_csv("actual_data/HEMS_ROTA.csv", index=False)

        callsign_reg_lookup = pd.read_csv("tests/HISTORIC_callsign_registration_lookup.csv")
        callsign_reg_lookup.to_csv("actual_data/callsign_registration_lookup.csv", index=False)

        service_history = pd.read_csv("tests/HISTORIC_service_history.csv")
        service_history.to_csv("actual_data/service_history.csv", index=False)

        service_sched = pd.read_csv("tests/HISTORIC_service_schedules_by_model.csv")
        service_sched.to_csv("actual_data/service_schedules_by_model.csv", index=False)

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

        df = pd.read_csv(SERVICE_DATES_CSV_PATH)
        df.to_csv(SERVICE_DATES_CSV_PATH_OUT)
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

    # Remove generated csvs to avoid accidental usage
    for filepath in [RESULTS_CSV_PATH, RESULTS_CSV_PATH_OUT, SERVICE_DATES_CSV_PATH, SERVICE_DATES_CSV_PATH_OUT]:

        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Removed cached simulation results: {filepath}")
            except Exception as e:
                print(f"Warning: Failed to remove {filepath} — {e}")

    # Revert all rotas to defaults
    rota = pd.read_csv("actual_data/HEMS_ROTA_DEFAULT.csv")
    rota.to_csv("actual_data/HEMS_ROTA.csv", index=False)

    callsign_reg_lookup = pd.read_csv("actual_data/callsign_registration_lookup_DEFAULT.csv")
    callsign_reg_lookup.to_csv("actual_data/callsign_registration_lookup.csv", index=False)

    service_history = pd.read_csv("actual_data/callsign_registration_lookup_DEFAULT.csv")
    service_history.to_csv("actual_data/callsign_registration_lookup.csv", index=False)

    service_sched = pd.read_csv("actual_data/service_schedules_by_model_DEFAULT.csv")
    service_sched.to_csv("actual_data/service_schedules_by_model.csv", index=False)
