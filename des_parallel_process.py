from datetime import datetime
import logging
import time
import glob
import os, sys
import pandas as pd
from utils import Utils
from des_hems import DES_HEMS
import multiprocessing as mp
from joblib import Parallel, delayed

try:
     __file__
except NameError: 
    __file__ = sys.argv[0]

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def runSim(run: int, total_runs: int, sim_duration: int, warm_up_time: int, sim_start_date: datetime, amb_data: bool, demand_increase_percent: float):
    #print(f"Inside runSim and {sim_start_date} and {what_if_sim_run}")

    print(f'{Utils.current_time()}: Demand increase set to {demand_increase_percent*100}%')
    logging.debug(f'{Utils.current_time()}: Demand increase set to {demand_increase_percent*100}%')

    start = time.process_time()

    print (f"{Utils.current_time()}: Run {run+1} of {total_runs}")
    logging.debug(f"{Utils.current_time()}: Run {run+1} of {total_runs}")

    #print(f"Sim start date is {sim_start_date}")
    daa_model = DES_HEMS(run, sim_duration, warm_up_time, sim_start_date, amb_data, demand_increase_percent)
    daa_model.run()

    print(f'{Utils.current_time()}: Run {run+1} took {round((time.process_time() - start)/60, 1)} minutes to run')
    logging.debug(f'{Utils.current_time()}: Run {run+1} took {round((time.process_time() - start)/60, 1)} minutes to run')

    return daa_model.results_df

def collateRunResults() -> None:
        """
            Collates results from a series of runs into a single csv
        """
        matching_files = glob.glob(os.path.join(Utils.RESULTS_FOLDER, "output_run_*.csv"))

        combined_df = pd.concat([pd.read_csv(f) for f in matching_files], ignore_index=True)

        combined_df.to_csv(Utils.RUN_RESULTS_CSV, index=False)

        for file in matching_files:
             os.remove(file)

def removeExistingResults() -> None:
        """
            Removes results from previous simulation runs
        """
        matching_files = glob.glob(os.path.join(Utils.RESULTS_FOLDER, "output_run_*.csv"))

        for file in matching_files:
             os.remove(file)

        all_results_file_path = os.path.join(Utils.RESULTS_FOLDER, "all_results.csv")
        if os.path.isfile(all_results_file_path):
            os.unlink(all_results_file_path)

def parallelProcessJoblib(total_runs: int, sim_duration: int, warm_up_time: int, sim_start_date: datetime, amb_data: bool, demand_increase_percent: float):

    return Parallel(n_jobs=-1)(delayed(runSim)(run, total_runs, sim_duration, warm_up_time, sim_start_date, amb_data, demand_increase_percent) for run in range(total_runs))

if __name__ == "__main__":
    removeExistingResults()
    #parallelProcessJoblib(1, (0.5*365*24*60), (0*60), datetime.strptime("2022-07-24 05:47:00", "%Y-%m-%d %H:%M:%S"), False, 1)
    parallelProcessJoblib(5, (2*365*24*60), (0*60), datetime.strptime("2022-07-24 05:47:00", "%Y-%m-%d %H:%M:%S"), False, 1.2)

# Testing ----------
# python des_parallel_process.py