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


def runSim(run: int, total_runs: int, sim_duration: int, warm_up_time: int, sim_start_date: datetime, amb_data: bool):
    #print(f"Inside runSim and {sim_start_date} and {what_if_sim_run}")

    start = time.process_time()

    print (f"{Utils.current_time()}: Run {run+1} of {total_runs}")
    logging.debug(f"{Utils.current_time()}: Run {run+1} of {total_runs}")

    daa_model = DES_HEMS(run, sim_duration, warm_up_time, sim_start_date, amb_data)
    daa_model.run()

    print(f'{Utils.current_time()}: Run {run+1} took {round((time.process_time() - start)/60, 1)} minutes to run')
    logging.debug(f'{Utils.current_time()}: Run {run+1} took {round((time.process_time() - start)/60, 1)} minutes to run')

    return daa_model.results_df

def parallelProcess(nprocess = mp.cpu_count() - 1):
    logging.debug('Model called')

    number_of_runs = 2
    sim_duration = 2 * 24 * 60
    warm_up_time = 8 * 60
    sim_start_date =  datetime.strptime("2024-08-01 07:00:00", "%Y-%m-%d %H:%M:%S") 
    amb_data = False

    pool = mp.Pool(processes = nprocess)
    pool.starmap(runSim, zip(
        list(range(0, number_of_runs)),
            [number_of_runs] * number_of_runs,
            [sim_duration] * number_of_runs,
            [warm_up_time] * number_of_runs,
            [sim_start_date] * number_of_runs,
            [amb_data] * number_of_runs
        )
    )

    logging.debug('Reached end of script')
    logging.shutdown()

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

def parallelProcessJoblib(total_runs: int, sim_duration: int, warm_up_time: int, sim_start_date: datetime, amb_data: bool):

    return Parallel(n_jobs=-1)(delayed(runSim)(run, total_runs, sim_duration, warm_up_time, sim_start_date, amb_data) for run in range(total_runs))

if __name__ == "__main__":
    removeExistingResults()
    parallelProcessJoblib(4, (1*365*24*60), (0*60), datetime.strptime("2022-07-24 09:19:00", "%Y-%m-%d %H:%M:%S"), False)
    #parallelProcessJoblib(5, (1*365*24*60), (0*60), datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"), False)

# Testing ----------
# python des_parallel_process.py