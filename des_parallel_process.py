from datetime import datetime, timedelta
import logging
import time
import glob
import os, sys
import pandas as pd
from utils import Utils
from des_hems import DES_HEMS
import multiprocessing as mp
from joblib import Parallel, delayed

def write_run_params(model) -> None:
        """
        Writes the parameters used for the model to a csv file

        SR NOTE: this is likely to be expanded further in the future to

        SR NOTE: It would also be good to add some sort of identifier to both the run results csv
        and this csv so you can confirm that they came from the same model execution (to avoid
        issues with calculations being incorrect if e.g. it was not possible to write one of the
        outputs due to an error, write protection, etc.)
        """


        # Ensure sim_start_date is a datetime object
        sim_start_date = model.sim_start_date
        if isinstance(sim_start_date, str):
            sim_start_date = datetime.fromisoformat(sim_start_date)  # Convert string to datetime

        sim_end_date = sim_start_date + timedelta(minutes=model.sim_duration)
        warm_up_end_date = sim_start_date + timedelta(minutes=model.warm_up_duration)

        params_df = pd.DataFrame.from_dict({
            'sim_duration': [model.sim_duration],
            'warm_up_duration': [model.warm_up_duration],
            'sim_start_date': [sim_start_date],
            'sim_end_date': [sim_end_date],
            'warm_up_end_date': [warm_up_end_date],
            'amb_data': [model.amb_data],
            'model_exec_time': [datetime.now()],
             # Assuming summer hours are quarters 2 and 3 i.e. April-September
             # This is defined in class_hems and will need updating here too
            'summer_start_date': [f'{sim_start_date.year}-04-01'],
            'winter_start_date':  [f'{sim_start_date.year}-10-01'],
            'activity_duration_multiplier': [model.activity_duration_multiplier]
        }, orient='index', columns=['value'])

        params_df.index.name = "parameter"

        params_df.to_csv(f"{Utils.RESULTS_FOLDER}/run_params_used.csv", header="column_names")

        # TODO: This will need adjusting for this being changeable in the frontend
        # model.utils.HEMS_ROTA.to_csv(f"{Utils.RESULTS_FOLDER}/hems_rota_used.csv", index=False)
        # pd.read_csv("actual_data/HEMS_ROTA.csv").to_csv(f"{Utils.RESULTS_FOLDER}/hems_rota_used.csv", index=False)

try:
     __file__
except NameError:
    __file__ = sys.argv[0]

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

def runSim(run: int,
           total_runs: int,
           sim_duration: int,
           warm_up_time: int,
           sim_start_date: datetime,
           amb_data: bool,
           save_params_csv: bool = True,
           demand_increase_percent: float = 1.0,
           activity_duration_multiplier: float = 1.0):
    #print(f"Inside runSim and {sim_start_date} and {what_if_sim_run}")

    print(f'{Utils.current_time()}: Demand increase set to {demand_increase_percent*100}%')
    logging.debug(f'{Utils.current_time()}: Demand increase set to {demand_increase_percent*100}%')

    start = time.process_time()

    print (f"{Utils.current_time()}: Run {run+1} of {total_runs}")
    logging.debug(f"{Utils.current_time()}: Run {run+1} of {total_runs}")

    #print(f"Sim start date is {sim_start_date}")
    daa_model = DES_HEMS(run_number=run,
                        sim_duration=sim_duration,
                        warm_up_duration=warm_up_time,
                        sim_start_date=sim_start_date,
                        amb_data=amb_data,
                        demand_increase_percent=demand_increase_percent,
                        activity_duration_multiplier=activity_duration_multiplier
                        )
    daa_model.run()

    print(f'{Utils.current_time()}: Run {run+1} took {round((time.process_time() - start)/60, 1)} minutes to run')
    logging.debug(f'{Utils.current_time()}: Run {run+1} took {round((time.process_time() - start)/60, 1)} minutes to run')

    # SR NOTE: This could cause issues if we decide to use 1 as the starting number of runs
    if (run==0) and (save_params_csv):
        write_run_params(daa_model)

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

def parallelProcessJoblib(total_runs: int,
                          sim_duration: int,
                          warm_up_time: int,
                          sim_start_date: datetime,
                          amb_data: bool,
                          save_params_csv: bool = True,
                          demand_increase_percent: float = 1.0,
                          activity_duration_multiplier: float = 1.0):

    return Parallel(n_jobs=-1)(delayed(runSim)(run, total_runs, sim_duration, warm_up_time, sim_start_date, amb_data, save_params_csv, demand_increase_percent, activity_duration_multiplier) for run in range(total_runs))

if __name__ == "__main__":
    removeExistingResults()
    #parallelProcessJoblib(1, (0.25*365*24*60), (0*60), datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"), False, False, 1.0, 1.0)
    parallelProcessJoblib(5, (2*365*24*60), (24*60), datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"), False, False, 1.0, 1.0)

# Testing ----------
# python des_parallel_process.py
