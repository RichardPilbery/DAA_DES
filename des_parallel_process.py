import logging
import time
from utils import Utils
from des_hems import DES_HEMS


def runSim(run: int, total_runs: int, sim_duration: int, warm_up_time: int, sim_start_date: str):
    #print(f"Inside runSim and {sim_start_date} and {what_if_sim_run}")

    start = time.process_time()

    print (f"{Utils.current_time()}: Run {run+1} of {total_runs}")
    logging.debug(f"{Utils.current_time()}: Run {run+1} of {total_runs}")

    daa_model = DES_HEMS(run, sim_duration, warm_up_time, sim_start_date)
    daa_model.run()

    print(f'{Utils.current_time()}: Run {run+1} took {round((time.process_time() - start)/60, 1)} minutes to run')
    logging.debug(f'{Utils.current_time()}: Run {run+1} took {round((time.process_time() - start)/60, 1)} minutes to run')


runSim(0, 1, 1440, 0, "2021-01-01 00:00:00")
