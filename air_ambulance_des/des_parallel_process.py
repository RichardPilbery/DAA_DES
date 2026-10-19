from datetime import datetime, timedelta
import logging
import time
import glob
import os
import sys
import pandas as pd
import multiprocessing as mp
from joblib import Parallel, delayed
from numpy.random import SeedSequence
from pathlib import Path

from air_ambulance_des.utils import Utils
from air_ambulance_des.des_hems import DES_HEMS


def write_run_params(model) -> None:
    """
    Write the simulation parameters used in a DES model run to a CSV file.

    Extracts key configuration parameters from the given model instance and writes them
    to a CSV file (`run_params_used.csv`) in the designated results folder. This provides
    a record of the conditions under which the simulation was executed.

    SR NOTE: It would also be good to add some sort of identifier to both the run results csv
        and this csv so you can confirm that they came from the same model execution (to avoid
        issues with calculations being incorrect if e.g. it was not possible to write one of the
        outputs due to an error, write protection, etc.)

    Parameters
    ----------
    model : DES_HEMS
        The simulation model instance from which to extract run parameters. Must have attributes
        including `sim_start_date`, `sim_duration`, `warm_up_duration`, `amb_data`, and
        `activity_duration_multiplier`.

    Returns
    -------
    None

    Notes
    -----
    - The function calculates `sim_end_date` and `warm_up_end_date` based on the provided
      `sim_start_date` and durations.
    - Output CSV includes timing and configuration values such as:
        - Simulation duration and warm-up duration
        - Simulation start, end, and warm-up end datetimes
        - Whether ambulance data was used
        - Activity duration multiplier
        - Model execution timestamp
        - Assumed summer and winter period start dates
    - The output CSV is saved to `Utils.RESULTS_FOLDER/run_params_used.csv`.
    - Only supports a single simulation's parameters at a time.
    - Future improvements may include adding a unique identifier for linking this file
      with the corresponding simulation results.

    See Also
    --------
    runSim : Runs an individual simulation and optionally calls this function.
    parallelProcessJoblib : Executes multiple `runSim` runs in parallel.
    """
    # Ensure sim_start_date is a datetime object
    sim_start_date = model.sim_start_date
    if isinstance(sim_start_date, str):
        sim_start_date = datetime.fromisoformat(
            sim_start_date
        )  # Convert string to datetime

    sim_end_date = sim_start_date + timedelta(minutes=model.sim_duration)
    warm_up_end_date = sim_start_date + timedelta(minutes=model.warm_up_duration)

    params_df = pd.DataFrame.from_dict(
        {
            "sim_duration": [model.sim_duration],  # model time unit is minutes
            "sim_duration_hours": [model.sim_duration / 60],
            "sim_duration_days": [model.sim_duration / 60 / 24],
            "warm_up_duration": [model.warm_up_duration],
            "sim_start_date": [sim_start_date],
            "sim_end_date": [sim_end_date],
            "warm_up_end_date": [warm_up_end_date],
            "amb_data": [model.amb_data],
            "model_exec_time": [datetime.now()],
            # Assuming summer hours are quarters 2 and 3 i.e. April-September
            # This is defined in class_hems and will need updating here too
            "summer_start_date": [f"{sim_start_date.year}-04-01"],
            "winter_start_date": [f"{sim_start_date.year}-10-01"],
            "activity_duration_multiplier": [model.activity_duration_multiplier],
        },
        orient="index",
        columns=["value"],
    )

    params_df.index.name = "parameter"

    params_df.to_csv(
        f"{Utils.RESULTS_FOLDER}/run_params_used.csv", header="column_names"
    )


try:
    __file__
except NameError:
    __file__ = sys.argv[0]


def runSim(
    run: int,
    total_runs: int,
    sim_duration: int,
    warm_up_time: int,
    sim_start_date: datetime,
    amb_data: bool,
    random_seed: int = 101,
    save_params_csv: bool = True,
    demand_increase_percent: float = 1.0,
    activity_duration_multiplier: float = 1.0,
    print_debug_messages: bool = False,
):
    """
    Run a single discrete event simulation (DES) for the specified configuration.

    This function initializes and runs a DES_HEMS simulation model for a given run number,
    logs performance and configuration details, and optionally saves simulation parameters.

    Parameters
    ----------
    run : int
        The index of the current simulation run (starting from 0).
    total_runs : int
        Total number of simulation runs being executed.
    sim_duration : int
        The total simulation duration (excluding warm-up) in minutes or other time unit.
    warm_up_time : int
        The warm-up period to discard before recording results.
    sim_start_date : datetime
        The datetime representing the start of the simulation.
    amb_data : bool
        Flag indicating whether ambulance-specific data should be generated in the simulation.
    save_params_csv : bool, optional
        If True, simulation parameters will be saved to CSV (only on the first run). Default is True.
    demand_increase_percent : float, optional
        Factor by which demand is increased (e.g., 1.10 for a 10% increase). Default is 1.0.
    activity_duration_multiplier : float, optional
        Multiplier to adjust generated durations of activities (e.g., 1.10 for a 10% increase). Default is 1.0.
    print_debug_messages : bool, optional
        If True, enables additional debug message output. Default is False.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the simulation results.

    Notes
    -----
    - Only the first run (i.e., `run == 0`) will trigger the saving of run parameters if `save_params_csv` is True.
    - Timing information and configuration details are printed and logged for transparency.
    """
    # print(f"Inside runSim and {sim_start_date} and {what_if_sim_run}")

    print(
        f"{Utils.current_time()}: Demand increase set to {demand_increase_percent * 100}%"
    )
    logging.debug(
        f"{Utils.current_time()}: Demand increase set to {demand_increase_percent * 100}%"
    )

    start = time.process_time()

    print(f"{Utils.current_time()}: Run {run + 1} of {total_runs}")
    logging.debug(f"{Utils.current_time()}: Run {run + 1} of {total_runs}")

    # print(f"Sim start date is {sim_start_date}")
    daa_model = DES_HEMS(
        run_number=run,
        sim_duration=sim_duration,
        warm_up_duration=warm_up_time,
        sim_start_date=sim_start_date,
        amb_data=amb_data,
        random_seed=random_seed,
        demand_increase_percent=demand_increase_percent,
        activity_duration_multiplier=activity_duration_multiplier,
        print_debug_messages=print_debug_messages,
    )
    daa_model.run()

    print(
        f"{Utils.current_time()}: Run {run + 1} took {round((time.process_time() - start) / 60, 1)} minutes to run"
    )
    logging.debug(
        f"{Utils.current_time()}: Run {run + 1} took {round((time.process_time() - start) / 60, 1)} minutes to run"
    )

    # SR NOTE: This could cause issues if we decide to use 1 as the starting number of runs
    if (run == 0) and (save_params_csv):
        write_run_params(daa_model)

    return daa_model.results_df


def collateRunResults() -> None:
    """
    Collates results from a series of runs into a single csv
    """
    matching_files = glob.glob(os.path.join(Utils.RESULTS_FOLDER, "output_run_*.csv"))

    combined_df = pd.concat([pd.read_csv(f) for f in matching_files], ignore_index=True)

    combined_df.to_csv(Utils.RUN_RESULTS_CSV, index=True, encoding="utf-8-sig")

    for file in matching_files:
        os.remove(file)


def removeExistingResults(remove_run_results_csv=False) -> None:
    """
    Removes results from previous simulation runs
    """
    matching_files = glob.glob(os.path.join(Utils.RESULTS_FOLDER, "output_run_*.csv"))

    for file in matching_files:
        os.remove(file)

    all_results_file_path = os.path.join(Utils.RESULTS_FOLDER, "all_results.csv")
    if os.path.isfile(all_results_file_path):
        os.unlink(all_results_file_path)

    if remove_run_results_csv:
        run_results_file_path = os.path.join(Utils.RESULTS_FOLDER, "run_results.csv")
        if os.path.isfile(run_results_file_path):
            os.unlink(run_results_file_path)


def parallelProcessJoblib(
    total_runs: int,
    sim_duration: int,
    warm_up_time: int,
    sim_start_date: datetime,
    amb_data: bool,
    save_params_csv: bool = True,
    demand_increase_percent: float = 1.0,
    activity_duration_multiplier: float = 1.0,
    print_debug_messages: bool = False,
    master_seed=42,
    n_cores=-1,
):
    """
    Execute multiple simulation runs in parallel using joblib.

    Parameters
    ----------
    total_runs : int
        The total number of simulation runs to execute.
    sim_duration : int
        The duration of each simulation (excluding warm-up).
    warm_up_time : int
        The warm-up period to discard before recording results.
    sim_start_date : datetime
        The datetime representing the start of the simulation.
    amb_data : bool
        Flag indicating whether ambulance-specific data should be generated in the simulation.
    save_params_csv : bool, optional
        If True, simulation parameters will be saved to CSV during the first run. Default is True.
    demand_increase_percent : float, optional
        Factor by which demand is increased (e.g., 1.10 for a 10% increase). Default is 1.0.
    activity_duration_multiplier : float, optional
        Multiplier to adjust generated durations of activities (e.g., 1.10 for a 10% increase). Default is 1.0.
    print_debug_messages : bool, optional
        If True, enables additional debug message output during each run. Default is False.
    master_seed : int, optional
        Master seed used to generate the uncorrelated random number streams for replication consistency
    n_cores : int, optional
        Determines how many parallel simulations will be run at a time (which is equivalent to the
        number of cores). Default is -1, which means all available cores will be utilised.

    Returns
    -------
    list of pandas.DataFrame
        A list of DataFrames, each containing the results of an individual simulation run.

    Notes
    -----
    - This function distributes simulation runs across available CPU cores using joblib's
    `Parallel` and `delayed` utilities. Each run is executed with the `runSim` function,
    with the given configuration parameters.
    - Runs are distributed across all available CPU cores (`n_jobs=-1`).
    - Only the first run will save parameter data if `save_params_csv` is True.
    - If a single output csv is required, use of this function must be followed by
      collateRunResults()
    """

    # seeds = Utils.get_distribution_seeds(master_seed=master_seed, n_replications=total_runs,
    #                                      n_dists_per_rep=30)

    # Generate a number of uncorrelated seeds that will always be the same given the same
    # master seed (which is determined as a parameter)
    # We start with a SeedSequence from the master seed, and then generate a number of
    # child SeedSequences equal to the total number of runs
    seed_sequence = SeedSequence(master_seed).spawn(total_runs)
    # We then turn these seeds into integer random numbers, and we will pass a different seed
    # into each run of the simulation.
    seeds = [i.generate_state(1)[0] for i in seed_sequence]

    # Run the simulation in parallel, using all available cores
    return Parallel(n_jobs=n_cores)(
        delayed(runSim)(
            run=run,
            total_runs=total_runs,
            sim_duration=sim_duration,
            warm_up_time=warm_up_time,
            sim_start_date=sim_start_date,
            amb_data=amb_data,
            random_seed=seeds[run],
            save_params_csv=save_params_csv,
            demand_increase_percent=demand_increase_percent,
            activity_duration_multiplier=activity_duration_multiplier,
            print_debug_messages=print_debug_messages,
        )
        for run in range(total_runs)
    )
