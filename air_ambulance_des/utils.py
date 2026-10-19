from datetime import datetime, time, timedelta
import json
import random
import numpy as np
import pandas as pd
import ast
import scipy
import calendar
from streamlit_extras.stylable_container import stylable_container
from numpy.random import SeedSequence, default_rng
from scipy.stats import (
    poisson,
    bernoulli,
    triang,
    erlang,
    weibull_min,
    exponweib,
    betabinom,
    pearson3,
    cauchy,
    chi2,
    expon,
    exponpow,
    gamma,
    lognorm,
    norm,
    powerlaw,
    rayleigh,
    uniform,
)
import logging
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent  # points to air_ambulance_des/
REPO_ROOT = PACKAGE_ROOT.parent  # points to repo root


class Utils:
    RESULTS_FOLDER = REPO_ROOT / "data"
    ACTUAL_DATA_FOLDER = REPO_ROOT / "actual_data"
    HISTORICAL_FOLDER = REPO_ROOT / "historical_data"
    DISTRIBUTION_FOLDER = REPO_ROOT / "distribution_data"

    ALL_RESULTS_CSV = RESULTS_FOLDER / "all_results.csv"
    RUN_RESULTS_CSV = RESULTS_FOLDER / "run_results.csv"

    # External file containing details of resources
    # hours of operation and servicing schedules
    HEMS_ROTA_DEFAULT = pd.read_csv(
        REPO_ROOT / ACTUAL_DATA_FOLDER / "HEMS_ROTA_DEFAULT.csv"
    )
    HEMS_ROTA = pd.read_csv(REPO_ROOT / ACTUAL_DATA_FOLDER / "HEMS_ROTA.csv")

    SERVICING_SCHEDULES_BY_MODEL = pd.read_csv(
        REPO_ROOT / ACTUAL_DATA_FOLDER / "service_schedules_by_model.csv"
    )

    TIME_TYPES = [
        "call start",
        "mobile",
        "at scene",
        "leaving scene",
        "at hospital",
        "handover",
        "clear",
        "stand down",
    ]

    def __init__(self, master_seed=SeedSequence(42), print_debug_messages=False):
        # Record the primary master seed sequence passed in to the sequence
        self.master_seed_sequence = master_seed

        ###############################
        # Set up remaining attributes #
        ###############################
        self.print_debug_messages = print_debug_messages

        # Load in mean inter_arrival_times
        self.hourly_arrival_by_qtr_probs_df = pd.read_csv(
            self.DISTRIBUTION_FOLDER / "hourly_arrival_by_qtr_probs.csv"
        )
        self.hour_by_ampds_df = pd.read_csv(
            self.DISTRIBUTION_FOLDER / "hour_by_ampds_card_probs.csv"
        )
        self.sex_by_ampds_df = pd.read_csv(
            self.DISTRIBUTION_FOLDER / "sex_by_ampds_card_probs.csv"
        )
        self.care_cat_by_ampds_df = pd.read_csv(
            self.DISTRIBUTION_FOLDER
            / "enhanced_or_critical_care_by_ampds_card_probs.csv"
        )
        # self.callsign_by_care_category_df = pd.read_csv('distribution_data/callsign_group_by_care_category_probs.csv')
        self.callsign_group_df = pd.read_csv(
            self.DISTRIBUTION_FOLDER / "callsign_group_probs.csv"
        )
        # New addition without stratification by month
        self.vehicle_type_df = pd.read_csv(
            self.DISTRIBUTION_FOLDER / "vehicle_type_probs.csv"
        )
        self.vehicle_type_quarter_df = pd.read_csv(
            self.DISTRIBUTION_FOLDER / "vehicle_type_by_quarter_probs.csv"
        )

        # ========= ARCHIVED CODE ================== ##
        # self.vehicle_type_by_month_df = pd.read_csv('distribution_data/vehicle_type_by_month_probs.csv')
        # self.inter_arrival_rate_df = pd.read_csv('distribution_data/inter_arrival_times.csv')
        # self.callsign_by_ampds_and_hour_df = pd.read_csv('distribution_data/callsign_group_by_ampds_card_and_hour_probs.csv')
        # self.callsign_by_ampds_df = pd.read_csv('distribution_data/callsign_group_by_ampds_card_probs.csv')
        # self.hems_result_by_callsign_group_and_vehicle_type_df = pd.read_csv('distribution_data/hems_result_by_callsign_group_and_vehicle_type_probs.csv')
        # self.hems_result_by_care_category_and_helicopter_benefit_df = pd.read_csv('distribution_data/hems_result_by_care_cat_and_helicopter_benefit_probs.csv')
        # self.pt_outcome_by_hems_result_and_care_category_df = pd.read_csv('distribution_data/pt_outcome_by_hems_result_and_care_category_probs.csv')
        # ========= END ARCHIVED CODE ============== #

        # NEW PATIENT OUTCOME AND HEMS RESULT DATA FRAMES
        self.patient_outcome_by_care_category_and_quarter_probs_df = pd.read_csv(
            self.DISTRIBUTION_FOLDER
            / "patient_outcome_by_care_category_and_quarter_probs.csv"
        )
        # self.hems_results_by_patient_outcome_and_quarter_and_vehicle_type_and_callsign_group_probs_df = pd.read_csv('distribution_data/hems_results_by_patient_outcome_and_quarter_and_vehicle_type_and_callsign_group_probs.csv')
        self.hems_results_by_patient_outcome_and_time_of_day_and_quarter_and_vehicle_type_and_callsign_group_probs_df = pd.read_csv(
            self.DISTRIBUTION_FOLDER
            / "hems_results_by_patient_outcome_and_time_of_day_and_quarter_and_vehicle_type_and_callsign_group_probs.csv"
        )

        # Import maximum call duration times
        self.min_max_values_df = pd.read_csv(
            self.ACTUAL_DATA_FOLDER / "upper_allowable_time_bounds.csv"
        )

        # Load ad hoc unavailability probability table
        try:
            # This file may not exist depending on when utility is called e.g. fitting
            self.ad_hoc_probs = pd.read_csv(
                self.DISTRIBUTION_FOLDER / "ad_hoc_unavailability.csv"
            )
            # Remove ad-hoc probs for any times outside of historical
        except FileNotFoundError:
            self.debug("ad_hoc spreadsheet does not exist yet")

        # Read in age distribution data into a dictionary
        age_data = []
        with open(self.DISTRIBUTION_FOLDER / "age_distributions.txt", "r") as inFile:
            age_data = ast.literal_eval(inFile.read())
        inFile.close()
        self.age_distr = age_data

        # Read in activity time distribution data into a dictionary
        activity_time_data = []
        with open(
            self.DISTRIBUTION_FOLDER / "activity_time_distributions.txt", "r"
        ) as inFile:
            activity_time_data = ast.literal_eval(inFile.read())
        inFile.close()
        self.activity_time_distr = activity_time_data

        # Read in incident per day distribution data into a dictionary
        inc_per_day_data = []
        with open(
            self.DISTRIBUTION_FOLDER / "inc_per_day_distributions.txt", "r"
        ) as inFile:
            inc_per_day_data = ast.literal_eval(inFile.read())
        inFile.close()
        self.inc_per_day_distr = inc_per_day_data

        inc_per_day_per_qtr_data = []
        with open(
            self.DISTRIBUTION_FOLDER / "inc_per_day_qtr_distributions.txt", "r"
        ) as inFile:
            inc_per_day_per_qtr_data = ast.literal_eval(inFile.read())
        inFile.close()
        self.inc_per_day_per_qtr_distr = inc_per_day_per_qtr_data

        # Incident per day samples
        with open(self.DISTRIBUTION_FOLDER / "inc_per_day_samples.json", "r") as f:
            self.incident_per_day_samples = json.load(f)

        # Turn the min-max activity times data into a format that supports easier/faster
        # lookups
        self.min_max_cache = {
            row["time"]: (row["min_value_mins"], row["max_value_mins"])
            for _, row in self.min_max_values_df.iterrows()
        }

    def setup_seeds(self):
        #########################################################
        # Control generation of required random number streams  #
        #########################################################

        # Spawn substreams for major simulation modules
        module_keys = [
            "activity_times",
            "ampds_code_selection",
            "care_category_selection",
            "callsign_group_selection",
            "vehicle_type_selection",
            "hems_result_by_callsign_group_and_vehicle_type_selection",
            "hems_result_by_care_category_and_helicopter_benefit_selection",
            "hems_results_by_patient_outcome_and_quarter_and_vehicle_type_and_callsign_group_selection",
            "pt_outcome_selection",
            "sex_selection",
            "age_sampling",
            "calls_per_day",
            "calls_per_hour",
            "predetermine_call_arrival",
            "call_iat",
            "helicopter_benefit_from_reg",
            "hems_case",
            "ad_hoc_reason_selection",
            "know_heli_benefit",
            "helicopter_benefit_from_cc",
            "helicopter_benefit_from_ec",
            "know_cc_ec_benefit",
        ]
        # Efficiently spawn substreams
        spawned = self.master_seed_sequence.spawn(len(module_keys))

        # Map substreams and RNGs to keys
        self.seed_substreams = dict(zip(module_keys, spawned))
        self.rngs = {key: default_rng(ss) for key, ss in self.seed_substreams.items()}

        self.build_seeded_distributions(self.master_seed_sequence)

    def debug(self, message: str):
        if self.print_debug_messages:
            logging.debug(message)

    def current_time() -> str:
        """
        Return the current time as a string in the format hh:mm:ss
        """
        now = datetime.now()
        return now.strftime("%H:%M:%S")

    def date_time_of_call(
        self, start_dt: str, elapsed_time: int
    ) -> list[int, int, str, int, pd.Timestamp]:
        """
        Calculate a range of time-based parameters given a specific date-time

        **Returns:**
            list(
                `dow`             : int
                `current_hour`    : int
                `weekday`         : str
                `current_month`   : int
                `current_quarter` : int
                `current_dt`      : datetime
            )

        """
        # Elapsed_time = time in minutes since simulation started

        start_dt = pd.to_datetime(start_dt)

        current_dt = start_dt + pd.Timedelta(elapsed_time, unit="min")

        dow = current_dt.strftime("%a")
        # 0 = Monday, 6 = Sunday
        weekday = "weekday" if current_dt.dayofweek < 5 else "weekend"

        current_hour = current_dt.hour

        current_month = current_dt.month

        current_quarter = current_dt.quarter

        return [dow, current_hour, weekday, current_month, current_quarter, current_dt]

    # SR 2025-04-17: Commenting out as I believe this is no longer used due to changes
    # in the way inter-arrival times for calls are managed
    # def inter_arrival_rate(self, hour: int, quarter: int) -> float:
    #     """
    #         This function will return the current mean inter_arrival rate in minutes
    #         for the provided hour of day and yearly quarter

    #         NOTE: not used with NSPPThinning
    #     """

    #     #print(f"IA with values hour {hour} and quarter {quarter}")
    #     df = self.inter_arrival_rate_df
    #     mean_ia = df[(df['hour'] == hour) & (df['quarter'] == quarter)]['mean_iat']

    #     # Currently have issue in that if hour and quarter not in data e.g. 0200 in quarter 3
    #     # then iloc value broken. Set default to 120 in that case.

    #     return 120 if len(mean_ia) == 0 else mean_ia.iloc[0]
    #     #return mean_ia.iloc[0]

    def ampds_code_selection(self, hour: int) -> int:
        """
        This function will allocate and return an AMPDS card category
        based on the hour of day
        """

        df = self.hour_by_ampds_df[self.hour_by_ampds_df["hour"] == hour]

        return pd.Series.sample(
            df["ampds_card"],
            weights=df["proportion"],
            random_state=self.rngs["ampds_code_selection"],
        ).iloc[0]

    def is_time_in_range(self, current: int, start: int, end: int) -> bool:
        """
        Function to check if a given time is within a range of start and end times on a 24-hour clock.

        Parameters:
        - current (datetime.time): The time to check.
        - start (datetime.time): The start time.
        - end (datetime.time): The end time.

        """

        current = time(current, 0)
        start = time(start, 0)
        end = time(end, 0)

        if start <= end:
            # Range does not cross midnight
            return start <= current < end
        else:
            # Range crosses midnight
            return current >= start or current < end

    def care_category_selection(self, ampds_card: str) -> str:
        """
        This function will allocate and return an care category
        based on AMPDS card
        """

        # print(f"Callsign group selection with {hour} and {ampds_card}")

        df = self.care_cat_by_ampds_df[
            (self.care_cat_by_ampds_df["ampds_card"] == ampds_card)
        ]

        return pd.Series.sample(
            df["care_category"],
            weights=df["proportion"],
            random_state=self.rngs["care_category_selection"],
        ).iloc[0]

    # def LEGACY_callsign_group_selection(self, ampds_card: str) -> int:
    #     """
    #         This function will allocate and return an callsign group
    #         based on AMPDS card
    #     """

    #     #print(f"Callsign group selection with {hour} and {ampds_card}")

    #     df = self.callsign_by_ampds_df[
    #         (self.callsign_by_ampds_df['ampds_card'] == ampds_card)
    #     ]

    #     return pd.Series.sample(df['callsign_group'], weights = df['proportion'],
    #                             random_state=self.rngs["callsign_group_selection"]).iloc[0]

    # def callsign_group_selection(self, care_category: str) -> int:
    #     """
    #         This function will allocate and return an callsign group
    #         based on the care category
    #     """

    #     #print(f"Callsign group selection with {hour} and {ampds_card}")

    #     df = self.callsign_by_care_category_df[
    #         (self.callsign_by_care_category_df['care_category'] == care_category)
    #     ]

    #     return pd.Series.sample(df['callsign_group'], weights = df['proportion'],
    #                             random_state=self.rngs["callsign_group_selection"]).iloc[0]

    def callsign_group_selection(self) -> int:
        """
        This function will allocate and return an callsign group
        based on the care category
        """

        # print(f"Callsign group selection with {hour} and {ampds_card}")

        return pd.Series.sample(
            self.callsign_group_df["callsign_group"],
            weights=self.callsign_group_df["proportion"],
            random_state=self.rngs["callsign_group_selection"],
        ).iloc[0]

    # def vehicle_type_selection(self, callsign_group: str) -> int:
    #     """
    #         This function will allocate and return a vehicle type
    #         based callsign group
    #     """

    #     df = self.vehicle_type_df[
    #         (self.vehicle_type_df['callsign_group'] == int(callsign_group)) # Cater for Other
    #     ]

    #     return pd.Series.sample(df['vehicle_type'], weights = df['proportion'],
    #                             random_state=self.rngs["vehicle_type_selection"]).iloc[0]

    def vehicle_type_selection_qtr(self, callsign_group: str, qtr: int) -> int:
        """
        This function will allocate and return a vehicle type
        based callsign group
        """

        df = self.vehicle_type_quarter_df[
            (
                self.vehicle_type_quarter_df["callsign_group"] == int(callsign_group)
            )  # Cater for Other
            & (self.vehicle_type_quarter_df["quarter"] == int(qtr))
        ]

        return pd.Series.sample(
            df["vehicle_type"],
            weights=df["proportion"],
            random_state=self.rngs["vehicle_type_selection"],
        ).iloc[0]

    # def hems_result_by_callsign_group_and_vehicle_type_selection(self, callsign_group: str, vehicle_type: str) -> str:
    #     """
    #         This function will allocate a HEMS result based on callsign group and vehicle type
    #     """

    #     df = self.hems_result_by_callsign_group_and_vehicle_type_df[
    #         (self.hems_result_by_callsign_group_and_vehicle_type_df['callsign_group'] == int(callsign_group)) &
    #         (self.hems_result_by_callsign_group_and_vehicle_type_df['vehicle_type'] == vehicle_type)
    #     ]

    #     return pd.Series.sample(df['hems_result'], weights = df['proportion'],
    #                             random_state=self.rngs["hems_result_by_callsign_group_and_vehicle_type_selection"]).iloc[0]

    # def hems_result_by_care_category_and_helicopter_benefit_selection(self, care_category: str, helicopter_benefit: str) -> str:
    #     """
    #         This function will allocate a HEMS result based on care category and helicopter benefit
    #     """

    #     df = self.hems_result_by_care_category_and_helicopter_benefit_df[
    #         (self.hems_result_by_care_category_and_helicopter_benefit_df['care_cat'] == care_category) &
    #         (self.hems_result_by_care_category_and_helicopter_benefit_df['helicopter_benefit'] == helicopter_benefit)
    #     ]

    #     return pd.Series.sample(df['hems_result'], weights = df['proportion'],
    #                             random_state=self.rngs["hems_result_by_care_category_and_helicopter_benefit_selection"]).iloc[0]

    def hems_results_by_patient_outcome_and_time_of_day_and_quarter_and_vehicle_type_and_callsign_group_probs(
        self,
        pt_outcome: str,
        quarter: int,
        vehicle_type: str,
        callsign_group: int,
        hour: int,
    ):
        """
        This function will allocate a HEMS result based on patient outcome, yearly quarter and HEMS deets.
        """

        if (hour >= 7) and (hour <= 18):
            time_of_day = "day"
        else:
            time_of_day = "night"

        self.debug(f"Hour is {hour}: for hems_result sampling, this is {time_of_day}")

        # (self.hems_results_by_patient_outcome_and_quarter_and_vehicle_type_and_callsign_group_probs_df.head())

        df = self.hems_results_by_patient_outcome_and_time_of_day_and_quarter_and_vehicle_type_and_callsign_group_probs_df[
            (
                self.hems_results_by_patient_outcome_and_time_of_day_and_quarter_and_vehicle_type_and_callsign_group_probs_df[
                    "pt_outcome"
                ]
                == pt_outcome
            )
            & (
                self.hems_results_by_patient_outcome_and_time_of_day_and_quarter_and_vehicle_type_and_callsign_group_probs_df[
                    "quarter"
                ]
                == quarter
            )
            & (
                self.hems_results_by_patient_outcome_and_time_of_day_and_quarter_and_vehicle_type_and_callsign_group_probs_df[
                    "vehicle_type"
                ]
                == vehicle_type
            )
            & (
                self.hems_results_by_patient_outcome_and_time_of_day_and_quarter_and_vehicle_type_and_callsign_group_probs_df[
                    "callsign_group"
                ]
                == callsign_group
            )
            & (
                self.hems_results_by_patient_outcome_and_time_of_day_and_quarter_and_vehicle_type_and_callsign_group_probs_df[
                    "time_of_day"
                ]
                == time_of_day
            )
        ]

        # print(df.head())

        return pd.Series.sample(
            df["hems_result"],
            weights=df["proportion"],
            random_state=self.rngs[
                "hems_results_by_patient_outcome_and_quarter_and_vehicle_type_and_callsign_group_selection"
            ],
        ).iloc[0]

    def pt_outcome_selection(self, care_category: str, quarter: int) -> int:
        """
        This function will allocate and return an patient outcome
        based on the HEMS result
        """

        df = self.patient_outcome_by_care_category_and_quarter_probs_df[
            (
                self.patient_outcome_by_care_category_and_quarter_probs_df[
                    "care_category"
                ]
                == care_category
            )
            & (
                self.patient_outcome_by_care_category_and_quarter_probs_df["quarter"]
                == quarter
            )
        ]

        # print(df)
        return pd.Series.sample(
            df["pt_outcome"],
            weights=df["proportion"],
            random_state=self.rngs["pt_outcome_selection"],
        ).iloc[0]

    # TODO: RANDOM SEED SETTING
    def sex_selection(self, ampds_card: int) -> str:
        """
        This function will allocate and return the patient sex
        based on allocated AMPDS card category
        """

        prob_female = self.sex_by_ampds_df[
            self.sex_by_ampds_df["ampds_card"] == ampds_card
        ]["proportion"]

        return (
            "Female"
            if (self.rngs["sex_selection"].uniform(0, 1) < prob_female.iloc[0])
            else "Male"
        )

    # TODO: RANDOM SEED SETTING
    def age_sampling(self, ampds_card: int, max_age: int) -> float:
        """
        This function will return the patient's age based
        on sampling from the distribution that matches the allocated AMPDS card
        """

        distribution = {}

        # print(self.age_distr)

        for i in self.age_distr:
            # print(i)
            if i["ampds_card"] == str(ampds_card):
                # print('Match')
                distribution = i["best_fit"]

        # print(f"Getting age for {ampds_card}")
        # print(distribution)

        age = 100000
        while age > max_age:
            age = self.sample_from_distribution(
                distribution, rng=self.rngs["age_sampling"]
            )

        return age

    def activity_time(self, vehicle_type: str, time_type: str) -> float:
        """
        This function will return a dictionary containing
        the distribution and parameters for the distribution
        that match the provided HEMS vehicle type and time type

        """

        dist = self.activity_time_distr.get((vehicle_type, time_type))
        # print(dist)

        if dist is None:
            raise ValueError(f"No distribution found for ({vehicle_type}, {time_type})")

        try:
            min_time, max_time = self.min_max_cache[time_type]
            # print(f"Min time {time_type}: {min_time}")
            # print(f"Max time {time_type}: {max_time}")
        except KeyError:
            raise ValueError(f"Min/max bounds not found for time_type='{time_type}'")

        sampled_time = -1
        while not (min_time <= sampled_time <= max_time):
            sampled_time = dist.sample()

        # print(sampled_time)

        return sampled_time

    def inc_per_day(self, quarter: int, quarter_or_season: str = "season") -> float:
        """
        This function will return the number of incidents for a given
        day

        """

        season = "summer" if quarter in [2, 3] else "winter"

        distribution = {}
        max_n = 0
        min_n = 0

        distr_data = (
            self.inc_per_day_distr
            if quarter_or_season == "season"
            else self.inc_per_day_per_qtr_distr
        )

        for i in distr_data:
            # print(i)
            if quarter_or_season == "season":
                if i["season"] == season:
                    # print('Match')
                    distribution = i["best_fit"]
                    max_n = i["max_n_per_day"]
                    min_n = i["min_n_per_day"]
            else:
                if i["quarter"] == quarter:
                    # print('Match')
                    distribution = i["best_fit"]
                    max_n = i["max_n_per_day"]
                    min_n = i["min_n_per_day"]

        sampled_inc_per_day = -1

        while not (sampled_inc_per_day >= min_n and sampled_inc_per_day <= max_n):
            sampled_inc_per_day = self.sample_from_distribution(
                distribution, rng=self.rngs["calls_per_day"]
            )

        return sampled_inc_per_day

    def inc_per_day_samples(
        self, quarter: int, quarter_or_season: str = "season"
    ) -> float:
        """
        This function will return a dictionary containing
        the distribution and parameters for the distribution
        that match the provided HEMS vehicle type and time type

        """

        season = "summer" if quarter in [2, 3] else "winter"

        if quarter_or_season == "season":
            return self.rngs["calls_per_day"].choice(
                self.incident_per_day_samples[season]
            )
        else:
            return self.rngs["calls_per_day"].choice(
                self.incident_per_day_samples[f"Q{quarter}"]
            )

    def sample_from_distribution(self, distr: dict, rng: np.random.Generator) -> float:
        """
        Sample a single float value from a seeded scipy distribution.

        Parameters
        ----------
        distr : dict
            A dictionary with one key (the distribution name) and value as the parameters.
        rng : np.random.Generator
            A seeded RNG from the simulation's RNG stream pool.

        Returns
        -------
        float
            A positive sampled value from the specified distribution.
        """
        if len(distr) != 1:
            raise ValueError("Expected one distribution name in distr dictionary")

        dist_name, params = list(distr.items())[0]
        sci_distr = getattr(scipy.stats, dist_name)

        while True:
            sampled_value = np.floor(sci_distr.rvs(random_state=rng, **params))
            if sampled_value > 0:
                return sampled_value

    def get_nth_weekday(self, year: int, month: int, weekday: int, n: int):
        """
        Calculate  date of the nth occurrence of a weekday in a given month and year.
        """

        first_day = datetime(year, month, 1)
        first_weekday = first_day.weekday()
        days_until_weekday = (weekday - first_weekday + 7) % 7
        first_occurrence = first_day + timedelta(days=days_until_weekday)

        return first_occurrence + timedelta(weeks=n - 1)

    def get_last_weekday(self, year, month, weekday):
        """
        Return the date of the last occurrence of a weekday in a given month and year.
        """

        last_day = datetime(year, month, calendar.monthrange(year, month)[1])
        last_weekday = last_day.weekday()
        days_since_weekday = (last_weekday - weekday + 7) % 7

        return last_day - timedelta(days=days_since_weekday)

    def calculate_term_holidays(self, year):
        holidays = []

        # Jan to Easter
        start_date = datetime(year, 1, 1)

        first_mon_of_year = self.get_nth_weekday(year, 1, calendar.MONDAY, 1)

        jan_term_start = first_mon_of_year

        if start_date.weekday() in [0, 6]:
            # print(f"1st jan is {start_date.weekday()}")
            # 1st Jan is a a weekday
            jan_term_start += timedelta(days=1)

        # print(f"Year {year} - start_date: {start_date} and term_start is {jan_term_start}")

        holidays.append(
            {
                "year": int(year),
                "start_date": start_date,
                "end_date": jan_term_start - timedelta(days=1),
            }
        )

        # Spring half-term

        spring_half_term_start = self.get_nth_weekday(year, 2, calendar.MONDAY, 2)
        spring_half_term_end = spring_half_term_start + timedelta(days=4)

        holidays.append(
            {
                "year": int(year),
                "start_date": spring_half_term_start,
                "end_date": spring_half_term_end,
            }
        )

        # Easter hols

        # Calculate Good Friday
        easter_sunday = self.calculate_easter(year)

        # print(f"Easter Sunday is {easter_sunday}")
        good_friday = easter_sunday - timedelta(days=2)

        # If ES is in March, 1st two weeks of Apri
        # Otherwise, Monday of ES week + 1 week
        start_date = easter_sunday - timedelta(days=6)
        end_date = start_date + timedelta(days=13)

        if easter_sunday.month == 3 or (
            easter_sunday.month == 4
            and easter_sunday >= self.get_nth_weekday(year, 4, calendar.SUNDAY, 2)
        ):
            start_date = self.get_nth_weekday(year, 4, calendar.MONDAY, 1)
            if easter_sunday.month == 4 and easter_sunday >= self.get_nth_weekday(
                year, 4, calendar.SUNDAY, 2
            ):
                # Will also likely be a late Easter Monday
                end_date = start_date + timedelta(days=14)
            else:
                end_date = start_date + timedelta(days=13)

        holidays.append(
            {"year": int(year), "start_date": start_date, "end_date": end_date}
        )

        # Summer half-term

        summer_half_term_start = self.get_last_weekday(year, 5, calendar.MONDAY)
        summer_half_term_end = summer_half_term_start + timedelta(days=6)

        holidays.append(
            {
                "year": int(year),
                "start_date": summer_half_term_start,
                "end_date": summer_half_term_end,
            }
        )

        # Summer Holidays

        summer_start = self.get_last_weekday(year, 7, calendar.MONDAY)
        summer_end = self.get_nth_weekday(year, 9, calendar.MONDAY, 1)

        holidays.append(
            {"year": int(year), "start_date": summer_start, "end_date": summer_end}
        )

        # Autumn Term

        autumn_half_term_start = summer_end + timedelta(weeks=8)

        if summer_end.day >= 4:
            autumn_half_term_start = summer_end + timedelta(weeks=7)

        autumn_half_term_end = autumn_half_term_start + timedelta(days=6)

        holidays.append(
            {
                "year": int(year),
                "start_date": autumn_half_term_start,
                "end_date": autumn_half_term_end,
            }
        )

        # Christmas Hols

        start_date = self.get_last_weekday(year, 12, calendar.MONDAY) - timedelta(
            days=7
        )

        holidays.append(
            {
                "year": int(year),
                "start_date": start_date,
                "end_date": datetime(year, 12, 31),
            }
        )

        return pd.DataFrame(holidays)

    def calculate_easter(self, year):
        """
        Calculate the date of Easter Sunday for a given year using the Anonymous Gregorian algorithm.
        Converted to Python from this SO answer: https://stackoverflow.com/a/49558298/3650230
        Really interesting rabbit hole to go down about this in the whole thread: https://stackoverflow.com/questions/2192533/function-to-return-date-of-easter-for-the-given-year
        """

        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        k = c % 4
        i = (c - k) // 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1

        return datetime(year, month, day)

    def years_between(self, start_date: datetime, end_date: datetime) -> list[int]:
        return list(range(start_date.year, end_date.year + 1))

    def biased_mean(series: pd.Series, bias: float = 0.5) -> float:
        """

        Compute a weighted mean, favoring the larger value since demand
        likely to only increase with time

        """

        if len(series) == 1:
            return series.iloc[0]  # Return the only value if there's just one

        sorted_vals = np.sort(series)  # Ensure values are sorted
        weights = np.linspace(
            1, bias * 2, len(series)
        )  # Increasing weights with larger values
        return np.average(sorted_vals, weights=weights)

    def build_seeded_distributions(self, seed_seq):
        """
        Build a dictionary of seeded distributions keyed by (vehicle_type, time_type)

        Returns
        -------
        dict of (vehicle_type, time_type) -> SeededDistribution

        e.g. {('helicopter', 'time_allocation'): <utils.SeededDistribution object at 0x000001521627C750>,
        ('car', 'time_allocation'): <utils.SeededDistribution object at 0x000001521627D410>,
        ('helicopter', 'time_mobile'): <utils.SeededDistribution object at 0x0000015216267E90>}
        """
        n = len(self.activity_time_distr)
        rngs = [default_rng(s) for s in seed_seq.spawn(n)]

        dist_map = {
            "poisson": poisson,
            "bernoulli": bernoulli,
            "triang": triang,
            "erlang": erlang,
            "weibull_min": weibull_min,
            "expon_weib": exponweib,
            "betabinom": betabinom,
            "pearson3": pearson3,
            "cauchy": cauchy,
            "chi2": chi2,
            "expon": expon,
            "exponential": expon,  # alias for consistency
            "exponpow": exponpow,
            "gamma": gamma,
            "lognorm": lognorm,
            "norm": norm,
            "normal": norm,  # alias for consistency
            "powerlaw": powerlaw,
            "rayleigh": rayleigh,
            "uniform": uniform,
        }

        seeded_distributions = {}

        # print(activity_time_distr)
        # print(len(self.activity_time_distr))
        i = 0
        for entry, rng in zip(self.activity_time_distr, rngs):
            i += 1
            vt = entry["vehicle_type"]
            tt = entry["time_type"]
            best_fit = entry["best_fit"]

            # dist_name = best_fit['dist'].lower()
            dist_name = list(best_fit.keys())[0]
            dist_cls = dist_map.get(dist_name)
            self.debug(
                f"{i}/{len(self.activity_time_distr)} for {vt} {tt} set up {dist_name} {dist_cls} with params: {best_fit[dist_name]}"
            )

            if dist_cls is None:
                raise ValueError(f"Unsupported distribution type: {dist_name}")

            params = best_fit[dist_name]

            seeded_distributions[(vt, tt)] = SeededDistribution(dist_cls, rng, **params)

        self.activity_time_distr = seeded_distributions

        return seeded_distributions

    def sample_ad_hoc_reason(self, hour: int, quarter: int, registration: str) -> bool:
        """
        Sample from ad hoc unavailability probability table based on time bin and quarter.
        Returns a reason string (e.g. 'available', 'crew', 'weather', etc.).
        """
        # Determine time bin
        if 0 <= hour <= 5:
            bin_label = "00-05"
        elif 6 <= hour <= 11:
            bin_label = "06-11"
        elif 12 <= hour <= 17:
            bin_label = "12-17"
        else:
            bin_label = "18-23"

        # Main subset for registration, bin, and quarter
        subset = self.ad_hoc_probs[
            (self.ad_hoc_probs["registration"] == registration)
            & (self.ad_hoc_probs["six_hour_bin"] == bin_label)
            & (self.ad_hoc_probs["quarter"] == quarter)
        ]

        if subset.empty:
            self.debug(
                f"[AD HOC] No data for {registration} in bin {bin_label}, Q{quarter}. Falling back to all registrations."
            )
            subset = self.ad_hoc_probs[
                (self.ad_hoc_probs["six_hour_bin"] == bin_label)
                & (self.ad_hoc_probs["quarter"] == quarter)
            ]
            if subset.empty:
                self.debug(
                    f"[AD HOC] No data at all for bin {bin_label}, Q{quarter}. Defaulting to 'available'."
                )
                return "available"

        filled_subset = subset.copy()

        for idx, row in filled_subset.iterrows():
            if pd.isna(row["probability"]):
                reason = row["reason"]
                # Try registration-wide average across all bins in this quarter
                reg_avg = self.ad_hoc_probs[
                    (self.ad_hoc_probs["registration"] == row["registration"])
                    & (self.ad_hoc_probs["quarter"] == row["quarter"])
                    & (self.ad_hoc_probs["reason"] == reason)
                    & (self.ad_hoc_probs["probability"].notna())
                ]["probability"].mean()

                if pd.isna(reg_avg):
                    # Fall back: average for this bin/quarter across all registrations
                    fallback_avg = self.ad_hoc_probs[
                        (self.ad_hoc_probs["six_hour_bin"] == row["six_hour_bin"])
                        & (self.ad_hoc_probs["quarter"] == row["quarter"])
                        & (self.ad_hoc_probs["reason"] == reason)
                        & (self.ad_hoc_probs["probability"].notna())
                    ]["probability"].mean()

                    if pd.notna(fallback_avg):
                        self.debug(
                            f"[AD HOC] Missing prob for {registration}, {bin_label}, Q{quarter}, reason '{reason}'. Using fallback avg across registrations: {fallback_avg:.4f}"
                        )
                        filled_subset.at[idx, "probability"] = fallback_avg
                    else:
                        self.debug(
                            f"[AD HOC] Missing prob for {registration}, {bin_label}, Q{quarter}, reason '{reason}'. No fallback avg found. Defaulting to 'available'."
                        )
                        return "available"
                else:
                    self.debug(
                        f"[AD HOC] Missing prob for {registration}, {bin_label}, Q{quarter}, reason '{reason}'. Using reg-wide avg: {reg_avg:.4f}"
                    )
                    filled_subset.at[idx, "probability"] = reg_avg

        if filled_subset["probability"].isna().any():
            self.debug(
                f"[AD HOC] Still missing probabilities after imputation for {registration}, {bin_label}, Q{quarter}. Defaulting to 'available'."
            )
            return "available"

        total_prob = filled_subset["probability"].sum()
        if total_prob == 0 or pd.isna(total_prob):
            self.debug(
                f"[AD HOC] Total probability is zero or NaN for {registration}, {bin_label}, Q{quarter}. Defaulting to 'available'."
            )
            return "available"

        norm_probs = filled_subset["probability"] / total_prob

        sampled_reason = self.rngs["ad_hoc_reason_selection"].choice(
            filled_subset["reason"].tolist(), p=norm_probs.tolist()
        )

        return sampled_reason


class SeededDistribution:
    def __init__(self, dist, rng, **kwargs):
        self.dist = dist(**kwargs)
        self.rng = rng

    def sample(self, size=None):
        return self.dist.rvs(size=size, random_state=self.rng)


def format_diff(value):
    if value > 0:
        return f"**:red[+{value:.0f}]** from historical"
    elif value < 0:
        return f"**:green[{value:.0f}]** from historical"
    else:
        return "**:gray[no difference]** from historical"


# Mapping months to numbers
MONTH_MAPPING = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

REVERSE_MONTH_MAPPING = {v: k for k, v in MONTH_MAPPING.items()}


def get_rota_month_strings(start_month, end_month):
    # Convert selected months to numbers
    start_month_num = MONTH_MAPPING[start_month]
    end_month_num = MONTH_MAPPING[end_month]

    # Summer rota
    summer_start_date = f"1st {start_month}"
    summer_end_day = calendar.monthrange(2024, end_month_num)[
        1
    ]  # Assume leap year for Feb
    summer_end_date = f"{summer_end_day}th {end_month}"

    # Winter rota
    winter_start_num = (end_month_num % 12) + 1  # month after summer end
    winter_end_num = (
        (start_month_num - 1) if start_month_num > 1 else 12
    )  # month before summer start

    winter_start_date = f"1st {REVERSE_MONTH_MAPPING[winter_start_num]}"
    winter_end_day = calendar.monthrange(2024, winter_end_num)[
        1
    ]  # same leap year assumption
    winter_end_date = f"{winter_end_day}th {REVERSE_MONTH_MAPPING[winter_end_num]}"

    return (
        start_month_num,
        end_month_num,
        summer_start_date,
        summer_end_date,
        summer_end_day,
        winter_start_date,
        winter_end_date,
        winter_end_day,
    )


def format_sigfigs(x, sigfigs=4):
    from math import log10, floor

    try:
        if x == 0:
            return "0.0"
        elif x < 1e-10:
            return "<1e-10"
        else:
            digits = sigfigs - 1 - floor(log10(abs(x)))
            return f"{x:.{digits}f}"
    except (ValueError, TypeError):
        return str(x)


COLORSCHEME = {
    "red": "#D50032",
    "navy": "#00205B",
    "blue": "#1D428A",
    "teal": "#00B0B9",
    "lightblue": "#C0F0F2",
    "green": "#56E39F",
    "orange": "#FFA400",
    "yellow": "#F8C630",
    "darkgreen": "#264027",
    "verylightblue": "#D5F5F6",
    "lightgrey": "#CCCCCC",
    "darkgrey": "#4D4D4D",
    "charcoal": "#1F1F1F",
}


# 99th Percentile
def q99(x):
    return x.quantile(0.99)


# 95th Percentile
def q95(x):
    return x.quantile(0.95)


# 90th Percentile
def q90(x):
    return x.quantile(0.9)


# 10th Percentile
def q10(x):
    return x.quantile(0.1)


# 75th Percentile
def q75(x):
    return x.quantile(0.75)


# 25th Percentile
def q25(x):
    return x.quantile(0.25)


def to_military_time(hour: int) -> str:
    return f"{hour:02d}00"


def iconMetricContainer(
    key, icon_unicode, css_style=None, icon_color="grey", family="filled", type="icons"
):
    """Function that returns a CSS styled container for adding a Material Icon to a Streamlit st.metric value

    CREDIT for starter version of this code: https://discuss.streamlit.io/t/adding-an-icon-to-a-st-metric-easily/59140?u=sammi1

    Args:
        key (str): Unique key for the component
        iconUnicode (str): Code point for a Material Icon, you can find them here https://fonts.google.com/icons. Sample \e8b6
        css_style(str, optional): Additional CSS to apply
        icon_color (str, optional): HTML Hex color value for the icon. Defaults to 'grey'.
        family(str, optional): "filled" or "outline". Only works with type = "icons"
        type(str, optional): "icons" or "symbols"

    Returns:
        DeltaGenerator: A container object. Elements can be added to this container using either the 'with'
        notation or by calling methods directly on the returned object.
    """

    if (family == "filled") and (type == "icons"):
        font_family = "Material Icons"
    elif (family == "outline") and (type == "icons"):
        font_family = "Material Icons Outlined"
    # elif (family == "filled") and (type=="symbols"):
    #     font_family = "Material Symbols"
    elif type == "symbols":
        font_family = "Material Symbols Outlined"
    else:
        print("ERROR - Check Params for iconMetricContainer")
        font_family = "Material Icons"

    css_style_icon = f"""
                    div[data-testid="stMetricValue"]>div::before
                    {{
                        font-family: {font_family};
                        content: "\{icon_unicode}";
                        vertical-align: -20%;
                        color: {icon_color};
                    }}
                    """

    if css_style is not None:
        css_style_icon += """

        """

        css_style_icon += css_style

    iconMetric = stylable_container(key=key, css_styles=css_style_icon)
    return iconMetric
