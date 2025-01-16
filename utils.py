from datetime import datetime, time
import random
import numpy as np
import pandas as pd
import ast
import scipy

class Utils:

    RESULTS_FOLDER = 'data'
    ALL_RESULTS_CSV = f'{RESULTS_FOLDER}/all_results.csv'
    RUN_RESULTS_CSV = f'{RESULTS_FOLDER}/run_results.csv'

    # Based on summer Apr-Sept and winter Oct-Mar
    # This rota is going to be split into vehicle (car/helicopter) and personnel
    # Each row will only have two sets of start/end times (one pair for summer and one for winter)
    HEMS_ROTA = pd.DataFrame({
        "callsign"             : ["H70", "CC70", "H71", "CC71", "CC72"],
        "category"             : ["CC", "CC", "EC", "EC", "CC"],
        "vehicle_type"         : ["helicopter", "car", "helicopter", "car", "car"],
        "callsign_group"       : ["70", "70", "71", "71", "72"],
        "summer_start"         : [7, 7, 7, 9, 8],
        "winter_start"         : [7, 7, 7, 7, 8],
        "summer_end"           : [2, 2, 19, 19, 18],
        "winter_end"           : [2, 2, 17, 17, 18]
    })
    HEMS_ROTA.set_index("callsign", inplace=True)

    # TODO: Add servicing based on dates and typical servicing durations

    TIME_TYPES = ["call start", "mobile", "at scene", "leaving scene", "at hospital", "handover", "clear", "stand down"]


    def __init__(self):

        # Load in mean inter_arrival_times
        self.inter_arrival_rate_df = pd.read_csv('distribution_data/inter_arrival_times.csv')
        self.hour_by_ampds_df = pd.read_csv('distribution_data/hour_by_ampds_card_probs.csv')
        self.sex_by_ampds_df = pd.read_csv('distribution_data/sex_by_ampds_card_probs.csv')
        self.callsign_by_ampds_and_hour_df = pd.read_csv('distribution_data/callsign_group_by_ampds_card_and_hour_probs.csv')
        self.vehicle_type_by_month_df = pd.read_csv('distribution_data/vehicle_type_by_month_probs.csv')
        self.hems_result_by_callsign_group_and_vehicle_type_df = pd.read_csv('distribution_data/hems_result_by_callsign_group_and_vehicle_type_probs.csv')
        self.pt_outcome_by_hems_result_df = pd.read_csv('distribution_data/pt_outcome_by_hems_result_probs.csv')

        # Read in age distribution data into a dictionary
        age_data = []
        with open("distribution_data/age_distributions.txt", "r") as inFile:
            age_data = ast.literal_eval(inFile.read())
        inFile.close()
        self.age_distr = age_data

        # Read in activity time distribution data into a dictionary
        activity_time_data = []
        with open("distribution_data/activity_time_distributions.txt", "r") as inFile:
            activity_time_data = ast.literal_eval(inFile.read())
        inFile.close()
        self.activity_time_distr = activity_time_data


    def current_time() -> str:
        """
            Return the current time as a string in the format hh:mm:ss
        """
        now = datetime.now()
        return now.strftime("%H:%M:%S")

    def date_time_of_call(self, start_dt: str, elapsed_time: int) -> list[int, int, str, int, int, pd.Timestamp]:
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

        current_dt = start_dt + pd.Timedelta(elapsed_time, unit='min')

        dow = current_dt.strftime('%a')
        # 0 = Monday, 6 = Sunday
        weekday = "weekday" if current_dt.dayofweek < 5 else "weekend"

        current_hour = current_dt.hour

        current_month = current_dt.month

        current_quarter = current_dt.quarter
        
        return [dow, current_hour, weekday, current_month, current_quarter, current_dt]
    

    def inter_arrival_rate(self, hour: int, quarter: int) -> float:
        """
            This function will return the current mean inter_arrival rate in minutes
            for the provided hour of day and yearly quarter

            NOTE: not used with NSPPThinning
        """

        #print(f"IA with values hour {hour} and quarter {quarter}")
        df = self.inter_arrival_rate_df
        mean_ia = df[(df['hour'] == hour) & (df['quarter'] == quarter)]['mean_iat']

        # Currently have issue in that if hour and quarter not in data e.g. 0200 in quarter 3
        # then iloc value broken. Set default to 120 in that case.

        return 120 if len(mean_ia) == 0 else mean_ia.iloc[0]
        #return mean_ia.iloc[0]


    def ampds_code_selection(self, hour: int) -> int:
        """
            This function will allocate and return an AMPDS card category
            based on the hour of day
        """

        df = self.hour_by_ampds_df[self.hour_by_ampds_df['hour'] == hour]
        
        return pd.Series.sample(df['ampds_card'], weights = df['proportion']).iloc[0]
    
    
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


    def callsign_group_selection(self, hour: int, ampds_card: str) -> int:
        """
            This function will allocate and return an callsign group
            based on the hour of day and AMPDS card
        """

        #print(f"Callsign group selection with {hour} and {ampds_card}")

        df = self.callsign_by_ampds_and_hour_df[
            (self.callsign_by_ampds_and_hour_df['hour'] == int(hour)) &
            (self.callsign_by_ampds_and_hour_df['ampds_card'] == ampds_card)
        ]

        return  pd.Series.sample(df['callsign_group'], weights = df['proportion']).iloc[0]

    def vehicle_type_selection(self, month: int, callsign_group: str) -> int:
        """
            This function will allocate and return an callsign group
            based on the hour of day and AMPDS card
        """

        df = self.vehicle_type_by_month_df[
            (self.vehicle_type_by_month_df['month'] == int(month)) &
            (self.vehicle_type_by_month_df['callsign_group'] == int(callsign_group))
        ]

        return pd.Series.sample(df['vehicle_type'], weights = df['proportion']).iloc[0]

    def hems_result_by_callsign_group_and_vehicle_type_selection(self, callsign_group: str, vehicle_type: str) -> str:
        """
            This function will allocate a HEMS result based on callsign group and vehicle type
        """

        df = self.hems_result_by_callsign_group_and_vehicle_type_df[
            (self.hems_result_by_callsign_group_and_vehicle_type_df['callsign_group'] == int(callsign_group)) &
            (self.hems_result_by_callsign_group_and_vehicle_type_df['vehicle_type'] == vehicle_type)
        ]

        return pd.Series.sample(df['hems_result'], weights = df['proportion']).iloc[0]

    def pt_outcome_selection(self, hems_result: str) -> int:
        """
            This function will allocate and return an AMPDS card category
            based on the hour of day
        """

        #print(f"Hems result is {hems_result}")

        df = self.pt_outcome_by_hems_result_df[self.pt_outcome_by_hems_result_df['hems_result'] == hems_result]
        
        #print(df)
        return pd.Series.sample(df['pt_outcome'], weights = df['proportion']).iloc[0]

    def sex_selection(self, ampds_card: int) -> str:       
        """
            This function will allocate and return the patient sex
            based on allocated AMPDS card category
        """

        prob_female = self.sex_by_ampds_df[self.sex_by_ampds_df['ampds_card'] == ampds_card]['proportion']

        return 'Female' if (random.uniform(0, 1) < prob_female.iloc[0]) else 'Male'
    
    def age_sampling(self, ampds_card: int, max_age: int) -> float:
        """
            This function will return the patient's age based
            on sampling from the distribution that matches the allocated AMPDS card
        """

        distribution = {}

        #print(self.age_distr)

        for i in self.age_distr:
            #print(i)
            if i['ampds_card'] == ampds_card:
                #print('Match')
                distribution = i['best_fit']

        # print(f"Getting age for {ampds_card}")
        # print(distribution)

        age = 100000
        while age > max_age:
            age = self.sample_from_distribution(distribution)
        
        return age

    def activity_time(self, vehicle_type: str, time_type: str) -> float:
        """
            This function will return a dictionary containing
            the distribution and parameters for the distribution 
            that match the provided HEMS vehicle type and time type

        """

        distribution = {}

        for i in self.activity_time_distr:
            #print(i)
            if (i['vehicle_type'] == vehicle_type) & (i['time_type'] == time_type):
                #print('Match')
                distribution = i['best_fit']

        sampled_time = self.sample_from_distribution(distribution)

        return sampled_time

    def sample_from_distribution(self, distr: dict) -> float:
        """
            This function will return a single sampled float value from
            the specified distribution and parameters in the dictionay, distr.
        """

        distribution = {}
        return_list = []

        #print(distribution)

        for k,v in distr.items():
            #print(f"Key is {k}")
            sci_distr = getattr(scipy.stats, k)
            values = v

        while True:
            sampled_value = np.floor(sci_distr.rvs(**values))
            if sampled_value > 0:
                break

        return sampled_value
