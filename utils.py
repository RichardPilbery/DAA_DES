from datetime import datetime
import random
import numpy as np
import pandas as pd
import ast

import scipy

class Utils:


    RESULTS_FOLDER = 'data'
    ALL_RESULTS_CSV = f'{RESULTS_FOLDER}/all_results.csv'
    RUN_RESULTS_CSV = f'{RESULTS_FOLDER}/run_results.csv'

    # We are going to turn this on its head and start with AMPDS call category (chief complaint)
    # From there we can calculate age and sex, and desired response (HEMS might be dependent on availability, perhaps)?

    TRIAGE_CODE_DISTR = pd.DataFrame({
        "ampds_code" : ["07C03", "09E01", "12D01", "17D02P", "17D06", "17D06P", "29D06", "29D06V", "29D07V"],
        "category" : ["Burns", "Cardiac/respiratory", "Convulsions/fitting", "Falls", "Falls", "Falls", "RTC", "RTC", "RTC"],
        "prob" : [0.01, 0.2, 0.09, 0.10, 0.10, 0.10, 0.1, 0.2, 0.1], # Completely made up!
        "sex_female": [0.50, 0.27, 0.51, 0.33, 0.33, 0.33, 0.28, 0.28, 0.28] # Still need confirmation for burns proportion
    })
    TRIAGE_CODE_DISTR.set_index("ampds_code", inplace = True)


    # Based on summer Apr-Sept and winter Oct-Mar
    # This rota is going to be split into vehicle (car/helicopter) and personnel
    # Each row will only have two sets of start/end times (one pair for summer and one for winter)
    HEMS_ROTA = pd.DataFrame({
        "callsign"              : ["H70", "CC70", "H71", "CC71", "CC72"],
        "category"              : ["CC", "CC", "EC", "EC", "CC"],
        "type"                  : ["helicopter", "car", "helicopter", "car", "car"],
        "summer_start"         : [7, 7, 7, 9, 8],
        "winter_start"         : [7, 7, 7, 7, 8],
        "summer_end"           : [2, 2, 19, 19, 18],
        "winter_end"           : [2, 2, 17, 17, 18],
        "service_freq"          : [40, 0, 40, 0, 0],                         # Not applicable for cars, since always available, Service intervals for helicopters based on flying hours, which will tot up as the resource is (re)used on jobs
        "service_dur"           : [3, 1, 3, 1, 1]                               # weeks
    })
    HEMS_ROTA.set_index("callsign", inplace=True)

    TIME_TYPES = ["call start", "mobile", "at scene", "leaving scene", "at hospital", "handover", "clear", "stand down"]


    def __init__(self):

        # Load in mean inter_arrival_times
        self.inter_arrival_rate_df = pd.read_csv('distribution_data/interarrival_times.csv')
        self.hour_by_ampds_df = pd.read_csv('distribution_data/hour_by_amdpd_card_probs.csv')
        self.sex_by_ampds_df = pd.read_csv('distribution_data/sex_by_ampds_card_probs.csv')

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

    def date_time_of_call(start_dt: str, elapsed_time: int) -> list[int, int, str, int, int, pd.Timestamp]:
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
        """

        df = self.inter_arrival_rate_df
        df[df['hour'] == hour & df['quarter'] == quarter]['mean_inter_arrival_time']

    def ampds_code_selection(self, hour: int) -> int:
        """
            This function will allocate and return an AMPDS card category
            based on the hour of day
        """

        df = self.hour_by_ampds_df[self.hour_by_ampds_df['hour'] == hour]
        
        pd.Series.sample(df['ampds_card'], weights = df['proportion'])

    def sex_selection(self, ampds_card: int) -> str:       
        """
            This function will allocate and return the patient sex
            based on allocated AMPDS card category
        """

        prob_female = self.sex_by_ampds_df[self.sex_by_ampds_df['ampds_card'] == ampds_card]['proportion']

        return 'Female' if (random.uniform(0, 1) < prob_female.iloc[0]) else 'Male'
    
    def age_sampling(self, ampds_card: int) -> dict:
        """
            This function will return a dictionary containing
            the distribution and parameters for the distribution 
            that match the provided ampds_card category
        """

        distribution = {}

        for i in self.age_distr:
            #print(i)
            if i['ampds_card'] == ampds_card:
                #print('Match')
                distribution = i['best_fit']

        return distribution

    def activity_time(self, callsign: str, time_type: str, hems_result: str, pt_outcome: str) -> float:
        """
            This function will return a dictionary containing
            the distribution and parameters for the distribution 
            that match the provided ampds_card category, time type, 
            HEMS result and patient outcome
        """

        distribution = {}

        for i in self.activity_time_distr:
            #print(i)
            if (i['callsign'] == callsign) & (i['time_type'] == time_type) & (i['hems_result'] == hems_result) & (i['pt_outcome'] == pt_outcome):
                #print('Match')
                distribution = i['best_fit']


        # Use the appropriate distribution function from scipy.stats
        sci_distr = getattr(scipy.stats, list(distribution)[0])

        activity_time_min = 0

        distr = ""

        while True:
            # Use the dictionary values, identify them as kwargs for use by scipy distribution function
            a_time = np.floor(sci_distr.rvs(**list(distribution.values())[0]))
            if a_time > 0:
                # Sometimes, a negative number can crop up, which is nonsense with respect to response and activity times.
                activity_time_min = a_time
                break


        return activity_time_min

    