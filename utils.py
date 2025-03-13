from datetime import datetime, time, timedelta
import random
import numpy as np
import pandas as pd
import ast
import scipy
import calendar

class Utils:

    RESULTS_FOLDER = 'data'
    ALL_RESULTS_CSV = f'{RESULTS_FOLDER}/all_results.csv'
    RUN_RESULTS_CSV = f'{RESULTS_FOLDER}/run_results.csv'
    HISTORICAL_FOLDER = 'historical_data'
    DISTRIBUTION_FOLDER = 'distribution_data'

    # External file containing details of resources
    # hours of operation and servicing schedules
    HEMS_ROTA_DEFAULT = pd.read_csv('actual_data/HEMS_ROTA_DEFAULT.csv')
    HEMS_ROTA = pd.read_csv('actual_data/HEMS_ROTA.csv')

    SERVICING_SCHEDULES_BY_MODEL = pd.read_csv('actual_data/service_schedules_by_model.csv')

    TIME_TYPES = ["call start", "mobile", "at scene", "leaving scene", "at hospital", "handover", "clear", "stand down"]


    def __init__(self):

        # Load in mean inter_arrival_times
        self.inter_arrival_rate_df = pd.read_csv('distribution_data/inter_arrival_times.csv')
        self.hourly_arrival_by_qtr_probs_df = pd.read_csv('distribution_data/hourly_arrival_by_qtr_probs.csv')
        self.hour_by_ampds_df = pd.read_csv('distribution_data/hour_by_ampds_card_probs.csv')
        self.sex_by_ampds_df = pd.read_csv('distribution_data/sex_by_ampds_card_probs.csv')
        self.callsign_by_ampds_and_hour_df = pd.read_csv('distribution_data/callsign_group_by_ampds_card_and_hour_probs.csv')
        self.vehicle_type_by_month_df = pd.read_csv('distribution_data/vehicle_type_by_month_probs.csv')
        self.hems_result_by_callsign_group_and_vehicle_type_df = pd.read_csv('distribution_data/hems_result_by_callsign_group_and_vehicle_type_probs.csv')
        self.pt_outcome_by_hems_result_df = pd.read_csv('distribution_data/pt_outcome_by_hems_result_probs.csv')
        # Import maximum call duration times
        self.min_max_values_df = pd.read_csv('actual_data/upper_allowable_time_bounds.csv')

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

        # Read in incident per day distribution data into a dictionary
        activity_time_data = []
        with open("distribution_data/inc_per_day_distributions.txt", "r") as inFile:
            inc_per_day_data = ast.literal_eval(inFile.read())
        inFile.close()
        self.inc_per_day_distr = inc_per_day_data


    def current_time() -> str:
        """
            Return the current time as a string in the format hh:mm:ss
        """
        now = datetime.now()
        return now.strftime("%H:%M:%S")

    def date_time_of_call(self, start_dt: str, elapsed_time: int) -> list[int, int, str, int, pd.Timestamp]:
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
            This function will allocate and return a vehicle type
            based on the month and callsign group
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
            This function will allocate and return an patient outcome
            based on the HEMS result
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
            if i['ampds_card'] == str(ampds_card):
                #print('Match')
                distribution = i['best_fit']

        #print(f"Getting age for {ampds_card}")
        #print(distribution)

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

        # Calculate the maximum time allowed for given type of job cycle time
        max_time = self.min_max_values_df[self.min_max_values_df['time'] == time_type].max_value_mins.iloc[0]
        min_time = self.min_max_values_df[self.min_max_values_df['time'] == time_type].min_value_mins.iloc[0]

        for i in self.activity_time_distr:
            #print(i)
            if (i['vehicle_type'] == vehicle_type) & (i['time_type'] == time_type):
                #print('Match')
                distribution = i['best_fit']

        sampled_time = -1000

        while (min_time > sampled_time) or (sampled_time > max_time):
            sampled_time = self.sample_from_distribution(distribution)

        return sampled_time

    def inc_per_day(self, quarter: int) -> float:
        """
            This function will return a dictionary containing
            the distribution and parameters for the distribution
            that match the provided HEMS vehicle type and time type

        """

        season = 'summer' if quarter in [2, 3] else 'winter'

        distribution = {}
        max_n = 0
        min_n = 0

        for i in self.inc_per_day_distr:
            #print(i)
            if (i['season'] == season):
                #print('Match')
                distribution = i['best_fit']
                max_n = i['max_n_per_day']
                min_n = i['min_n_per_day']

        sampled_inc_per_day = -1
        
        while not (sampled_inc_per_day >= min_n and sampled_inc_per_day <= max_n):
            sampled_inc_per_day = self.sample_from_distribution(distribution)

        return sampled_inc_per_day

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

    def get_nth_weekday(self, year: int, month: int, weekday: int, n: int):

        """
            Calculate  date of the nth occurrence of a weekday in a given month and year.
        """

        first_day = datetime(year, month, 1)
        first_weekday = first_day.weekday()
        days_until_weekday = (weekday - first_weekday + 7) % 7
        first_occurrence = first_day + timedelta(days=days_until_weekday)

        return first_occurrence + timedelta(weeks = n - 1)

    def get_last_weekday(self, year, month, weekday):
        """
            Return the date of the last occurrence of a weekday in a given month and year.
        """

        last_day = datetime(year, month, calendar.monthrange(year, month)[1])
        last_weekday = last_day.weekday()
        days_since_weekday = (last_weekday - weekday + 7) % 7

        return last_day - timedelta(days = days_since_weekday)

    def calculate_term_holidays(self, year):

        holidays = []

        # Jan to Easter
        start_date = datetime(year, 1, 1)

        first_mon_of_year = self.get_nth_weekday(year, 1, calendar.MONDAY, 1)

        jan_term_start = first_mon_of_year

        if start_date.weekday() in [0, 6]:
            #print(f"1st jan is {start_date.weekday()}")
            # 1st Jan is a a weekday
            jan_term_start += timedelta(days = 1)

        #print(f"Year {year} - start_date: {start_date} and term_start is {jan_term_start}")

        holidays.append({'year': int(year), 'start_date': start_date, 'end_date' : jan_term_start - timedelta(days = 1)})

        # Spring half-term

        spring_half_term_start = self.get_nth_weekday(year, 2, calendar.MONDAY, 2)
        spring_half_term_end = spring_half_term_start + timedelta(days = 4)

        holidays.append({'year': int(year), 'start_date': spring_half_term_start, 'end_date' : spring_half_term_end})

        # Easter hols

        # Calculate Good Friday
        easter_sunday = self.calculate_easter(year)

        #print(f"Easter Sunday is {easter_sunday}")
        good_friday = easter_sunday - timedelta(days = 2)

        # If ES is in March, 1st two weeks of Apri
        # Otherwise, Monday of ES week + 1 week
        start_date = easter_sunday - timedelta(days = 6)
        end_date = start_date + timedelta(days = 13)

        if easter_sunday.month == 3 or (easter_sunday.month == 4 and easter_sunday >= self.get_nth_weekday(year, 4, calendar.SUNDAY, 2)):
            start_date = self.get_nth_weekday(year, 4, calendar.MONDAY, 1)
            if  easter_sunday.month == 4 and easter_sunday >= self.get_nth_weekday(year, 4, calendar.SUNDAY, 2):
                # Will also likely be a late Easter Monday
                end_date = start_date + timedelta(days = 14)
            else:
                end_date = start_date + timedelta(days = 13)

        holidays.append({
            'year': int(year),
            'start_date': start_date,
            'end_date' : end_date
        })

        # Summer half-term

        summer_half_term_start = self.get_last_weekday(year, 5, calendar.MONDAY)
        summer_half_term_end = summer_half_term_start + timedelta(days = 6)

        holidays.append({
            'year': int(year),
            'start_date': summer_half_term_start,
            'end_date' : summer_half_term_end
        })

        # Summer Holidays

        summer_start = self.get_last_weekday(year, 7, calendar.MONDAY)
        summer_end = self.get_nth_weekday(year, 9, calendar.MONDAY, 1)

        holidays.append({
            'year': int(year),
            'start_date': summer_start,
            'end_date' : summer_end
        })

        # Autumn Term

        autumn_half_term_start = summer_end + timedelta(weeks = 8)

        if summer_end.day >= 4:
            autumn_half_term_start = summer_end + timedelta(weeks = 7)

        autumn_half_term_end = autumn_half_term_start + timedelta(days = 6)

        holidays.append({
            'year': int(year),
            'start_date': autumn_half_term_start,
            'end_date' : autumn_half_term_end
        })

        # Christmas Hols

        start_date = self.get_last_weekday(year, 12, calendar.MONDAY) - timedelta(days = 7)

        holidays.append({
            'year': int(year),
            'start_date': start_date,
            'end_date' : datetime(year, 12, 31)
        })

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
    
    
    def biased_mean(series: pd.Series, bias: float = .6) -> float:
        """

        Compute a weighted mean, favoring the larger value since demand
        likely to only increase with time

        """

        if len(series) == 1:
            return series.iloc[0]  # Return the only value if there's just one
        
        sorted_vals = np.sort(series)  # Ensure values are sorted
        weights = np.linspace(1, bias * 2, len(series))  # Increasing weights with larger values
        return np.average(sorted_vals, weights=weights)