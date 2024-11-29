from datetime import datetime
import pandas as pd

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

    def current_time():
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