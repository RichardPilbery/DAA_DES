from datetime import datetime
import pandas as pd

class Utils:

    ALL_RESULTS_CSV = 'data/all_results.csv'

    TRIAGE_CODE_DISTR = pd.DataFrame({
        "ampds_code" : ["07C03", "09E01", "12D01", "17D02P", "17D06", "17D06P", "29D06", "29D06V", "29D07V"],
        "category" : ["Burns", "Cardiac/respiratory", "Convulsions/fitting", "Falls", "Falls", "Falls", "RTC", "RTC", "RTC"],
        "prob" : [0.01, 0.2, 0.09, 0.10, 0.10, 0.10, 0.1, 0.2, 0.1], # Completely made up!
        "sex_female": [0.50, 0.27, 0.51, 0.33, 0.33, 0.33, 0.28, 0.28, 0.28] # Still need confirmation for burns proportion
    })
    TRIAGE_CODE_DISTR.set_index("ampds_code", inplace = True)


    # Based on summer Apr-Sept and winter Oct-Mar
    HEMS_ROTA = pd.DataFrame({
        "callsign"      : ["H70", "CC70", "H71", "CC71", "CC72"],
        "category"      : ["CC", "CC", "EC", "EC", "CC"],
        "type"          : ["helicopter", "car", "helicopter", "car", "car"],
        "summer_start1" : ["07:00", "07:00", "09:00", "09:00", "08:00"],
        "winter_start1" : ["07:00", "07:00", "07:00", "07:00", "08:00"],
        "summer_end1"   : ["17:00", "17:00", "19:00", "19:00", "18:00"],
        "winter_end1"   : ["17:00", "17:00", "17:00", "17:00", "18:00"],
        "summer_start2" : ["16:00", "16:00", "", "", ""],
        "winter_start2" : ["16:00", "16:00", "", "", ""],
        "summer_end2"   : ["02:00", "02:00", "", "", ""],
        "winter_end2"   : ["02:00", "02:00", "", "", ""],
        "service_freq"  : [12, 24, 12, 24, 24],                         # weeks
        "service_dur"   : [3, 1, 3, 1, 1]                               # weeks
    })
    HEMS_ROTA.set_index("callsign", inplace=True)

    TIME_TYPES = ["mobile", "at scene", "leaving scene", "at hospital", "clear", "stand down"]

    def current_time():
        now = datetime.now()
        return now.strftime("%H:%M:%S")

    def date_time_of_call(start_dt: str, elapsed_time: int) -> list[int, int, str, int, pd.Timestamp]:
        """
        Calculate a range of time-based parameters given a specific date-time

        **Returns:**  
            list(  
                `dow`             : int  
                `current_hour`    : int  
                `weekday`         : str   
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
        
        return [dow, current_hour, weekday, current_month, current_dt]