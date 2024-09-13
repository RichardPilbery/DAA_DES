
from utils import Utils
import pandas as pd

class HEMS:

    def __init__(self, callsign: str, current_date: str):

        self.callsign = callsign
        self.current_date = current_date
        self.available = 1
        self.being_serviced = 0
        self.servicing_frequency_weeks = Utils.HEMS_ROTA["service_freq"][Utils.HEMS_ROTA["callsign"] == self.callsign]
        self.servicing_duration_weeks = Utils.HEMS_ROTA["service_dur"][Utils.HEMS_ROTA["callsign"] == self.callsign]
        self.service_start_date = pd.to_datetime(current_date) + pd.Timedelta(self.servicing_frequency_weeks * 7, unit="days")
        self.service_end_date = self.service_start_date + pd.Timedelta(self.servicing_duration_weeks * 7, unit="days")

    def next_service(self):
        print(f"Next service for {self.callsign} is {self.service_start_date}")
        
        