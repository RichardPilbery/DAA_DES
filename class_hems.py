
from utils import Utils
import pandas as pd
from class_ambulance import Ambulance

class HEMS(Ambulance):
    """
        # The HEMS class

        This class defines a HEMS resource

    
    """

    def __init__(self, callsign: str, current_date: str):
        # Inherit all parent class functions
        super().__init__(ambulance_type = "HEMS")

        self.callsign = callsign
        self.available = 1
        self.being_serviced = 0
        # NOTE: HEMS_ROTA is indexed on callsign
        self.servicing_frequency_weeks = Utils.HEMS_ROTA["service_freq"][Utils.HEMS_ROTA.index == self.callsign].iloc[0]
        self.servicing_duration_weeks = Utils.HEMS_ROTA["service_dur"][Utils.HEMS_ROTA.index == self.callsign].iloc[0]
        self.service_start_date = pd.to_datetime(current_date) + pd.Timedelta(self.servicing_frequency_weeks * 7, unit="days")
        self.service_end_date = self.service_start_date + pd.Timedelta(self.servicing_duration_weeks * 7, unit="days")

    def next_service(self):
        print(f"Next service for {self.callsign} is {self.service_start_date}")
        
        