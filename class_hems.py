
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
        self.flying_time = 0
        self.type = Utils.HEMS_ROTA["type"][Utils.HEMS_ROTA.index == self.callsign].iloc[0]
        # NOTE: HEMS_ROTA is indexed on callsign
        self.servicing_frequency_hours = 100
        self.servicing_duration_weeks = 4
        # This will need revising once we've settled on how to keep track of service start and end times
        self.service_start_date = 0
        self.service_end_date = 0

    def next_service(self):
        print(f"Next service for {self.callsign} is {self.service_start_date}")
        

    def update_flying_time(self, service_start_time):
        """
            Update flying hours of a HEMS resource

            This function will make it possible to update the flying hours of a given HEMS resource
            and if necessary, mark it as being serviced.

        """

        if self.type == "helicopter" and (self.flying_time > self.servicing_frequency_hours * 60):
            self.being_serviced = 1
            self.service_start_date = service_start_time
            self.service_end_date = service_start_time + (self.servicing_duration_weeks * 7 * 24 * 60)

            print(f'{service_start_time:0.2f} Callsign {self.callsign} flying time updated to {self.flying_time} and start is {self.service_start_date} and end {self.service_end_date}')


    def operational_after_service(self, current_time):
        """
            Update service status of HEMS' resources

            This function will periodically check to see whether HEMS' resources have
            now completed the service interval and can be returned to operational use

        """
        if self.type == "helicopter" and (current_time >= self.service_end_date):
            self.being_serviced = 0
            self.flying_time = 0

        