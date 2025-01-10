
from utils import Utils
import pandas as pd
from class_ambulance import Ambulance

class HEMS(Ambulance):
    """
        # The HEMS class

        This class defines a HEMS resource

    
    """

    def __init__(self, callsign: str):
        # Inherit all parent class functions
        super().__init__(ambulance_type = "HEMS")
        self.utilityClass = Utils()

        df = self.utilityClass.HEMS_ROTA[self.utilityClass.HEMS_ROTA.index == callsign]

        self.callsign = callsign
        self.callsign_group = df['callsign_group']
        self.available = 1
        self.being_serviced = 0
        self.flying_time = 0
        self.vehicle_type = df['vehicle_type'].iloc[0]
        self.category = df['category'].iloc[0]
        self.summer_start = df["summer_start"].iloc[0]
        self.winter_start = df["winter_start"].iloc[0]
        self.summer_end = df["summer_end"].iloc[0]
        self.winter_end = df["winter_end"].iloc[0]

        # NOTE: HEMS_ROTA is indexed on callsign
        self.servicing_frequency_hours = 100
        self.servicing_duration_weeks = 4
        # This will need revising once we've settled on how to keep track of service start and end times
        self.service_start_date = 0
        self.service_end_date = 0

        self.in_use = False
        
        
    def operational_after_service(self, current_time):
        """
            Update service status of HEMS' resources

            This function will periodically check to see whether HEMS' resources have
            now completed the service interval and can be returned to operational use

            CURRENTLY NOT IN USE

        """
        if self.vehicle_type == "helicopter" and (current_time >= self.service_end_date):
            self.being_serviced = 0
            self.flying_time = 0

    def hems_resource_on_shift(self, hour: int, season: int):

        """
            Function to determine whether the HEMS resource is within
            its operational hours
        """

        #print(f"on shift callsign {self.callsign}, hour {hour}, season {season} with summer_start {self.summer_start} and winter_start = {self.winter_start}")
        
        # Assuming summer hours are quarters 2 and 3 i.e. April-September
        # Can be modified if required.
        start = self.summer_start if season in [2, 3] else self.winter_start
        end = self.summer_end if season in [2, 3] else self.winter_end

        #print(f"Start is {start} and end is {end} and current hour is {hour}")

        #print(f"Is time in range: {self.utilityClass.is_time_in_range(int(hour), int(start), int(end))}")

        return self.utilityClass.is_time_in_range(int(hour), int(start), int(end))

        