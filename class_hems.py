
from utils import Utils
import pandas as pd
from class_ambulance import Ambulance

class HEMS(Ambulance):
    """
        # The HEMS class

        This class defines a HEMS resource


    """

    def __init__(self, callsign: str, resource_id=None):
        # Inherit all parent class functions
        super().__init__(ambulance_type = "HEMS")
        self.utilityClass = Utils()

        df = self.utilityClass.HEMS_ROTA[self.utilityClass.HEMS_ROTA.index == callsign]

        self.callsign = callsign
        self.callsign_group = df['callsign_group']
        self.available = 1
        self.being_serviced = False
        self.flying_time = 0
        self.vehicle_type = df['vehicle_type'].iloc[0]
        self.category = df['category'].iloc[0]
        self.summer_start = df["summer_start"].iloc[0]
        self.winter_start = df["winter_start"].iloc[0]
        self.summer_end = df["summer_end"].iloc[0]
        self.winter_end = df["winter_end"].iloc[0]

        # Pre-determine the servicing schedule when the resource is created
        self.servicing_schedule = pd.DataFrame(columns=['year', 'service_start_date', 'service_end_date'])

        self.in_use = False
        self.resource_id = resource_id


    def unavailable_due_to_service(self, current_dt: pd.Timestamp) -> bool:
        """
            Returns logical value denoting whether the HEMS resource is currently
            unavailable due to being serviced

        """

        curr_year_servicing_schedule = self.servicing_schedule[self.servicing_schedule['year'] == current_dt.year]

        if curr_year_servicing_schedule['service_start_date'] >= current_dt.date <= curr_year_servicing_schedule['service_end_date']:
            return True
        
        return False

    def hems_resource_on_shift(self, hour: int, season: int) -> bool:

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

        