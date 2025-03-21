
from utils import Utils
import pandas as pd
from class_ambulance import Ambulance

class HEMS(Ambulance):
    """
        # The HEMS class

        This class defines a HEMS resource


    """

    def __init__(
            self, 
            callsign: str, 
            callsign_group: str, 
            vehicle_type: str, 
            category: str, 
            registration: str,
            summer_start: str,
            winter_start: str,
            summer_end: str,
            winter_end: str,
            servicing_schedule: pd.DataFrame, 
            resource_id: str = None
        ):
        # Inherit all parent class functions
        super().__init__(ambulance_type = "HEMS")

        self.utilityClass = Utils()

        self.callsign = callsign
        self.callsign_group = callsign_group
        self.available = 1
        self.being_serviced = False
        self.flying_time = 0
        self.vehicle_type = vehicle_type
        self.category = category
        self.registration = registration
        self.summer_start = summer_start
        self.winter_start = winter_start
        self.summer_end = summer_end
        self.winter_end = winter_end

        # Pre-determine the servicing schedule when the resource is created
        self.servicing_schedule = servicing_schedule

        self.in_use = False
        self.resource_id = resource_id


    def unavailable_due_to_service(self, current_dt: pd.Timestamp) -> bool:
        """
            Returns logical value denoting whether the HEMS resource is currently
            unavailable due to being serviced

        """

        for index, row in self.servicing_schedule.iterrows():
            #print(row)
            if row['service_start_date'] <= current_dt <= row['service_end_date']:
                self.being_serviced = True
                return True

        self.being_serviced = False
        return False

    def hems_resource_on_shift(self, hour: int, season: int) -> bool:

        """
            Function to determine whether the HEMS resource is within
            its operational hours
        """
        
        # Assuming summer hours are quarters 2 and 3 i.e. April-September
        # Can be modified if required.
        # SR NOTE: If changing these, please also modify in
        # write_run_params() function in des_parallel_process
        start = self.summer_start if season in [2, 3] else self.winter_start
        end = self.summer_end if season in [2, 3] else self.winter_end

        return self.utilityClass.is_time_in_range(int(hour), int(start), int(end))

        