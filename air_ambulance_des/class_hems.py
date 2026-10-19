import pandas as pd
from air_ambulance_des.utils import Utils
from air_ambulance_des.class_ambulance import Ambulance


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
        resource_id: str = None,
        summer_season: list[int] = None,
    ):
        # Inherit all parent class functions
        super().__init__(ambulance_type="HEMS")

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

        # Read in the summer months from a file
        summer_months = pd.read_csv("actual_data/rota_start_end_months.csv")
        summer_start_month = int(
            summer_months[summer_months["what"] == "summer_start_month"][
                "month"
            ].values[0]
        )
        summer_end_month = int(
            summer_months[summer_months["what"] == "summer_end_month"]["month"].values[
                0
            ]
        )

        self.summer_season = [
            x for x in range(summer_start_month, summer_end_month + 1)
        ]

        # Pre-determine the servicing schedule when the resource is created
        self.servicing_schedule = servicing_schedule

        self.in_use = False
        self.resource_id = resource_id

    def service_check(self, current_dt: pd.Timestamp, GDAAS_service: bool) -> bool:
        if self.registration == "g-daan":
            if GDAAS_service:
                self.callsign_group = 70
                self.callsign = "H70"

            else:
                self.callsign_group = 71
                self.callsign = "H71"
                self.in_use = (
                    False  # We might need to re-think this in 24/7 scenarios although
                )
                # presumably the callsign will not change during an incident, only after.

            # GDASS being serviced

            # print(f"reg {self.registration} now has callsign {self.callsign}")

        return self.unavailable_due_to_service(current_dt)

    def unavailable_due_to_service(self, current_dt: pd.Timestamp) -> bool:
        """
        Returns logical value denoting whether the HEMS resource is currently
        unavailable due to being serviced

        """

        for index, row in self.servicing_schedule.iterrows():
            # print(row)
            if row["service_start_date"] <= current_dt <= row["service_end_date"]:
                self.being_serviced = True
                return True

        self.being_serviced = False
        return False

    def hems_resource_on_shift(self, hour: int, month: int) -> bool:
        """
        Function to determine whether the HEMS resource is within
        its operational hours
        """

        # Assuming summer hours are quarters 2 and 3 i.e. April-September
        # Can be modified if required.
        # SR NOTE: If changing these, please also modify in
        # write_run_params() function in des_parallel_process
        start = self.summer_start if month in self.summer_season else self.winter_start
        end = self.summer_end if month in self.summer_season else self.winter_end

        return self.utilityClass.is_time_in_range(int(hour), int(start), int(end))
