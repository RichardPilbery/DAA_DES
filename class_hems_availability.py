from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Any, Generator
from class_patient import Patient
from utils import Utils
import pandas as pd
from class_hems import HEMS
from simpy import FilterStore, Interrupt, Event

class HEMSAvailability():
    """
        # The HEMS Availability class

        This class is a filter store which can provide HEMS resources
        based on the time of day and servicing schedule

    
    """

    def __init__(self, env, sim_start_date, sim_duration, servicing_overlap_allowed = False, servicing_buffer_weeks = 4, servicing_preferred_month = 1):
       
        self.env = env
        self.utilityClass = Utils()

        # Adding options to set servicing parameters.
        self.servicing_overlap_allowed = servicing_overlap_allowed
        self.serviing_buffer_weeks = servicing_buffer_weeks
        self.servicing_preferred_month = servicing_preferred_month
        self.sim_start_date = sim_start_date

        print(f"Sim start date {self.sim_start_date}")
        # For belts and braces, add an additional year to
        # calculate the service schedules since service dates can be walked back to the 
        # previous year
        self.sim_end_date = sim_start_date + timedelta(minutes=sim_duration + (1*365*24*60)) 

        # School holidays
        self.school_holidays = pd.read_csv('actual_data/school_holidays.csv')

        self.HEMS_resources_list = []

        # Create a store for HEMS resources
        self.store = FilterStore(env)

        # Prepare HEMS resources for ingesting into store
        self.prep_HEMS_resources()

        # Populate the store with HEMS resources
        self.populate_store()

    def daily_servicing_check(self, current_dt: datetime) -> None:
        """
            Function to iterate through the store and trigger the service check
            function in the HEMS class
        """
        h: HEMS
        for h in self.store.items:
            h.unavailable_due_to_service(current_dt)


    def prep_HEMS_resources(self) -> None:
        """
            This function ingests HEMS resource data from a user-supplied CSV file
            and populates a list of HEMS class objects. The key activity here is
            the calculation of service schedules for each HEMS object, taking into account a
            user-specified preferred month of servicing, service duration, and a buffer period 
            following a service to allow for over-runs and school holidays

        """

        schedule = []
        service_dates = []

        HEMS_ROTA = pd.read_csv('actual_data/HEMS_rota.csv')

        for index, row in HEMS_ROTA.iterrows():
            current_resource_service_dates = []
            # Check if service date provided
            if not pd.isna(row['last_service']):
                #print(f"Checking {row['callsign']} with previous service date of {row['last_service']}")
                last_service = datetime.strptime(row['last_service'], "%Y-%m-%d")
                service_date = last_service
                
                while last_service < self.sim_end_date:

                    end_date = last_service + timedelta(weeks = int(row['service_duration_weeks'])) + timedelta(weeks=self.serviing_buffer_weeks)

                    service_date, end_date = self.find_next_service_date(last_service, row["service_schedule_months"], service_dates, row['service_duration_weeks'])
                    
                    schedule.append((row['callsign'], service_date))
                    #print(service_date)
                    service_dates.append({'service_start_date': service_date, 'service_end_date': end_date})

                    current_resource_service_dates.append({'service_start_date': service_date, 'service_end_date': end_date})
                    #print(service_dates)
                    last_service = service_date

            # Create new HEMS resource and add to HEMS_resource_list
            #pd.DataFrame(columns=['year', 'service_start_date', 'service_end_date'])
            hems = HEMS(
                callsign            = row['callsign'],
                callsign_group      = row['callsign_group'],
                vehicle_type        = row['vehicle_type'],
                category            = row['category'],
                summer_start        = row['summer_start'],
                winter_start        = row['winter_start'],
                summer_end          = row['summer_end'],
                winter_end          = row['winter_end'],
                servicing_schedule  = pd.DataFrame(current_resource_service_dates),
                resource_id         = row['callsign']
            )

            self.HEMS_resources_list.append(hems)

        # Write servicing schedules to a file for use in calculations and resource visualisations
        (pd.DataFrame(schedule, columns=["resource", "service_start_date"])
        .merge(pd.DataFrame(service_dates))
        .to_csv("data/service_dates.csv"))


    def populate_store(self):
        """
            Function to populate the filestore with HEMS class objects
            contained in a class list
        """

        h: HEMS
        for h in self.HEMS_resources_list:
            print(f"Populating resource store: HEMS({h.callsign})")
            print(h.servicing_schedule)
            self.store.put(h)


    def add_hems(self):
        """
            Future function to allow for adding HEMS resources.
            We might not use this (we could just amend the HEMS_ROTA dataframe, for example)
            but might be useful for 'what if' simulations
        """
        pass


    def preferred_group_available(self, pt: Patient, preferred_group: int, preferred_vehicle_type: str) -> list[HEMS | None, int, bool]:
        """
            Check whether the preferred resource group is available. Returns a list with either the HEMS resource or None, 
            an indication as to whether the resource was available, or another resource in the same callsign_group, and
            the service status of the preferred resource.
        """

        # Initialise object HEMS as a placeholder object
        hems = HEMS
        # Initialise variable 'preferred' to False
        preferred = 0
        # Initialise variable 'service_status' to indicate whether resource is currently being serviced
        service_status = False

        # Iterates through items **available** in store at the time the function is called
        h: HEMS
        for h in self.store.items:

            # If callsign group is preferred group AND is preferred vehicle type, returns that item at that point
            # Rest of code will not be reached as the return statement will terminate this for loop as soon as
            # this condition is met
            # (i.e. IF the preferred callsign group and vehicle type is available, we only care about that -
            # so return)
            if not h.in_use and h.hems_resource_on_shift(pt.hour, pt.qtr) and not h.unavailable_due_to_service(pt.current_dt):

                if int(h.callsign_group) == int(preferred_group) and h.vehicle_type == preferred_vehicle_type:
                    # if h.being_serviced:
                    #     print(f"Preferred resource service status {h.being_serviced}")
                    service_status = h.being_serviced

                if int(h.callsign_group) == int(preferred_group) and h.vehicle_type == preferred_vehicle_type:
                    hems = h
                    preferred = 1
                    break

                # If it's the preferred group but not the preferred vehicle type, the variable
                # hems becomes the HEMS resource object that we are currently looking at in the store
                # so we will basically - in the event of not finding the exact resource we want - find
                # the next best thing from the callsign group
                # SR note 13/1/25 - double check this logic - as we would send a critical care car over
                # a different available helicopter if I'm interpreting this correctly. Just need to confirm
                # this was the order of priority agreed on.
                # RP note 13/02/2025 - Based on discussions with despatcher, sounds like closest free resource
                # tends to get sent. Option to add subsequent request/requirement for enhanced care, for example
                # Perhaps a TODO?

                elif h.callsign_group == preferred_group:
                    hems = h
                    preferred = 2

        # If we have not found an exact match for preferred callsign and vehicle type out of the
        # resources currently available in our store, we will then reach this code
        # If the preferred variable was set to True at any point, we will return HEMS
        # Note that this will be the last resource that met the condition h.callsign_group == preferred_group
        # which may be relevant if there is more than one other resource within that callsign group
        # (though this is not currently a situation that occurs within the naming conventions at DAAT)

        if preferred in [1, 2]:
            return [hems, preferred, service_status]
        else:
            return [None, preferred, service_status]


    def allocate_resource(self, pt: Patient) -> Any | Event:
        """
            Attempt to allocate a resource from the preferred group.
        """

        #print(f"Attempting to allocate resource with callsign group {pt.hems_pref_callsign_group} and preferred vehicle type {pt.hems_pref_vehicle_type}")

        # Pref Res will be either
        # - a HEMS resource object if the preferred callsign group+vehicle is available
        # - OR if some vehicle from the preferred callsign group is available even if the preferred vehicle is not
        # - OR None if neither of those conditions are met

        pref_res = self.preferred_group_available(
            pt=pt,
            preferred_group=pt.hems_pref_callsign_group,
            preferred_vehicle_type=pt.hems_pref_vehicle_type
        )

        resource_event: Event = self.env.event()

        def process(pres_res: list[HEMS | None, int, bool]) -> Generator[Any, Any, None]:

            def resource_filter(resource: HEMS, pref_res: list[HEMS | None, int, bool]) -> bool:
                """
                Checks whether the resource the incident wants is available in the
                simpy FilterStore
                Returns True if resource is
                - not in use
                - on shift
                - not being serviced
                Otherwise, returns False
                """
                #print(f"Resource filter with hour {hour} and qtr {qtr}")

                if pref_res[0] != None:
                   # print('Preferred resource is available')
                    if pref_res[1] == 1:
                        # Need to find preferred resource
                        return True if resource.callsign_group == pref_res[0].callsign_group else False
                    else:
                        # Need to find resource in preferred group
                        return True if resource.callsign_group == pref_res[0].callsign_group else False

                else:
                    #print('Preferred resource is NOT available')
                    # If the resource **is not currently in use** AND **is currently on shift** AND not being serviced
                    if not resource.in_use and resource.hems_resource_on_shift(pt.hour, pt.qtr) and not resource.unavailable_due_to_service(pt.current_dt):
                        return True
                    else:
                        return False

            with self.store.get(lambda hems_resource: resource_filter(hems_resource, pref_res)) as request:

                #print(request)
                
                resource = yield request | self.env.timeout(0.1)

                if request in resource:
                    #print(f"Allocating HEMS resource {resource[request].callsign} at time {self.env.now:.3f}")
                    resource.in_use = True
                    pt.hems_callsign_group = resource[request].callsign_group
                    pt.hems_vehicle_type = resource[request].vehicle_type

                    resource_event.succeed([resource[request], pref_res[1], pref_res[2]])
                else:
                    #print(f"No HEMS (helimed or ccc) resource available; using Non-DAAT land ambulance")
                    resource_event.succeed([None, pref_res[1], pref_res[2]])

        self.env.process(process(pref_res))
    
        return resource_event


    def return_resource(self, resource: HEMS) -> None:
        """
            Class to return HEMS class object back to the filestore
        """
        resource.in_use = False
        self.store.put(resource)


    def years_between(self, start_date: datetime, end_date: datetime) -> list[int]:
        """
            Function to return a list of years between given start and end date
        """
        return list(range(start_date.year, end_date.year + 1))


    def do_ranges_overlap(self, start1: datetime, end1: datetime, start2: datetime, end2: datetime) -> bool:
        """
            Function to determine whether two sets of datetimes overlap
        """
        return max(start1, start2) <= min(end1, end2)


    def is_during_school_holidays(self, start_date: datetime, end_date: datetime) -> bool:
        """
            Function to calculate whether given start and end date time period falls within
            a school holiday
        """

        for index, row in self.school_holidays.iterrows():
            
            if self.do_ranges_overlap(pd.to_datetime(row['start_date']), pd.to_datetime(row['end_date']), start_date, end_date):
                return True

        return False


    def is_other_resource_being_serviced(self, start_date, end_date, service_dates):
        """
            Function to determine whether any resource is being services between a
            given start and end date period.
        """

        for sd in service_dates:
            if self.do_ranges_overlap(sd['service_start_date'], sd['service_end_date'], start_date, end_date):
                return True

        return False


    def find_next_service_date(self, last_service_date: datetime, interval_months: int, service_dates: list, service_duration: int) -> list[datetime]:
        """
            Function to determine the next service date for a resource. The date is determine by
            the servicing schedule for the resource, the preferred month of servicing, and to 
            avoid dates that fall in either school holidays or when other resources are being serviced.
        """

        next_due_date = last_service_date + relativedelta(months = interval_months) # Approximate month length
        end_date = next_due_date + timedelta(weeks = service_duration) 

        preferred_date = datetime(next_due_date.year, self.servicing_preferred_month , 2)
        preferred_end_date = preferred_date + timedelta(weeks = service_duration) 

        if next_due_date.month > preferred_date.month:
            preferred_date += relativedelta(years = 1)

        # print(f"Next due: {next_due_date} with end date {end_date} and preferred_date is {preferred_date} with pref end {preferred_end_date}")
        
        # If preferred date is valid, use it
        if preferred_date <= next_due_date and not self.is_during_school_holidays(preferred_date, preferred_end_date):
            next_due_date = preferred_date
        
        while True:
            if self.is_during_school_holidays(next_due_date, end_date) or self.is_other_resource_being_serviced(next_due_date, end_date, service_dates):    
                next_due_date -= timedelta(days = 1)
                end_date = next_due_date + timedelta(weeks = service_duration)
                continue
            else:
                break
            
        return [next_due_date, end_date]
