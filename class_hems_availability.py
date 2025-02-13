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
        self.sim_end_date = sim_start_date + timedelta(minutes=sim_duration)

        # School holidays
        self.school_holidays = pd.read_csv('actual_data/school_holidays.csv')

        self.HEMS_resources_list = []

        # Create a store for HEMS resources
        self.store = FilterStore(env)

        # Prepare HEMS resources for ingesting into store
        self.prep_HEMS_resources()

        # Populate the store with HEMS resources
        self.populate_store()

    def daily_servicing_check(self, current_dt: datetime):
        for h in self.store.items:
            h.unavailable_due_to_service(current_dt)


    def prep_HEMS_resources(self):

        schedule = []
        service_dates = []

        HEMS_ROTA = pd.read_csv('actual_data/HEMS_rota.csv')

        for index, row in HEMS_ROTA.iterrows():
            # Check if service date provided
            if not pd.isna(row['last_service']):
                print(f"Checking {row['callsign']} with previous service date of {row['last_service']}")
                last_service = datetime.strptime(row['last_service'], "%Y-%m-%d")
                service_date = last_service
                current_resource_service_dates = []

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

                print(self.HEMS_resources_list)

    def populate_store(self):
        for h in self.HEMS_resources_list:
            print(f"Populating resource store: HEMS({h.callsign})")
            self.store.put(h)

    def add_hems(self):
        """
            Future function to allow for adding HEMS resources.
            We might not use this (we could just amend the HEMS_ROTA dataframe, for example)
            but might be useful for 'what if' simulations
        """
        pass


    def preferred_group_available(self, preferred_group: int, preferred_vehicle_type: str) -> HEMS | None:
        """
            Check whether the preferred resource group is available and respond accordingly
        """

        # Initialise object HEMS as a placeholder object
        hems = HEMS
        # Initialise variable 'preferred' to False
        preferred = False

        # Iterates through items **available** in store at the time the function is called
        h: HEMS
        for h in self.store.items:

            # If callsign group is preferred group AND is preferred vehicle type, returns that item at that point
            # Rest of code will not be reached as the return statement will terminate this for loop as soon as
            # this condition is met
            # (i.e. IF the preferred callsign group and vehicle type is available, we only care about that -
            # so return)
            if int(h.callsign_group) == int(preferred_group) and h.vehicle_type == preferred_vehicle_type and not h.being_serviced:
                return h

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
                preferred = True

        # If we have not found an exact match for preferred callsign and vehicle type out of the
        # resources currently available in our store, we will then reach this code
        # If the preferred variable was set to True at any point, we will refer HEMS
        # Note that this will be the last resource that met the condition h.callsign_group == preferred_group
        # which may be relevant if there is more than one other resource within that callsign group
        # (though this is not currently a situation that occurs within the naming conventions at DAAT)

        if preferred:
            return hems
        else:
            return None

    def allocate_resource(self, pt: Patient) -> Any | Event:
        """
            Attempt to allocate a resource from the preferred group.
        """

        print(f"Attempting to allocate resource with callsign group {pt.hems_pref_callsign_group} and preferred vehicle type {pt.hems_pref_vehicle_type}")

        # Pref Res will be either
        # - a HEMS resource object if the preferred callsign group+vehicle is available
        # - OR if some vehicle from the preferred callsign group is available even if the preferred vehicle is not
        # - OR None if neither of those conditions are met

        pref_res = self.preferred_group_available(
            preferred_group=pt.hems_pref_callsign_group,
            preferred_vehicle_type=pt.hems_pref_vehicle_type
        )

        resource_event: Event = self.env.event()

        def process() -> Generator[Any, Any, None]:
            def resource_filter(resource: HEMS, pref_res: HEMS) -> bool:
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

                # If the resource **is not currently in use** AND **is currently on shift** AND not being serviced
                if not resource.in_use and resource.hems_resource_on_shift(pt.hour, pt.qtr) and not resource.unavailable_due_to_service(pt.current_dt):
                    # Check whether the resource is the preferred resource
                    if pref_res != None:
                        if resource.callsign == pref_res.callsign:
                            print(f"Preferred resource {pref_res.callsign} available")
                            return True
                    else:
                        print("Other (non-preferred) resource available")
                        return True
                else:
                    # If neither of these are available, return 'False'
                    # print("Neither preferred or non-preferred resource available")
                    return False


            with self.store.get(lambda hems_resource: resource_filter(hems_resource, pref_res)) as request:
                resource = yield request | self.env.timeout(0.1)

                if request in resource:
                    print(f"Allocating HEMS resource at time {self.env.now:.3f}")
                    resource.in_use = True
                    pt.hems_callsign_group = resource[request].callsign_group
                    pt.hems_vehicle_type = resource[request].vehicle_type

                    resource_event.succeed(resource[request])
                else:
                    print(f"No HEMS (helimed or ccc) resource available; using Non-DAAT land ambulance")
                    resource_event.succeed()

        self.env.process(process())
    
        return resource_event

    def return_resource(self, resource):
        resource.in_use = False
        self.store.put(resource)

    def years_between(self, start_date, end_date):
        return list(range(start_date.year, end_date.year + 1))
    
    def do_ranges_overlap(self, start1: datetime, end1: datetime, start2: datetime, end2: datetime) -> bool:
        return max(start1, start2) <= min(end1, end2)
    
    def is_during_school_holidays(self, start_date, end_date):

        for index, row in self.school_holidays.iterrows():
            
            #if pd.to_datetime(row['start_date']) <= start_date <= pd.to_datetime(row['end_date']):
            if self.do_ranges_overlap(pd.to_datetime(row['start_date']), pd.to_datetime(row['end_date']), start_date, end_date):
                return True

        return False

    def is_other_resource_being_serviced(self, start_date, end_date, service_dates):

        for sd in service_dates:
            #if pd.to_datetime(row['start_date']) <= start_date <= pd.to_datetime(row['end_date']):
            if self.do_ranges_overlap(sd['service_start_date'], sd['service_end_date'], start_date, end_date):
                return True

        return False

    def find_next_service_date(self, last_service_date, interval_months, service_dates, service_duration):

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
