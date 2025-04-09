from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Any, Generator
from class_patient import Patient
from utils import Utils
import pandas as pd
from class_hems import HEMS
from simpy import FilterStore, Event

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

        self.serviceStore = FilterStore(env)

        # Prepare HEMS resources for ingesting into store
        self.prep_HEMS_resources()

        # Populate the store with HEMS resources
        self.populate_store()

        # Daily servicing check (in case sim starts during a service)
        [dow, hod, weekday, month, qtr, current_dt] = self.utilityClass.date_time_of_call(self.sim_start_date, self.env.now)
        self.daily_servicing_check(current_dt)

    def daily_servicing_check(self, current_dt: datetime) -> None:
        """
            Function to iterate through the store and trigger the service check
            function in the HEMS class
        """
        h: HEMS

        GDAAS_service = False
        for h in self.store.items:
            
            if h.registration == 'g-daas':
                
                GDAAS_service = h.unavailable_due_to_service(current_dt)

        for h in self.store.items:
            
            if h.service_check(current_dt, GDAAS_service):
                # Vehicle being serviced
                
                if h in self.store.items:
                    print("****************")
                    print(f"{h.callsign} being serviced so remove from store")
                    service_h = yield self.store.get(lambda item: item == h)
                    print(service_h)
                    yield self.serviceStore.put(service_h)
                    print(self.serviceStore.items)
                    print("***********")

        s: HEMS
        if len(self.serviceStore.items) > 0:
            print("Service store has items")
            for s in list(self.serviceStore.items):
                if s.service_check(current_dt, GDAAS_service):
                    h = yield self.serviceStore.get(lambda item: item == s)
                    yield self.store.put(h)


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


        # Calculate service schedules for each resource

        SERVICING_SCHEDULE = pd.read_csv('actual_data/service_schedules_by_model.csv')

        for index, row in SERVICING_SCHEDULE.iterrows():
            #print(row)
            current_resource_service_dates = []
            # Check if service date provided
            if not pd.isna(row['last_service']):
                #print(f"Checking {row['registration']} with previous service date of {row['last_service']}")
                last_service = datetime.strptime(row['last_service'], "%Y-%m-%d")
                service_date = last_service

                while last_service < self.sim_end_date:

                    end_date = last_service + timedelta(weeks = int(row['service_duration_weeks'])) + timedelta(weeks=self.serviing_buffer_weeks)

                    service_date, end_date = self.find_next_service_date(last_service, row["service_schedule_months"], service_dates, row['service_duration_weeks'])

                    schedule.append((row['registration'], service_date, end_date))
                    #print(service_date)
                    service_dates.append({'service_start_date': service_date, 'service_end_date': end_date})

                    current_resource_service_dates.append({'service_start_date': service_date, 'service_end_date': end_date})
                    #print(service_dates)
                    #print(current_resource_service_dates)
                    last_service = service_date
            else:
                schedule.append((row['registration'], None, None))
                #print(schedule)

            service_df = pd.DataFrame(schedule, columns=["registration", "service_start_date", "service_end_date"])

            service_df.to_csv("data/service_dates.csv", index=False)
               
        # Append those to the HEMS resource.

        HEMS_RESOURCES = (
            pd.read_csv("actual_data/HEMS_ROTA.csv")
                .merge(
                    SERVICING_SCHEDULE
                        .merge(
                            pd.read_csv("actual_data/callsign_registration_lookup.csv"),
                            on="registration",
                            how="left"
                        ),
                    on="callsign",
                    how="left"
                )
        )

        for index, row in HEMS_RESOURCES.iterrows():

            s = service_df[service_df['registration'] == row['registration']]
            #print(s)

            # Create new HEMS resource and add to HEMS_resource_list
            #pd.DataFrame(columns=['year', 'service_start_date', 'service_end_date'])
            hems = HEMS(
                callsign            = row['callsign'],
                callsign_group      = row['callsign_group'],
                vehicle_type        = row['vehicle_type'],
                category            = row['category'],
                registration        = row['registration'],
                summer_start        = row['summer_start'],
                winter_start        = row['winter_start'],
                summer_end          = row['summer_end'],
                winter_end          = row['winter_end'],
                servicing_schedule  = s,
                resource_id         = row['registration']
            )

            self.HEMS_resources_list.append(hems)

        #print(self.HEMS_resources_list)


    def populate_store(self):
        """
            Function to populate the filestore with HEMS class objects
            contained in a class list
        """

        h: HEMS
        for h in self.HEMS_resources_list:
            #print(f"Populating resource store: HEMS({h.callsign})")
            #print(h.servicing_schedule)
            self.store.put(h)


    def add_hems(self):
        """
            Future function to allow for adding HEMS resources.
            We might not use this (we could just amend the HEMS_ROTA dataframe, for example)
            but might be useful for 'what if' simulations
        """
        pass

    def resource_allocation_lookup(self, prefered_lookup: int):
        """
            Function to return description of lookup allocation choice

        """
        
        lookup_list = [
            "No HEMS resource available",
            "Preferred HEMS care category and vehicle type match",
            "Preferred HEMS care category match but not vehicle type",
            "HEMS CC case EC helicopter available",
            "HEMS CC case EC car available",
            "HEMS EC case CC helicopter available",
            "HEMS EC case CC car available",
            "HEMS helicopter case EC helicopter available",
            "HEMS helicopter case CC helicopter available",
            "HEMS REG case no helicopter benefit preferred group and vehicle type allocated",
	        "HEMS REG case no helicopter benefit preferred group allocated",
            "No HEMS resource available (pref vehicle type = 'Other')",
            "HEMS REG case no helicopter benefit first free resource allocated"
        ]

        return lookup_list[prefered_lookup]

    def current_store_status(self, pt: Patient) -> list[str]:
            """
                Debugging function to return current state of store
            """

            current_store_items = []

            h: HEMS
            for h in self.store.items:
                current_store_items.append(f"{h.callsign} ({h.category} online: {h.hems_resource_on_shift(pt.hour, pt.qtr)} {h.registration})")

            return current_store_items


    def preferred_resource_available(self, pt: Patient) -> list[HEMS | None, str]:
        """
            Check whether the preferred resource group is available. Returns a list with either the HEMS resource or None, and
            an indication as to whether the resource was available, or another resource in an established hierachy of
            availability can be allocated
        """

        # Initialise object HEMS as a placeholder object
        hems = HEMS
        # Initialise variable 'preferred' to False
        preferred = 999 # This will be used to ensure that the most desireable resource
                    # is allocated given that multiple matches may be found
        preferred_lookup = 0 # This will be used to code the resource allocation choice

        preferred_care_category = pt.hems_cc_or_ec

        #print(f"EC/CC resource with {preferred_care_category} and hour {pt.hour} and qtr {pt.qtr}")

        h: HEMS
        for h in self.store.items:

            # There is a hierachy of calls:
            # CC = H70 helicopter then car, then H71 helicopter then car then CC72
            # EC = H71 helicopter then car, then CC72, then H70 helicopter then car
            # If no resources then return None

            if not h.in_use and h.hems_resource_on_shift(pt.hour, pt.qtr):

                if h.category == preferred_care_category and h.vehicle_type == "helicopter" and not h.being_serviced:
                    hems = h
                    preferred = 1 # Top choice
                    preferred_lookup = 1
                    return [hems,  self.resource_allocation_lookup(preferred_lookup)]

                elif h.category == preferred_care_category and not h.being_serviced:
                    hems = h
                    preferred = 2 # Second choice (correct care category, but not vehicle type)
                    preferred_lookup = 2

                elif preferred_care_category == 'CC':
                    if h.vehicle_type == 'helicopter' and h.category == 'EC' and not h.being_serviced:
                        if preferred > 3:
                            hems = h
                            preferred = 3 # Third  choice (EC helicopter)
                            preferred_lookup = 3

                    elif h.category == 'EC' and not h.being_serviced:
                        if preferred > 4:
                            hems = h
                            preferred = 4
                            preferred_lookup = 4

                elif preferred_care_category == 'EC':
                    if h.vehicle_type == 'helicopter' and h.category == 'CC' and not h.being_serviced:
                        if preferred > 3:
                            hems = h
                            preferred = 3 # CC helicopter available
                            preferred_lookup = 5

                    elif h.category == 'CC' and not h.being_serviced:
                        hems = h
                        preferred = 4
                        preferred_lookup = 6

        print(f"preferred lookup {preferred_lookup} and preferred = {preferred}")

        if preferred_lookup != 999:
            return [hems, self.resource_allocation_lookup(preferred_lookup)]
        else:
            return [None, self.resource_allocation_lookup(0)]


    def allocate_resource(self, pt: Patient) -> Any | Event:
        """
            Attempt to allocate a resource from the preferred group.
        """

        # Pref Res will be either
        # - a HEMS resource object if one can be allocated otherwise None
        # - an indicator variable about the why the resource was chosen

        #print(f"Allocating resource for {pt.id} and care cat {pt.hems_cc_or_ec}")


        resource_event: Event = self.env.event()

        def process() -> Generator[Any, Any, None]:

            print(f"Allocating resource for {pt.id} and care cat {pt.hems_cc_or_ec}")

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
            #print(pref_res)
            pref_res: list[HEMS | None, str] = self.preferred_resource_available(pt)

            if pref_res[0] == None:

                return resource_event.succeed([None, pref_res[1], None])

            else:
                
                print(self.current_store_status(pt))

                with self.store.get(lambda hems_resource: hems_resource == pref_res[0]) as primary_resource_member:

                # Retrieve other group resource is there is one and make it unavailable.

                    resource = yield primary_resource_member | self.env.timeout(0.1)

                    if primary_resource_member in resource:
                        print(f"Allocating HEMS resource {resource[primary_resource_member].callsign} at time {self.env.now:.3f}")
                        resource.in_use = True
                        pt.hems_callsign_group = resource[primary_resource_member].callsign_group
                        pt.hems_vehicle_type = resource[primary_resource_member].vehicle_type
                        pt.hems_category = resource[primary_resource_member].category

                        # Also need to check if there is another vehicle in the group and make that unavailable

                        with self.store.get(lambda hems_resource2: 
                                            hems_resource2.callsign_group == pt.hems_callsign_group and 
                                            hems_resource2.category == pt.hems_category and
                                            hems_resource2.hems_resource_on_shift(pt.hour, pt.qtr)
                                        ) as secondary_callsign_group_member:

                            resource2 = yield secondary_callsign_group_member | self.env.timeout(0.1)

                            # We need to either return the second resource in the callsign_group or None
                            # back to the main model so that we can return it with the primary allocated
                            # resource at the end of the patient episode.

                            return_resource2_value = None

                            if secondary_callsign_group_member in resource2:
                                print(f"Secondary callsign group resource being allocated {resource2[secondary_callsign_group_member].callsign}")
                                print('------------------')
                                resource2.in_use = True
                                return_resource2_value = resource2[secondary_callsign_group_member]

                        resource_event.succeed([resource[primary_resource_member], pref_res[1], return_resource2_value])

        self.env.process(process())

        return resource_event


    def return_resource(self, resource: HEMS, secondary_resource: HEMS|None) -> None:
        """
            Class to return HEMS class object back to the filestore
        """

        
        resource.in_use = False
        self.store.put(resource)

        if secondary_resource != None:
            #print(f"Primary resource {resource.callsign} and Returning secondary resource {secondary_resource.callsign}")
            secondary_resource.in_use = False
            self.store.put(secondary_resource)

        #     print(f"Retruning primary: {resource.callsign} and secondary {secondary_resource.callsign}")
        # else:
        #     print(f"Retruning primary: {resource.callsign} and secondary NONE")


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

    def preferred_regular_group_available(self, pt: Patient) -> list[HEMS | None, int, bool]:
        """
            Check whether the preferred resource group is available. Returns a list with either the HEMS resource or None,
            an indication as to whether the resource was available, or another resource in the same callsign_group, and
            the service status of the preferred resource.
        """

        # if preferred_group == 70:
        #     print(f"Preferred group is {preferred_group} and vehicle_type {preferred_vehicle_type}")

        current_store_items = []

        # Initialise object HEMS as a placeholder object
        hems = HEMS

        preferred = 999 # This will be used to ensure that the most desireable resource
                    # is allocated given that multiple matches may be found
        preferred_lookup = 0 # This will be used to code the resource allocation choice

        preferred_group = pt.hems_pref_callsign_group,
        preferred_vehicle_type = pt.hems_pref_vehicle_type

        helicopter_benefit = pt.hems_helicopter_benefit

        # Iterates through items **available** in store at the time the function is called
        h: HEMS
        for h in self.store.items:

            current_store_items.append(h.callsign)

            # If callsign group is preferred group AND is preferred vehicle type, returns that item at that point
            # Rest of code will not be reached as the return statement will terminate this for loop as soon as
            # this condition is met
            # (i.e. IF the preferred callsign group and vehicle type is available, we only care about that -
            # so return)

            # if preferred_group == 70:
            #     print(f"{pt.current_dt} Preferred Resource status check: {h.callsign} in_use: {h.in_use} on_shift: {h.hems_resource_on_shift(pt.hour, pt.qtr)} and service {h.unavailable_due_to_service(pt.current_dt)}")

            if not h.in_use and h.hems_resource_on_shift(pt.hour, pt.qtr):

                if helicopter_benefit == 'y':
                    if h.vehicle_type == 'helicopter' and h.category == 'EC' and not h.being_serviced:
                        hems = h
                        preferred = 1
                        preferred_lookup = 7
                        return [hems, self.resource_allocation_lookup(preferred_lookup)]

                    elif h.vehicle_type == 'helicopter' and not h.being_serviced:
                        hems = h
                        preferred = 2
                        preferred_lookup = 8
                else:
                    # No helicopter benefit

                    if h.callsign_group == preferred_group and h.vehicle_type == preferred_vehicle_type and not h.being_serviced:
                        hems = h
                        preferred = 3
                        preferred_lookup = 9
                        print("REG job with no helicopter benefit and found preferred group and vehicle type")
                        return [hems, self.resource_allocation_lookup(preferred_lookup)]

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

                    elif h.callsign_group == preferred_group and not h.being_serviced:
                        if preferred > 4:
                            hems = h
                            preferred = 4
                            preferred_lookup = 10

                    elif not h.being_serviced:
                        if preferred > 5:
                            hems = h
                            preferred = 5
                            preferred_lookup = 12

        # If we have not found an exact match for preferred callsign and vehicle type out of the
        # resources currently available in our store, we will then reach this code
        # If the preferred variable was set to True at any point, we will return HEMS
        # Note that this will be the last resource that met the condition h.callsign_group == preferred_group
        # which may be relevant if there is more than one other resource within that callsign group
        # (though this is not currently a situation that occurs within the naming conventions at DAAT)

        #print(f"EC/CC Current store {current_store_items}")

        if preferred != 999:
            return [hems,  self.resource_allocation_lookup(preferred_lookup)]
        else:
            return [None, self.resource_allocation_lookup(0)]


    def allocate_regular_resource(self, pt: Patient) -> Any | Event:
        """
            Attempt to allocate a resource from the preferred group.
        """

        #print(f"Attempting to allocate resource with callsign group {pt.hems_pref_callsign_group} and preferred vehicle type {pt.hems_pref_vehicle_type}")

        # Pref Res will be either
        # - a HEMS resource object if the preferred callsign group+vehicle is available
        # - OR if some vehicle from the preferred callsign group is available even if the preferred vehicle is not
        # - OR None if neither of those conditions are met

        resource_event: Event = self.env.event()

        def process() -> Generator[Any, Any, None]:

            #print(f"Allocating resource for {pt.id} and care cat {pt.hems_cc_or_ec}")

            if pt.hems_pref_vehicle_type == "Other":
                # These are missed cases
                pref_res = [None,  self.resource_allocation_lookup(11)]
            else:
                pref_res = self.preferred_regular_group_available(pt)

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
                   # print('A resource is available')
                    if pref_res[1] in [ self.resource_allocation_lookup(0),  self.resource_allocation_lookup(11)]:
                        # Need to find preferred resource
                        return True if (resource.callsign_group == pref_res[0].callsign_group) and (resource.vehicle_type == pref_res[0].vehicle_type) else False
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
                    
            print(self.current_store_status(pt))

            with self.store.get(lambda hems_resource: resource_filter(hems_resource, pref_res)) as primary_callsign_group_member:

                # Retrieve other group resource is there is one and make it unavailable.

                resource = yield primary_callsign_group_member | self.env.timeout(0.1)

                if primary_callsign_group_member in resource:
                    print(f"{pt.id} Allocating HEMS resource {resource[primary_callsign_group_member].callsign} cat: {resource[primary_callsign_group_member].category} at time {pt.current_dt}")
                    resource.in_use = True
                    pt.hems_callsign_group = resource[primary_callsign_group_member].callsign_group
                    pt.hems_vehicle_type = resource[primary_callsign_group_member].vehicle_type
                    pt.hems_category = resource[primary_callsign_group_member].category

                    # Also need to check if there is another vehicle in the group and make that unavailable
                    with self.store.get(lambda hems_resource2: 
                                        hems_resource2.callsign_group == pt.hems_callsign_group and 
                                        hems_resource2.category == pt.hems_category and
                                        hems_resource2.hems_resource_on_shift(pt.hour, pt.qtr)
                                    ) as secondary_callsign_group_member:

                        resource2 = yield secondary_callsign_group_member | self.env.timeout(0.1)

                        # We need to either return the second resource in the callsign_group or None
                        # back to the main model so that we can return it with the primary allocated
                        # resource at the end of the patient episode.

                        return_resource2_value = None

                        if secondary_callsign_group_member in resource2:
                            #print('Secondary callsign group resource being allocated')
                            print(f"Secondary resource is {resource2[secondary_callsign_group_member].callsign} and cat: {resource2[secondary_callsign_group_member].category}")
                            resource2.in_use = True
                            return_resource2_value = resource2[secondary_callsign_group_member]


                    resource_event.succeed([resource[primary_callsign_group_member], pref_res[1], return_resource2_value])
                else:
                    print(f"No HEMS (helimed or ccc) resource available; using Non-DAAT land ambulance")
                    resource_event.succeed([None, pref_res[1], None])

        print('----------------------')

        self.env.process(process())


        return resource_event