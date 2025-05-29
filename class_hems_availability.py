from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Any, Generator
from class_patient import Patient
from utils import Utils
import pandas as pd
from class_hems import HEMS
from simpy import FilterStore, Event
from numpy.random import SeedSequence

import logging
from enum import IntEnum

class ResourceAllocationReason(IntEnum):
    NONE_AVAILABLE = 0
    MATCH_PREFERRED_CARE_CAT_HELI = 1
    MATCH_PREFERRED_CARE_CAT_CAR = 2
    CC_MATCH_EC_HELI = 3
    CC_MATCH_EC_CAR = 4
    EC_MATCH_CC_HELI = 5
    EC_MATCH_CC_CAR = 6
    REG_HELI_BENEFIT_MATCH_EC_HELI = 7
    REG_HELI_BENEFIT_MATCH_CC_HELI = 8
    REG_NO_HELI_BENEFIT_GROUP_AND_VEHICLE = 9
    REG_NO_HELI_BENEFIT_GROUP = 10
    OTHER_VEHICLE_TYPE = 11
    REG_NO_HELI_BENEFIT_ANY = 12
    REG_NO_HELI_BENEFIT_VEHICLE = 13

class HEMSAvailability():
    """
        # The HEMS Availability class

        This class is a filter store which can provide HEMS resources
        based on the time of day and servicing schedule


    """

    def __init__(self, env, sim_start_date, sim_duration, utility: Utils, servicing_overlap_allowed = False,
                 servicing_buffer_weeks = 4, servicing_preferred_month = 1,
                 print_debug_messages = False, master_seed=SeedSequence(42)):

        self.LOOKUP_LIST = [
            "No HEMS resource available", #0
            "Preferred HEMS care category and vehicle type match", # 1
            "Preferred HEMS care category match but not vehicle type", # 2
            "HEMS CC case EC helicopter available", # 3
            "HEMS CC case EC car available", # 4
            "HEMS EC case CC helicopter available", # 5
            "HEMS EC case CC car available", # 6
            "HEMS REG helicopter case EC helicopter available",
            "HEMS REG helicopter case CC helicopter available",
            "HEMS REG case no helicopter benefit preferred group and vehicle type allocated", # 9
            "HEMS REG case no helicopter benefit preferred group allocated", # 10
            "No HEMS resource available (pref vehicle type = 'Other')", # 11
            "HEMS REG case no helicopter benefit first free resource allocated", # 12
            "HEMS REG case no helicopter benefit free helicopter allocated" #13
        ]

        self.env = env
        self.print_debug_messages = print_debug_messages
        self.master_seed = master_seed
        self.utilityClass = utility

        # Adding options to set servicing parameters.
        self.servicing_overlap_allowed = servicing_overlap_allowed
        self.serviing_buffer_weeks = servicing_buffer_weeks
        self.servicing_preferred_month = servicing_preferred_month
        self.sim_start_date = sim_start_date

        self.debug(f"Sim start date {self.sim_start_date}")
        # For belts and braces, add an additional year to
        # calculate the service schedules since service dates can be walked back to the
        # previous year
        self.sim_end_date = sim_start_date + timedelta(minutes=sim_duration + (1*365*24*60))

        # School holidays
        self.school_holidays = pd.read_csv('actual_data/school_holidays.csv')

        self.HEMS_resources_list = []

        self.active_callsign_groups = set()     # Prevents same crew being used twice...hopefully...
        self.active_registrations = set()       # Prevents same vehicle being used twice
        self.active_callsigns = set()

        # Create a store for HEMS resources
        self.store = FilterStore(env)

        self.serviceStore = FilterStore(env)

        # Prepare HEMS resources for ingesting into store
        self.prep_HEMS_resources()

        # Populate the store with HEMS resources
        self.populate_store()

        # Daily servicing check (in case sim starts during a service)
        [dow, hod, weekday, month, qtr, current_dt] = self.utilityClass.date_time_of_call(self.sim_start_date, self.env.now)

        self.daily_servicing_check(current_dt, hod, month)

    def debug(self, message: str):
        if self.print_debug_messages:
            logging.debug(message)
            #print(message)

    def daily_servicing_check(self, current_dt: datetime, hour: int, month: int):
        """
            Function to iterate through the store and trigger the service check
            function in the HEMS class
        """
        h: HEMS

        self.debug('------ DAILY SERVICING CHECK -------')

        GDAAS_service = False

        all_resources = self.serviceStore.items + self.store.items
        for h in all_resources:
            if h.registration == 'g-daas':
                GDAAS_service = h.unavailable_due_to_service(current_dt)
                break

        self.debug(f"GDAAS_service is {GDAAS_service}")

        # --- Return from serviceStore to store ---
        to_return = [
            (s.category, s.registration)
            for s in self.serviceStore.items
            if not s.service_check(current_dt, GDAAS_service) # Note the NOT here!
        ]

        # Attempted fix for gap after return from H70 duties
        for h in self.store.items:
            if h.registration == 'g-daan' and not GDAAS_service:
                h.callsign_group = 71
                h.callsign = 'H71'

        if to_return:
            self.debug("Service store has items to return")

        for category, registration in to_return:
            s = yield self.serviceStore.get(
                lambda item: item.category == category and item.registration == registration
            )
            yield self.store.put(s)
            self.debug(f"Returned [{s.category} / {s.registration}] from service to store")

        # --- Send from store to serviceStore ---
        to_service = [
            (h.category, h.registration)
            for h in self.store.items
            if h.service_check(current_dt, GDAAS_service)
        ]

        for category, registration in to_service:
            self.debug("****************")
            self.debug(f"HEMS [{category} / {registration}] being serviced, removing from store")

            h = yield self.store.get(
                lambda item: item.category == category and item.registration == registration
            )

            self.debug(f"HEMS [{h.category} / {h.registration}] successfully removed from store")
            yield self.serviceStore.put(h)
            self.debug(f"HEMS [{h.category} / {h.registration}] moved to service store")
            self.debug("***********")

        self.debug(self.current_store_status(hour, month))
        self.debug(self.current_store_status(hour, month, 'service'))

        [dow, hod, weekday, month, qtr, current_dt] = self.utilityClass.date_time_of_call(self.sim_start_date, self.env.now)
        for h in self.store.items:
            if h.registration == 'g-daan':
                self.debug(f"[{self.env.now}] g-daan status: in_use={h.in_use}, callsign={h.callsign}, group={h.callsign_group}, on_shift={h.hems_resource_on_shift(hod, month)}")

        self.debug('------ END OF DAILY SERVICING CHECK -------')


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
        SERVICE_HISTORY = pd.read_csv('actual_data/service_history.csv', na_values=0)
        CALLSIGN_REGISTRATION = pd.read_csv('actual_data/callsign_registration_lookup.csv')

        SERVICING_SCHEDULE = SERVICING_SCHEDULE.merge(
            CALLSIGN_REGISTRATION,
            how="right",
            on="model"
            )

        SERVICING_SCHEDULE = SERVICING_SCHEDULE.merge(
            SERVICE_HISTORY,
            how="left",
            on="registration"
            )

        self.debug(f"prep_hems_resources: schedule {SERVICING_SCHEDULE}")

        for index, row in SERVICING_SCHEDULE.iterrows():
            #self.debug(row)
            current_resource_service_dates = []
            # Check if service date provided
            if not pd.isna(row['last_service']):
                #self.debug(f"Checking {row['registration']} with previous service date of {row['last_service']}")
                last_service = datetime.strptime(row['last_service'], "%Y-%m-%d")
                service_date = last_service

                while last_service < self.sim_end_date:

                    end_date = last_service + \
                        timedelta(weeks = int(row['service_duration_weeks'])) + \
                        timedelta(weeks=self.serviing_buffer_weeks)

                    service_date, end_date = self.find_next_service_date(
                        last_service,
                        row["service_schedule_months"],
                        service_dates,
                        row['service_duration_weeks']
                    )

                    schedule.append((row['registration'], service_date, end_date))
                    #self.debug(service_date)
                    service_dates.append({'service_start_date': service_date, 'service_end_date': end_date})

                    current_resource_service_dates.append({'service_start_date': service_date, 'service_end_date': end_date})
                    #self.debug(service_dates)
                    #self.debug(current_resource_service_dates)
                    last_service = service_date
            else:
                schedule.append((row['registration'], None, None))
                #self.debug(schedule)

            service_df = pd.DataFrame(schedule, columns=["registration", "service_start_date", "service_end_date"])

            service_df.to_csv("data/service_dates.csv", index=False)

        # Append those to the HEMS resource.

        HEMS_RESOURCES = (
            pd.read_csv("actual_data/HEMS_ROTA.csv")
                # Add model and servicing rules
                .merge(
                    SERVICING_SCHEDULE,
                    on=["callsign", "vehicle_type"],
                    how="left"
                )
        )

        for index, row in HEMS_RESOURCES.iterrows():

            s = service_df[service_df['registration'] == row['registration']]
            #self.debug(s)

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

        #self.debug(self.HEMS_resources_list)


    def populate_store(self):
        """
            Function to populate the filestore with HEMS class objects
            contained in a class list
        """

        h: HEMS
        for h in self.HEMS_resources_list:
            self.debug(f"Populating resource store: HEMS({h.callsign})")
            self.debug(h.servicing_schedule)
            self.store.put(h)


    def add_hems(self):
        """
            Future function to allow for adding HEMS resources.
            We might not use this (we could just amend the HEMS_ROTA dataframe, for example)
            but might be useful for 'what if' simulations
        """
        pass

    # def resource_allocation_lookup(self, prefered_lookup: int):
    #     """
    #         Function to return description of lookup allocation choice

    #     """

    #     lookup_list = [
    #         "No HEMS resource available",
    #         "Preferred HEMS care category and vehicle type match",
    #         "Preferred HEMS care category match but not vehicle type",
    #         "HEMS CC case EC helicopter available",
    #         "HEMS CC case EC car available",
    #         "HEMS EC case CC helicopter available",
    #         "HEMS EC case CC car available",
    #         "HEMS helicopter case EC helicopter available",
    #         "HEMS helicopter case CC helicopter available",
    #         "HEMS REG case no helicopter benefit preferred group and vehicle type allocated",
	#         "HEMS REG case no helicopter benefit preferred group allocated",
    #         "No HEMS resource available (pref vehicle type = 'Other')",
    #         "HEMS REG case no helicopter benefit first free resource allocated"
    #     ]

    #     return lookup_list[prefered_lookup]

    def resource_allocation_lookup(self, reason: ResourceAllocationReason) -> str:
        return self.LOOKUP_LIST[reason.value]

    def current_store_status(self, hour, month, store = 'resource') -> list[str]:
            """
                Debugging function to return current state of store
            """

            current_store_items = []

            h: HEMS

            if store == 'resource':
                for h in self.store.items:
                    current_store_items.append(f"{h.callsign} ({h.category} online: {h.hems_resource_on_shift(hour, month)} {h.registration})")
            else:
                for h in self.serviceStore.items:
                    current_store_items.append(f"{h.callsign} ({h.category} online: {h.hems_resource_on_shift(hour, month)} {h.registration})")

            return current_store_items


    # def preferred_resource_available(self, pt: Patient) -> list[HEMS | None, str]:

    #     """
    #         Check whether the preferred resource group is available. Returns a list with either the HEMS resource or None, and
    #         an indication as to whether the resource was available, or another resource in an established hierachy of
    #         availability can be allocated
    #     """

    #     # Initialise object HEMS as a placeholder object
    #     hems: HEMS | None = None
    #     # Initialise variable 'preferred' to False
    #     preferred = 999 # This will be used to ensure that the most desireable resource
    #                 # is allocated given that multiple matches may be found
    #     preferred_lookup = 0 # This will be used to code the resource allocation choice

    #     preferred_care_category = pt.hems_cc_or_ec

    #     self.debug(f"EC/CC resource with {preferred_care_category} and hour {pt.hour} and qtr {pt.qtr}")

    #     h: HEMS
    #     for h in self.store.items:

    #         # There is a hierachy of calls:
    #         # CC = H70 helicopter then car, then H71 helicopter then car then CC72
    #         # EC = H71 helicopter then car, then CC72, then H70 helicopter then car
    #         # If no resources then return None

    #         if not h.in_use and h.hems_resource_on_shift(pt.hour, pt.qtr):
    #             # self.debug(f"Checking whether to generate ad-hoc reason for {h} ({h.vehicle_type} - {h.callsign_group})")

    #             if ( # Skip this resource if any of the following are true:
    #                 h.in_use or
    #                 h.being_serviced or
    #                 not h.hems_resource_on_shift(pt.hour, pt.qtr) or
    #                 # Skip if crew is already in use
    #                 h.callsign_group in self.active_callsign_groups or
    #                 h.registration in self.active_registrations
    #             ):
    #                 # self.debug(f"Skipping ad-hoc unavailability check for {h}")
    #                 continue

    #             if h.vehicle_type == "car":
    #                 ad_hoc_reason = "available"
    #             else:
    #                 ad_hoc_reason = self.utilityClass.sample_ad_hoc_reason(pt.hour, pt.qtr, h.registration)

    #             self.debug(f"({h.callsign}) Sampled reason for patient {pt.id} ({pt.hems_cc_or_ec}) is: {ad_hoc_reason}")

    #             if ad_hoc_reason != "available":
    #                 continue

    #             if h.category == preferred_care_category and h.vehicle_type == "helicopter" and not h.being_serviced:
    #                 hems = h
    #                 preferred = 1 # Top choice
    #                 preferred_lookup = 1
    #                 return [hems,  self.resource_allocation_lookup(preferred_lookup)]

    #             elif h.category == preferred_care_category and not h.being_serviced:
    #                 hems = h
    #                 preferred = 2 # Second choice (correct care category, but not vehicle type)
    #                 preferred_lookup = 2

    #             elif preferred_care_category == 'CC':
    #                 if h.vehicle_type == 'helicopter' and h.category == 'EC' and not h.being_serviced:
    #                     if preferred > 3:
    #                         hems = h
    #                         preferred = 3 # Third  choice (EC helicopter)
    #                         preferred_lookup = 3

    #                 elif h.category == 'EC' and not h.being_serviced:
    #                     if preferred > 4:
    #                         hems = h
    #                         preferred = 4
    #                         preferred_lookup = 4

    #             elif preferred_care_category == 'EC':
    #                 if h.vehicle_type == 'helicopter' and h.category == 'CC' and not h.being_serviced:
    #                     if preferred > 3:
    #                         hems = h
    #                         preferred = 3 # CC helicopter available
    #                         preferred_lookup = 5

    #                 elif h.category == 'CC' and not h.being_serviced:
    #                     hems = h
    #                     preferred = 4
    #                     preferred_lookup = 6

    #     self.debug(f"preferred lookup {preferred_lookup} and preferred = {preferred}")

    #     if preferred_lookup != 999:
    #         return [hems, self.resource_allocation_lookup(preferred_lookup)]
    #     else:
    #         return [None, self.resource_allocation_lookup(0)]

    def preferred_resource_available(self, pt: Patient) -> list[HEMS | None, str]:
        """
        Determine the best available HEMS resource for an EC/CC case based on the
        patient's preferred care category (EC = Enhanced Care, CC = Critical Care),
        vehicle type, and ad-hoc availability. Returns the chosen HEMS unit and
        the reason for its selection.

        Returns:
            A list containing:
                - The selected HEMS unit (or None if none available)
                - A lookup code describing the allocation reason
        """
        # Retrieve patient’s preferred care category
        preferred_category = pt.hems_cc_or_ec
        self.debug(f"EC/CC resource with {preferred_category} and hour {pt.hour} and qtr {pt.qtr}")

        best_hems: HEMS | None = None       # Best-matching HEMS unit found so far
        best_priority = float('inf')        # Lower values = better matches
        best_lookup = ResourceAllocationReason.NONE_AVAILABLE # Reason for final allocation

        for h in self.store.items:
            # --- FILTER OUT UNAVAILABLE RESOURCES ---
            if (
                h.in_use or  # Already dispatched
                h.being_serviced or # Currently under maintenance
                not h.hems_resource_on_shift(pt.hour, pt.month) or # Not scheduled for shift now
                h.callsign_group in self.active_callsign_groups or # Another unit from this group is active (so crew is engaged elsewhere)
                h.registration in self.active_registrations # This specific unit is already dispatched
            ):
                continue  # Move to the next HEMS unit

            # Check ad-hoc reason
            # For "car" units, assume always available.
            # For helicopters, simulate availability using ad-hoc logic (e.g., weather, servicing).
            reason = "available" if h.vehicle_type == "car" else self.utilityClass.sample_ad_hoc_reason(pt.hour, pt.qtr, h.registration)
            self.debug(f"({h.callsign}) Sampled reason for patient {pt.id} ({pt.hems_cc_or_ec}) is: {reason}")

            if reason != "available":
                continue # Skip this unit if not usable

            # Decide priority and reason
            priority = None
            lookup = None

            # 1. Best case: resource category matches preferred care category *and* is a helicopter
            if h.category == preferred_category and h.vehicle_type == "helicopter":
                priority = 1
                lookup = ResourceAllocationReason.MATCH_PREFERRED_CARE_CAT_HELI

            # 2. Next best: resource category matches preferred care category, but is a car
            elif h.category == preferred_category:
                priority = 2
                lookup = ResourceAllocationReason.MATCH_PREFERRED_CARE_CAT_CAR

            # 3–4. Category mismatch fallback options:
            # For a CC preference, fall back to EC providers if needed
            elif preferred_category == "CC":
                if h.category == "EC" and h.vehicle_type == "helicopter":
                    priority = 3
                    lookup = ResourceAllocationReason.CC_MATCH_EC_HELI
                elif h.category == "EC":
                    priority = 4
                    lookup = ResourceAllocationReason.CC_MATCH_EC_CAR

            # For an EC preference, fall back to CC providers if needed
            elif preferred_category == "EC":
                if h.category == "CC" and h.vehicle_type == "helicopter":
                    priority = 3
                    lookup = ResourceAllocationReason.EC_MATCH_CC_HELI
                elif h.category == "CC":
                    priority = 4
                    lookup = ResourceAllocationReason.EC_MATCH_CC_CAR

            # --- CHECK IF THIS IS THE BEST OPTION SO FAR ---
            if priority is not None and priority < best_priority:
                best_hems = h
                best_priority = priority
                best_lookup = lookup

                # Immediate return if best possible match found (priority 1)
                if priority == 1:
                    self.debug(f"Top priority match found: {best_lookup.name} ({best_lookup.value})")
                    return [best_hems, self.resource_allocation_lookup(best_lookup)]

        # Final fallback: return the best match found (if any), or none with reason
        self.debug(f"Selected best lookup: {best_lookup.name} ({best_lookup.value}) with priority = {best_priority}")
        return [best_hems, self.resource_allocation_lookup(best_lookup)]


    def allocate_resource(self, pt: Patient) -> Any | Event:
        """
        Attempt to allocate a resource from the preferred group.
        """
        resource_event: Event = self.env.event()

        def process() -> Generator[Any, Any, None]:
            self.debug(f"Allocating resource for {pt.id} and care cat {pt.hems_cc_or_ec}")

            pref_res: list[HEMS | None, str] = self.preferred_resource_available(pt)

            if pref_res[0] is None:
                return resource_event.succeed([None, pref_res[1], None])

            primary = pref_res[0]

            # Block if in-use by callsign group, registration, or callsign
            if primary.callsign_group in self.active_callsign_groups:
                self.debug(f"[BLOCKED] Callsign group {primary.callsign_group} in use")
                return resource_event.succeed([None, pref_res[1], None])

            if primary.registration in self.active_registrations:
                self.debug(f"[BLOCKED] Registration {primary.registration} in use")
                return resource_event.succeed([None, pref_res[1], None])

            if primary.callsign in self.active_callsigns:
                self.debug(f"[BLOCKED] Callsign {primary.callsign} already in use")
                return resource_event.succeed([None, pref_res[1], None])

            self.active_callsign_groups.add(primary.callsign_group)
            self.active_registrations.add(primary.registration)
            self.active_callsigns.add(primary.callsign)

            with self.store.get(lambda r: r == primary) as primary_request:
                result = yield primary_request | self.env.timeout(0.1)

                if primary_request in result:
                    primary.in_use = True
                    pt.hems_callsign_group = primary.callsign_group
                    pt.hems_vehicle_type = primary.vehicle_type
                    pt.hems_category = primary.category

                    # self.active_callsign_groups.add(primary.callsign_group)
                    # self.active_registrations.add(primary.registration)
                    # self.active_callsigns.add(primary.callsign)

                    # Try to get a secondary resource
                    with self.store.get(lambda r:
                        r != primary and
                        r.callsign_group == pt.hems_callsign_group and
                        r.category == pt.hems_category and
                        r.hems_resource_on_shift(pt.hour, pt.month) and
                        r.callsign_group not in self.active_callsign_groups and
                        r.registration not in self.active_registrations and
                        r.callsign not in self.active_callsigns
                    ) as secondary_request:

                        result2 = yield secondary_request | self.env.timeout(0.1)
                        secondary = None

                        if secondary_request in result2:
                            secondary = result2[secondary_request]
                            secondary.in_use = True
                            self.active_callsign_groups.add(secondary.callsign_group)
                            self.active_registrations.add(secondary.registration)
                            self.active_callsigns.add(secondary.callsign)

                    return resource_event.succeed([primary, pref_res[1], secondary])
                else:
                        # Roll back if unsuccessful
                    self.active_callsign_groups.discard(primary.callsign_group)
                    self.active_registrations.discard(primary.registration)
                    self.active_callsigns.discard(primary.callsign)

                    return resource_event.succeed([None, pref_res[1], None])

        self.env.process(process())
        return resource_event


    def return_resource(self, resource: HEMS, secondary_resource: HEMS|None) -> None:
        """
            Class to return HEMS class object back to the filestore.
        """

        resource.in_use = False
        self.active_callsign_groups.discard(resource.callsign_group)
        self.active_registrations.discard(resource.registration)
        self.active_callsigns.discard(resource.callsign)
        self.store.put(resource)
        self.debug(f"{resource.callsign} finished job")

        if secondary_resource is not None:
            secondary_resource.in_use = False
            self.active_callsign_groups.discard(secondary_resource.callsign_group)
            self.active_registrations.discard(secondary_resource.registration)
            self.active_callsigns.discard(secondary_resource.callsign)
            self.store.put(secondary_resource)
            self.debug(f"{secondary_resource.callsign} free as {resource.callsign} finished job")



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

        # self.debug(f"Next due: {next_due_date} with end date {end_date} and preferred_date is {preferred_date} with pref end {preferred_end_date}")

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


    # def preferred_regular_group_available(self, pt: Patient) -> list[HEMS | None, str]:
    #     """
    #     Check availability for REG jobs while avoiding assigning two units from
    #     the same crew/callsign group.
    #     """
    #     hems: HEMS | None = None
    #     preferred = 999
    #     preferred_lookup = 0

    #     preferred_group = pt.hems_pref_callsign_group
    #     preferred_vehicle_type = pt.hems_pref_vehicle_type
    #     helicopter_benefit = pt.hems_helicopter_benefit

    #     for h in self.store.items:

    #         if (
    #             h.in_use or
    #             h.being_serviced or
    #             not h.hems_resource_on_shift(pt.hour, pt.qtr) or
    #             # Skip if crew is already in use
    #             h.callsign_group in self.active_callsign_groups or
    #             h.registration in self.active_registrations
    #             ):
    #             continue


    #         if h.vehicle_type == "car":
    #             ad_hoc_reason = "available"
    #         else:
    #             ad_hoc_reason = self.utilityClass.sample_ad_hoc_reason(pt.hour, pt.qtr, h.registration)

    #         self.debug(f"({h.callsign}) Sampled reason for patient {pt.id} (REG) is: {ad_hoc_reason}")

    #         if ad_hoc_reason != "available":
    #             continue

    #         # Helicopter benefit cases
    #         if helicopter_benefit == 'y':
    #             if h.vehicle_type == 'helicopter' and h.category == 'EC':
    #                 hems = h
    #                 preferred_lookup = 7
    #                 break
    #             elif h.vehicle_type == 'helicopter':
    #                 if preferred > 2:
    #                     hems = h
    #                     preferred = 2
    #                     preferred_lookup = 8

    #         # Regular (non-helicopter) cases
    #         else:
    #             if (h.callsign_group == preferred_group
    #                 and h.vehicle_type == preferred_vehicle_type):
    #                 hems = h
    #                 preferred_lookup = 9
    #                 break
    #             elif h.callsign_group == preferred_group:
    #                 if preferred > 4:
    #                     hems = h
    #                     preferred = 4
    #                     preferred_lookup = 10
    #             else:
    #                 if preferred > 5:
    #                     hems = h
    #                     preferred = 5
    #                     preferred_lookup = 12

    #     return [hems, self.resource_allocation_lookup(preferred_lookup if hems else 0)]

    def preferred_regular_group_available(self, pt: Patient) -> list[HEMS | None, str]:
        """
        Determine the most suitable HEMS (Helicopter Emergency Medical Service) unit
        for a REG (regular) job, ensuring that two units from the same crew/callsign
        group are not simultaneously active. This method prioritizes matching the
        patient's preferences and helicopter benefit where applicable.

        Returns:
            A list containing:
                - The selected HEMS unit (or None if none available)
                - A lookup code describing the allocation reason
        """
        # Initialize selection variables
        hems: HEMS | None = None # The selected resource
        preferred = float("inf") # Lower numbers mean more preferred options (2, 4, 5 are used below)
        preferred_lookup = ResourceAllocationReason.NONE_AVAILABLE  # Reason code for selection

        preferred_group = pt.hems_pref_callsign_group        # Preferred crew group
        preferred_vehicle_type = pt.hems_pref_vehicle_type    # e.g., "car" or "helicopter"
        helicopter_benefit = pt.hems_helicopter_benefit       # "y" if helicopter has clinical benefit
        # Iterate through all HEMS resources stored
        for h in self.store.items:
            if (
                h.in_use or # Already dispatched
                h.being_serviced or # Currently under maintenance
                not h.hems_resource_on_shift(pt.hour, pt.month) or # Not scheduled for shift now
                h.callsign_group in self.active_callsign_groups or  # Another unit from this group is active (so crew is engaged elsewhere)
                h.registration in self.active_registrations # This specific unit is already dispatched
            ):
                continue # Move to the next HEMS unit

            # Check ad hoc availability
            # For "car" units, assume always available.
            # For helicopters, simulate availability using ad-hoc logic (e.g., weather, servicing).
            reason = "available" if h.vehicle_type == "car" else self.utilityClass.sample_ad_hoc_reason(pt.hour, pt.qtr, h.registration)
            self.debug(f"({h.callsign}) Sampled reason for patient {pt.id} (REG) is: {reason}")

            if reason != "available":
                continue # Skip this unit if not usable

            # --- HELICOPTER BENEFIT CASE ---
            # P3 = Helicopter patient
            # Resources allocated in following order:
            # IF H70 available = SEND
            # ELSE H71 available = SEND

            if helicopter_benefit == "y":
                # Priority 1: CC-category helicopter (assumed most beneficial)
                if h.vehicle_type == "helicopter" and h.category == "CC":
                    hems = h
                    preferred_lookup = ResourceAllocationReason.REG_HELI_BENEFIT_MATCH_CC_HELI
                    break
                # Priority 2: Any helicopter (less preferred than CC, hence priority = 2)
                elif h.vehicle_type == "helicopter" and preferred > 2:
                    hems = h
                    preferred = 2
                    preferred_lookup = ResourceAllocationReason.REG_HELI_BENEFIT_MATCH_EC_HELI

                # If no EC or CC helicopters are available, then:
                # - hems remains None
                # - preferred_lookup remains at its initial value (ResourceAllocationReason.NONE_AVAILABLE)
                # - The function exits the loop without assigning a resource.

            # --- REGULAR JOB WITH NO SIMULATED HELICOPTER BENEFIT ---
            else:
                # Best match: matching both preferred callsign group and vehicle type
                if h.callsign_group == preferred_group and h.vehicle_type == preferred_vehicle_type:
                    hems = h
                    preferred_lookup = ResourceAllocationReason.REG_NO_HELI_BENEFIT_GROUP_AND_VEHICLE
                    break
                # Next best: send a helicopter
                elif h.vehicle_type == "helicopter" and preferred > 3:
                    hems = h
                    preferred = 3
                    preferred_lookup = ResourceAllocationReason.REG_NO_HELI_BENEFIT_VEHICLE
                # Next best: match only on preferred callsign group
                elif h.callsign_group == preferred_group and preferred > 4:
                    hems = h
                    preferred = 4
                    preferred_lookup = ResourceAllocationReason.REG_NO_HELI_BENEFIT_GROUP
                # Fallback: any available resource
                elif preferred > 5:
                    hems = h
                    preferred = 5
                    preferred_lookup = ResourceAllocationReason.REG_NO_HELI_BENEFIT_ANY

        # Return the best found HEMS resource and reason for selection
        self.debug(f"Selected REG (heli benefit = {helicopter_benefit}) lookup: {preferred_lookup.name} ({preferred_lookup.value})")
        return [hems, self.resource_allocation_lookup(preferred_lookup if hems else ResourceAllocationReason.NONE_AVAILABLE)]


    def allocate_regular_resource(self, pt: Patient) -> Any | Event:
        """
        Attempt to allocate a resource from the preferred group (REG jobs).
        """
        resource_event: Event = self.env.event()

        def process() -> Generator[Any, Any, None]:
            # if pt.hems_pref_vehicle_type == "Other":
            #     pref_res = [None, self.resource_allocation_lookup(11)]
            # else:
            pref_res = self.preferred_regular_group_available(pt)

            if pref_res[0] is None:
                return resource_event.succeed([None, pref_res[1], None])

            primary = pref_res[0]

            # Block if in-use by callsign group, registration, or callsign
            if primary.callsign_group in self.active_callsign_groups:
                self.debug(f"[BLOCKED] Regular Callsign group {primary.callsign_group} in use")
                return resource_event.succeed([None, pref_res[1], None])

            if primary.registration in self.active_registrations:
                self.debug(f"[BLOCKED] Regular Registration {primary.registration} in use")
                return resource_event.succeed([None, pref_res[1], None])

            if primary.callsign in self.active_callsigns:
                self.debug(f"[BLOCKED] Regular Callsign {primary.callsign} already in use")
                return resource_event.succeed([None, pref_res[1], None])

            self.active_callsign_groups.add(primary.callsign_group)
            self.active_registrations.add(primary.registration)
            self.active_callsigns.add(primary.callsign)

            with self.store.get(lambda r: r == primary) as primary_request:
                result = yield primary_request | self.env.timeout(0.1)

                if primary_request in result:
                    primary.in_use = True
                    pt.hems_callsign_group = primary.callsign_group
                    pt.hems_vehicle_type = primary.vehicle_type
                    pt.hems_category = primary.category

                    # self.active_callsign_groups.add(primary.callsign_group)
                    # self.active_registrations.add(primary.registration)
                    # self.active_callsigns.add(primary.callsign)

                    # Try to get a secondary resource
                    with self.store.get(lambda r:
                        r != primary and
                        r.callsign_group == pt.hems_callsign_group and
                        r.category == pt.hems_category and
                        r.hems_resource_on_shift(pt.hour, pt.month) and
                        r.callsign_group not in self.active_callsign_groups and
                        r.registration not in self.active_registrations and
                        r.callsign not in self.active_callsigns
                    ) as secondary_request:

                        result2 = yield secondary_request | self.env.timeout(0.1)
                        secondary = None

                        if secondary_request in result2:
                            secondary = result2[secondary_request]
                            secondary.in_use = True
                            self.active_callsign_groups.add(secondary.callsign_group)
                            self.active_registrations.add(secondary.registration)
                            self.active_callsigns.add(secondary.callsign)

                    return resource_event.succeed([primary, pref_res[1], secondary])

                else:
                    # Roll back if unsuccessful
                    self.active_callsign_groups.discard(primary.callsign_group)
                    self.active_registrations.discard(primary.registration)
                    self.active_callsigns.discard(primary.callsign)

                    return resource_event.succeed([None, pref_res[1], None])

        self.env.process(process())
        return resource_event
