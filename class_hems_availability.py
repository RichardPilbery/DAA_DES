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

    def __init__(self, env):
       
        self.env = env
        self.utilityClass = Utils()

        # Create a store for HEMS resources
        self.store = FilterStore(env)

        # Populate the store with HEMS resources
        for index, row in self.utilityClass.HEMS_ROTA.iterrows():
            self.store.put(HEMS(index))


    def add_hems(self):
        """
            Future function to allow for adding HEMS resources.
            We might not use this (we could just amend the HEMS_ROTA dataframe, for example)
            but might be useful for 'what if' simulations
        """
        pass


    def preferred_group_available(self, preferred_group, preferred_vehicle_type):
        """
        Check whether the preferred resource group is available and respond accordingly
        """
        # Initialise object HEMS as a placeholder object
        hems = HEMS
        # Initialise variable 'preferred' to False
        preferred = False

        # Iterates through items **available** in store at the time the function is called
        for h in self.store.items:
            # print(f"pref group is {preferred_group} and pref_veh is {preferred_vehicle_type} and h vehicle is {h.vehicle_type} and h callsign group is {h.callsign_group.iloc[0]}")
            #print(int(h.callsign_group.iloc[0]) == int(preferred_group))
            # print(f"int(h.callsign_group.iloc[0]): {int(h.callsign_group.iloc[0])}")
            # print(f"int(preferred_group): {int(preferred_group)}")

            # If callsign group is preferred group AND is preferred vehicle type, returns that item at that point
            # Rest of code will not be reached as the return statement will terminate this for loop as soon as
            # this condition is met
            # (i.e. IF the preferred callsign group and vehicle type is available, we only care about that -
            # so return)
            if int(h.callsign_group.iloc[0]) == int(preferred_group) and h.vehicle_type == preferred_vehicle_type:
                return h

            # If it's the preferred group but not the preferred vehicle type, the variable
            # hems becomes the HEMS resource object that we are currently looking at in the store
            # so we will basically - in the event of not finding the exact resource we want - find
            # the next best thing from the callsign group
            # SR note 13/1/25 - double check this logic - as we would send a critical care car over
            # a different available helicopter if I'm interpreting this correctly. Just need to confirm
            # this was the order of priority agreed on.
            elif h.callsign_group.iloc[0] == preferred_group:
                hems = h
                preferred = True

        # If we have not found an exact match for preferred callsign and vehicle type out of the
        # resources currently available in our store, we will then reach this code
        # If the preferred variable was set to True at any point, we will refer HEMS
        # Note that this will be the last resource that met the condition h.callsign_group.iloc[0] == preferred_group
        # which may be relevant if there is more than one other resource within that callsign group
        # (though this is not currently a situation that occurs within the naming conventions at DAAT)
        if preferred:
            return hems
        else:
            return None



    def allocate_resource(self, pt: Patient):
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

        def process():
            def resource_filter(resource: HEMS, pref_res: HEMS):
                """
                Checks whether the resource the incident wants is available in the
                simpy FilterStore
                Returns True if resource is
                - not in use
                - on shift
                Otherwise, returns False
                """
                #print(f"Resource filter with hour {hour} and qtr {qtr}")

                # If the resource **is not currently in use** AND **is currently on shift**...
                if not resource.in_use and resource.hems_resource_on_shift(pt.hour, pt.qtr):
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
                    # print(request)
                    # print(resource)
                    resource.in_use = True
                    pt.hems_callsign_group = resource[request].callsign_group.iloc[0]
                    pt.hems_vehicle_type = resource[request].vehicle_type

                    resource_event.succeed(resource[request])
                else:
                    print(f"No HEMS (helimed or ccc) resource available; using Non-DAAT land ambulance")
                    resource_event.succeed()

        self.env.process(process())
    
        return resource_event


    def return_resource(self, resource):
        #print(f"Returning resource {resource.callsign}")
        #print(f"Current store length is {len(self.store.items)}")
        resource.in_use = False
        self.store.put(resource)
        #print(f"Current store length is {len(self.store.items)}")

