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

        hems = HEMS
        preferred = False
        for h in self.store.items:
            #print(f"pref group is {preferred_group} and pref_veh is {preferred_vehicle_type} and h vehicle is {h.vehicle_type} and h callsing group is {h.callsign_group.iloc[0]}")
            #print(int(h.callsign_group.iloc[0]) == int(preferred_group))
            if int(h.callsign_group.iloc[0]) == int(preferred_group) and h.vehicle_type == preferred_vehicle_type:
                return h
            elif h.callsign_group.iloc[0] == preferred_group:
                hems = h
                preferred = True
            
        if preferred:
            return hems
        else:
            return None



    def allocate_resource(self, pt: Patient):
        """Attempt to allocate a resource from the preferred group."""
        
        print(f"Allocating resource with callsign group {pt.hems_pref_callsign_group} and vehicle {pt.hems_pref_vehicle_type}")

        pref_res = self.preferred_group_available(pt.hems_pref_callsign_group, pt.hems_pref_vehicle_type)

        resource_event: Event = self.env.event()

        def process():
            def resource_filter(resource: HEMS, pref_res: HEMS):
                #print(f"Resource filter with hour {hour} and qtr {qtr}")
                if not resource.in_use and resource.hems_resource_on_shift(pt.hour, pt.qtr):
                    if pref_res != None:
                        #print(f"{resource.callsign} and {pre_res.callsign}")
                        if resource.callsign == pref_res.callsign:
                            #print("Preferred resource available")
                            return True
                    else:
                        #print("Other resource available")
                        return True
            
                return False
            
            request = self.store.get(lambda item: resource_filter(item, pref_res))

            try:
                
                resource: HEMS = yield request
                #print(resource)
                resource.in_use = True
                #print(resource)
                #print(f"Allocating HEMS resource {resource.callsign} at time {hour}")
                pt.hems_callsign_group = resource.callsign_group.iloc[0]
                pt.hems_vehicle_type = resource.vehicle_type

                resource_event.succeed(resource)
                
            except Interrupt:
                print(f"No HEMS resource available using Ambulance")
                resource_event.succeed()

        self.env.process(process())
    
        return resource_event


    def return_resource(self, resource):
        #print(f"Returning resource {resource.callsign}")
        #print(f"Current store length is {len(self.store.items)}")
        resource.in_use = False
        self.store.put(resource)
        #print(f"Current store length is {len(self.store.items)}")

