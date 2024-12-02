from utils import Utils
import pandas as pd
from class_hems import HEMS
from simpy import FilterStore

class HEMSAvailability():
    """
        # The HEMS Availability class

        This class is a filter store which can provide HEMS resources
        based on the time of day and servicing schedule

    
    """

    def __init__(self, env, current_date: str):
       
        self.env = env

        # set number of possible HEMS resources
        self.number_hems_resources = len(Utils.HEMS_ROTA)

        self.hems_callsigns = Utils.HEMS_ROTA.index

        # Create a store for HEMS resources
        self.hems = FilterStore(env)

        # Populate the store with HEMS resources
        self.hems_list = []
        for index, row in Utils.HEMS_ROTA.iterrows():
            self.hems_list.append(HEMS(index, current_date))

        self.hems.items = self.hems_list

    def add_hems(self):
        """
            Future function to allow for adding HEMS resources.
            We might not use this (we could just amend the HEMS_ROTA dataframe, for example)
            but might be useful for 'what if' simulations
        """
        pass

    def hems_resource_on_shift(self, callsign: str, hour: int, season: int):

        #print(f"on shift callsign {callsign}, hour {hour}, season {season}")
        
        df = Utils.HEMS_ROTA
        df = df[df.index == callsign]

        #print(df)

        start = df.summer_start.iloc[0] if season == 3 else df.winter_start.iloc[0]
        end = df.summer_end.iloc[0] if season == 3 else df.winter_end.iloc[0]

        #print(f"Start is {start} and end is {end}")

        if start >= hour and hour <= end:
            return True
        
        return False


    def available_hems_resources(self, item: HEMS, hour: int, season: str):
        #print(f"Inside resource with {item.callsign} and hours {hour} and season {season}")

        # For now, check if HEMS resource has completed it's servicing schedule and is therefor ready to come
        # back into service. Not sure if there is a better place to check this
        
        item.operational_after_service(self.env.now)

        return (item.being_serviced == 0 and self.hems_resource_on_shift(item.callsign, hour, season))


    def get(self, hour: int, season: str):
        """
            Get a HEMS resource

            returns a get request that can be yield to
        """

        hems_res = self.hems.get(lambda item : self.available_hems_resources(item, hour, season))

        return hems_res

    def put(self, hems_res):
        """
            Free up HEMS resource
        """

        self.hems.put(hems_res)
        #print(f'{self.env.now:0.2f} HEMS returned')


        

