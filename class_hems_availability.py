from utils import Utils
import pandas as pd
from class_hems import HEMS
from simpy import Store

class HEMSAvailability():
    """
        # The HEMS Availability class

        This class is a store which can provide HEMS resources
        based on the time of day and servicing schedule

    
    """

    def __init__(self, env, current_date: str):
       
        self.env = env

        # set number of possible HEMS resources
        self.number_hems_resources = len(Utils.HEMS_ROTA)

        self.hems_callsigns = Utils.HEMS_ROTA.index

        # Create a store for HEMS resources
        self.hems = Store(env)

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

    def get(self):
        """
            Get a HEMS resource

            returns a get request that can be yield to
        """

        hems_res = self.hems.get()

        return hems_res

    def put(self, hems_res):
        """
            Free up HEMS resource
        """

        self.hems.put(hems_res)
        #print(f'{self.env.now:0.2f} HEMS returned')
        

