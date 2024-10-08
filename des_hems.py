import os, simpy
from random import expovariate
from utils import Utils
from class_patient import Patient
import pandas as pd

class DES_HEMS:
    """
        # The DES_HEMS class

        This class contains everything require to undertake a single DES run for multiple patients
        over the specified duration.

    """

    def __init__(self, run_number: int, sim_duration: int, warm_up_duration: int, sim_start_date: str):

        self.run_number = run_number
        self.sim_duration = sim_duration
        self.warm_up_duration = warm_up_duration
        self.sim_start_date = sim_start_date

        self.all_results_location = Utils.ALL_RESULTS_CSV


        self.env = simpy.Environment()
        self.patient_counter = 0

        # We need to create the resrouces, probably using the store function once I work
        # out how it does its thing! Allows for dynamically adjusting availability
        # in addition to usual resource in use style stuff.

        # Set up data frame to capture time points etc. during the simulation
        # We might not need all of these, but simpler to capture them all for now.
        self.results_df                             = pd.DataFrame()
        self.results_df["P_ID"]                     = []
        self.results_df["run_number"]               = []
        self.results_df["time_type"]                = [] # e.g. mobile, at scene, leaving scene etc.
        self.results_df["timestamp"]                = []
        self.results_df["day"]                      = []
        self.results_df["hour"]                     = []
        self.results_df["weekday"]                  = []
        self.results_df["callsign"]                 = []
        self.results_df["triage_code"]              = []
        self.results_df["age"]                      = []
        self.results_df["sex"]                      = []
        self.results_df["time_to_first_respone"]    = []
        self.results_df["time_to_cc"]               = []
        self.results_df["cc_conveyed"]              = []
        self.results_df["cc_flown"]                 = []
        self.results_df["cc_travelled_with"]        = []
        self.results_df["hems"]                     = []
        self.results_df["cc_desk"]                  = []
        self.results_df["dispatcher_intervention"]  = []   
        self.results_df.set_index("P_ID", inplace=True)



    def generate_calls(self):
            """
            **Patient generator**
            
            Keeps creating new patients until current time equals sim_duration + warm_up_duration
            
            """
            
            if self.env.now < self.sim_duration + self.warm_up_duration :
                while True:
                    self.patient_counter += 1
                    
                    # Create a new caller/patient
                    pt = Patient(self.patient_counter)
                                        
                    # Set caller/patient off on their HEMS healthcare journey
                    self.env.process(self.patient_journey(pt))
                    
                    # Get current day of week and hour of day
                    [dow, hod, weekday, month, current_dt] = Utils.date_time_of_call(self.sim_start_date, self.env.now)

                    # Update patient instance with time-based values so the current time is known
                    pt.day = dow
                    pt.hour = hod 
                    pt.weekday = weekday
                    pt.month = month

                    # Convery weekday/weekend into boolean value
                    weekday_bool = 1 if weekday == 'weekday' else 0
                    
                    # Determine the interarrival time for the next patient by sampling from the exponential distrubution

                    # We need a lookup table for mean inter arrival times. A tabulated version of Figure 1 from
                    # the UoR report would be a good starter for 10...
                    inter_time = 1800
                    sampled_interarrival = expovariate(1.0 / inter_time) 

                    # Use sampled interarrival time with a check to ensure it does not go over 60 minutes
                    # as this would technically be in the 'next' hour
                    sampled_interarrival = 59 if sampled_interarrival >= 60 else sampled_interarrival

                    # Freeze function until interarrival time has elapsed
                    yield self.env.timeout(sampled_interarrival)
    


    def patient_journey(self, patient: Patient):
        """
            Send patient on their journey!
        """
        #print('Patient is on journey')

        patient_enters_sim = self.env.now

        while patient.incident_completed == 0:

            # Add boolean to determine whether the patient is still within the simulation warm-up
            # period. If so, then we will not record the patient progress
            not_in_warm_up_period = False if self.env.now < self.warm_up_duration else True

            if not_in_warm_up_period:
                self.add_patient_result_row(patient)
    
            # We might actually yield to a process
            # So based on various characteristics, we'll want to know the job cycle times
            # etc.
            yield self.env.timeout(3600)

            patient.time_in_sim = self.env.now - patient_enters_sim



    def add_patient_result_row(self, patient: Patient, **kwargs) -> None :
        """
            Convenience function to create a row of data for the results table
        
        """
        results = {
            "P_ID"        : patient.id,
            "run_number"  : self.run_number,
            "time_type"   : "",
            "timestamp"   : self.env.now,         
            "day"         : patient.day,
            "hour"        : patient.hour,
            "weekday"     : patient.weekday,
            "callsign"    : "",
            "triage_code" : patient.triage_code,
            "age"         : patient.age,
            "sex"         : patient.sex,
        }

        # Add any additional items passed in **kwargs
        for key, value in kwargs.items():
             results[key] = value

        if self.env.now > self.warm_up_duration:
            self.store_patient_results(results)


    def store_patient_results(self, results: dict) -> None:      
        """
            Adds a row of data to the Class' `result_df` dataframe
        """

        df_dictionary = pd.DataFrame([results])
        
        self.results_df = pd.concat([self.results_df, df_dictionary], ignore_index=True)   

    def write_all_results(self) -> None:
        """
            Writes the content of `result_df` to a csv file
        """
        # https://stackoverflow.com/a/30991707/3650230
        
        # Check if file exists...if it does, append data, otherwise create a new file with headers matching
        # the column names of `results_df`
        if not os.path.isfile(self.all_results_location):
           self.results_df.to_csv(self.all_results_location, header='column_names')
        else: # else it exists so append without writing the header
            self.results_df.to_csv(self.all_results_location, mode='a', header=False) 


    def run(self) -> None:
        """
            Function to start the simulation.

        """
        print(f"HEMS class initialised with the following: {self.run_number} {self.sim_duration} {self.warm_up_duration} {self.sim_start_date}")

        # Start entity generators
        self.env.process(self.generate_calls())
        
        # Run simulation
        self.env.run(until=(self.sim_duration + self.warm_up_duration))
        
        # Write run results to file
        self.write_all_results() 