import os, simpy
import pandas as pd
from random import expovariate
from utils import Utils
from class_patient import Patient
from class_hems_availability import HEMSAvailability
from class_hems import HEMS
from class_ambulance import Ambulance

class DES_HEMS:
    """
        # The DES_HEMS class

        This class contains everything require to undertake a single DES run for multiple patients
        over the specified duration.

        NOTE: The unit of sim is minutes

    """

    def __init__(self, run_number: int, sim_duration: int, warm_up_duration: int, sim_start_date: str):

        self.run_number = run_number + 1 # Add 1 so we don't have a run_number 0
        self.sim_duration = sim_duration
        self.warm_up_duration = warm_up_duration
        self.sim_start_date = sim_start_date

        self.all_results_location = Utils.ALL_RESULTS_CSV

        self.env = simpy.Environment()
        self.patient_counter = 0

        # Going to need to work out how we can keep track of flying hours
        self.hems_resources = HEMSAvailability(self.env, sim_start_date)

        # Set up empty list to store results prior to conversion to dataframe
        self.results_list = []

        # Set up data frame to capture time points etc. during the simulation
        # We might not need all of these, but simpler to capture them all for now.

        # This might be better as a separate 'data collection' class of some sort.
        # I am thinking that each resource will need its own entry for any given
        # patient to account for dual response (ambulance service and HEMS)
        # stand downs etc.
        self.results_df = None

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

                    # Get current day of week and hour of day
                    [dow, hod, weekday, month, qtr, current_dt] = Utils.date_time_of_call(self.sim_start_date, self.env.now)

                    # Update patient instance with time-based values so the current time is known
                    pt.day = dow
                    pt.hour = hod
                    pt.weekday = weekday
                    pt.month = month
                    pt.qtr = qtr
                    pt.current_dt = current_dt

                    # Set caller/patient off on their HEMS healthcare journey
                    self.env.process(self.patient_journey(pt))

                    # Convery weekday/weekend into boolean value
                    weekday_bool = 1 if weekday == 'weekday' else 0

                    # Determine the interarrival time for the next patient by sampling from the exponential distrubution

                    # We need a lookup table for mean inter arrival times. A tabulated version of Figure 1 from
                    # the UoR report would be a good starter for 10...
                    inter_time = 30
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

        # Ambulance resource here?
        # Might also need some logic to determine what the resource(s) requirements are.
        # No point getting a HEMS resource for a non-HEMS job, for example.
        # Probably will determine whether it is a 'HEMS' job when the patient is created
        # then use the resource store to determine whether the hems resource is available
        if patient.hems_case == 1:
            hems = yield self.hems_resources.get(patient.hour, patient.qtr)

        ambulance = Ambulance()

        while patient.incident_completed == 0:

            # Add boolean to determine whether the patient is still within the simulation warm-up
            # period. If so, then we will not record the patient progress
            not_in_warm_up_period = False if self.env.now < self.warm_up_duration else True

            if not_in_warm_up_period:
                if patient.hems_case == 1:
                    self.add_patient_result_row(patient, hems, "HEMS call start")
                self.add_patient_result_row(patient, ambulance, "AMB call start")


            yield self.env.timeout(30)
            patient.time_in_sim = self.env.now - patient_enters_sim

            if not_in_warm_up_period:
                # Will need separate rows to keep track of ambulance and HEMS
                # Needs some thought that....
                if patient.hems_case == 1:
                    self.add_patient_result_row(patient, hems, "HEMS arrival at hospital")
                self.add_patient_result_row(patient, ambulance, "AMB arrival at hospital")

            if patient.hems_case == 1:
                hems.flying_time += self.env.now - patient_enters_sim
                hems.update_flying_time(self.env.now)

            # We might actually yield to a process
            # So based on various characteristics, we'll want to know the job cycle times
            # etc.
            yield self.env.timeout(60)

            patient.time_in_sim = self.env.now - patient_enters_sim
            patient.incident_completed = 1

            if not_in_warm_up_period:
                if patient.hems_case == 1:
                    self.add_patient_result_row(patient, hems, "AMB handover")
                self.add_patient_result_row(patient, ambulance, "AMB arrival at hospital")


            # TODO: Add turnaround time calculation here

            yield self.env.timeout(30)

            if not_in_warm_up_period:
                if patient.hems_case == 1:
                    self.add_patient_result_row(patient, hems, "HEMS clear")
                    self.hems_resources.put(hems)
                self.add_patient_result_row(patient, ambulance, "AMB clear")




    def add_patient_result_row(self, patient: Patient, resource: HEMS|Ambulance, time_type: str, **kwargs) -> None :
        """
            Convenience function to create a row of data for the results table

        """

        results = {
            "P_ID"        : patient.id,
            "run_number"  : self.run_number,
            "time_type"   : time_type,   # e.g. mobile, at scene, leaving scene etc.
            "timestamp"   : self.env.now,
            "timestamp_dt": patient.current_dt,
            "day"         : patient.day,
            "hour"        : patient.hour,
            "weekday"     : patient.weekday,
            "month"       : patient.month,
            "qtr"         : patient.qtr,
            "callsign"    : resource.callsign,
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

        self.results_list.append(results)

    def convert_results_to_df(self, results: dict) -> None:
        self.results_df = pd.DataFrame(results)
        self.results_df.set_index("P_ID", inplace=True)

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

        # Convert results to a dataframe
        self.convert_results_to_df(results=self.results_list)

        # Write run results to file
        self.write_all_results()
