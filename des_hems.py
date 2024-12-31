import os, simpy
import pandas as pd
from random import expovariate
from utils import Utils
from class_patient import Patient
from class_hems_availability import HEMSAvailability
from class_hems import HEMS
from class_ambulance import Ambulance
from datetime import timedelta

class DES_HEMS:
    """
        # The DES_HEMS class

        This class contains everything require to undertake a single DES run for multiple patients
        over the specified duration.

        NOTE: The unit of sim is minutes

    """

    def __init__(self, run_number: int, sim_duration: int, warm_up_duration: int, sim_start_date: str, amb_data: bool):

        self.run_number = run_number + 1 # Add 1 so we don't have a run_number 0
        self.sim_duration = sim_duration
        self.warm_up_duration = warm_up_duration
        self.sim_start_date = sim_start_date

        # Option to include/exclude ambulance service cases in addition to HEMS
        self.amb_data = amb_data
        print(f"Ambulance data values is {self.amb_data}")

        self.utils = Utils()

        self.all_results_location = self.utils.ALL_RESULTS_CSV
        self.run_results_location = self.utils.RUN_RESULTS_CSV

        self.env = simpy.Environment()
        self.patient_counter = 0

        # Going to need to work out how we can keep track of flying hours
        self.hems_resources = HEMSAvailability(self.env)

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
                    [dow, hod, weekday, month, qtr, current_dt] = self.utils.date_time_of_call(self.sim_start_date, self.env.now)

                    # Update patient instance with time-based values so the current time is known
                    pt.day = dow
                    pt.hour = hod
                    pt.weekday = weekday
                    pt.month = month
                    pt.qtr = qtr
                    pt.current_dt = current_dt

                    # Update patient instance with age, sex, AMPDS card, whether they are a HEMS' patient and if so, the HEMS result,
                    # and patient outcome
                    #print(f"Patient hour is {pt.hour}")
                    pt.ampds_card = self.utils.ampds_code_selection(pt.hour)
                    #print(f"AMPDS card is {pt.ampds_card}")
                    pt.age = self.utils.age_sampling(pt.ampds_card)
                    pt.sex = self.utils.sex_selection(pt.ampds_card)

                    if self.amb_data:
                        # TODO: We'll need the logic to decide whether it is an ambulance or HEMS case
                        # if ambulance data is being collected too.
                        continue
                    else:
                        pt.hems_case = 1

                    if pt.hems_case == 1:
                        pt.hems_callsign_group = self.utils.callsign_group_selection(pt.hour, pt.ampds_card)
                        #print(f"Callsign is {pt.hems_callsign_group}")
                        pt.hems_vehicle_type = self.utils.vehicle_type_selection(pt.month, pt.hems_callsign_group)
                        #print(f"Vehicle type is {pt.hems_vehicle_type}")
                        pt.hems_result = self.utils.hems_result_by_callsign_group_and_vehicle_type_selection(pt.hems_callsign_group, pt.hems_vehicle_type)
                        #print(f"HEMS result is {pt.hems_result}")
                        # Note patient outcome is generic, so we'll need to include this for non-HEMS cases too
                        pt.pt_outcome = self.utils.pt_outcome_selection(pt.hems_result)

                    self.env.process(self.patient_journey(pt))

                    # Convery weekday/weekend into boolean value
                    weekday_bool = 1 if weekday == 'weekday' else 0

                    # Determine the interarrival time for the next patient by sampling from the exponential distrubution

                    # We need a lookup table for mean inter arrival times. A tabulated version of Figure 1 from
                    # the UoR report would be a good starter for 10...
                    inter_time = self.utils.inter_arrival_rate(pt.hour, pt.qtr)
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

        not_in_warm_up_period = False if self.env.now < self.warm_up_duration else True

        # Check if HEMS result indicates no leaving scene/at hospital times
        no_HEMS_hospital = True if patient.hems_result in ["Stand Down Before Mobile", "Stand Down En Route", "Landed but no patient contact", "Patient Treated (not conveyed)"] else False
                    
        if not_in_warm_up_period:
            #print(f"Arrival for patient {patient.id} on run {self.run_number}")
            self.add_patient_result_row(patient, None, "arrival", "arrival_departure")
  
        if patient.hems_case == 1:
            hems = yield self.hems_resources.get(patient.hour, patient.qtr)

        if self.amb_data:
            ambulance = Ambulance()

        # Add boolean to determine whether the patient is still within the simulation warm-up
        # period. If so, then we will not record the patient progress
        not_in_warm_up_period = False if self.env.now < self.warm_up_duration else True

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if patient.hems_case == 1:
                self.add_patient_result_row(patient, hems, "HEMS call start", "event")
            
            if self.amb_data:
                self.add_patient_result_row(patient, ambulance, "AMB call start", "event")

        # Allocation time --------------

        if patient.hems_case == 1:
            allocation_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_allocation')
            yield self.env.timeout(allocation_time)

        if self.amb_data:
                print('Ambulance allocation time')

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if patient.hems_case == 1:
                if patient.hems_result != "Stand Down Before Mobile":
                    self.add_patient_result_row(patient, hems, "HEMS allocated to call", "event")
                else:
                    self.add_patient_result_row(patient, hems, "HEMS stand down before mobile", "event")

            if self.amb_data:
                print('Ambulance time to allocation')


        # Mobilisation time ---------------

        # Calculate mobile to time at scene (or stood down before)
        if patient.hems_case == 1:
            mobile_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_mobile')
            yield self.env.timeout(mobile_time)

        if self.amb_data:
                # Determine allocation time for ambulance
                # Yield until done
                print('Ambulance time to going mobile')

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if patient.hems_case == 1:
                self.add_patient_result_row(patient, hems, "HEMS mobile", "event")

            if self.amb_data:
                print('Ambulance time to allocation')

        # On scene ---------------

        if (patient.hems_case == 1) and (patient.hems_result != "Stand Down En Route"):
            tts_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_to_scene')
            yield self.env.timeout(tts_time)

        if self.amb_data:
                # Determine allocation time for ambulance
                # Yield until done
                print('Ambulance time to scene')

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if (patient.hems_case == 1):
                if patient.hems_result != "Stand Down En Route":
                    self.add_patient_result_row(patient, hems, "HEMS on scene", "event")
                else:
                    self.add_patient_result_row(patient, hems, "HEMS stand down en route","event")

            if self.amb_data:
                print('Ambulance on scene time')


        # Leaving scene ------------

        if (patient.hems_case == 1) and (patient.hems_result != "Stand Down En Route"):
            tos_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_on_scene')
            yield self.env.timeout(tos_time)

        if self.amb_data:
            print('Ambulance on scene duration')

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if (patient.hems_case == 1):
                if no_HEMS_hospital == False:
                    self.add_patient_result_row(patient, hems, "HEMS leaving scene", "event")
                else:
                    self.add_patient_result_row(patient, hems, f"HEMS {patient.hems_result.lower()}", "event")

            if self.amb_data:
                print('Ambulance leaving scene time')


        # Arrived destination time ------------

        if (patient.hems_case == 1) and no_HEMS_hospital == False:
            travel_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_to_hospital')
            yield self.env.timeout(travel_time)

        if self.amb_data:
            print('Ambulance travel time')

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if (patient.hems_case == 1) and no_HEMS_hospital == False:
                self.add_patient_result_row(patient, hems, "HEMS arrived destination", "event")

        if self.amb_data:
            print('Ambulance at destination time')


        # Handover time ---------------

        # Not currently available

        # Clear time ------------

        if (patient.hems_case == 1):
            clear_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_to_clear')
            yield self.env.timeout(clear_time)
            self.hems_resources.put(hems)

        if self.amb_data:
            print('Ambulance clear time')

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if patient.hems_case == 1:
                self.add_patient_result_row(patient, hems, "HEMS clear", "event")

            if self.amb_data:
                print('Ambulance clear time')

        if not_in_warm_up_period:
            #print(f"Depart for patient {patient.id} on run {self.run_number}")
            self.add_patient_result_row(patient, None, "depart", "arrival_departure")
 


    def add_patient_result_row(self,
                               patient: Patient,
                               resource: None|HEMS|Ambulance,
                               time_type: str,
                               event_type: str,
                               **kwargs) -> None :
        """
            Convenience function to create a row of data for the results table

        """
        if resource is not None:
            callsign = resource.callsign
        else:
            callsign = None

        results = {
            "P_ID"              : patient.id,
            "run_number"        : self.run_number,
            "time_type"         : time_type,   # e.g. mobile, at scene, leaving scene etc.
            "event_type"        : event_type,  # for animation: arrival_departure, queue, resource_use, resource_use_end
            "timestamp"         : self.env.now,
            "timestamp_dt"      : self.sim_start_date + timedelta(minutes=self.env.now),
            "day"               : patient.day,
            "hour"              : patient.hour,
            "weekday"           : patient.weekday,
            "month"             : patient.month,
            "qtr"               : patient.qtr,
            "callsign_group"    : patient.hems_callsign_group,
            "vehicle_type"      : patient.hems_vehicle_type,
            "ampds_card"        : patient.ampds_card,
            "age"               : patient.age,
            "sex"               : patient.sex,
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

    def write_run_results(self) -> None:
        """
            Writes the content of `result_dfs` to csv files that contains only the results from the
            single run

            Note that this cannot be done in a similar manner to write_all_results due to the impacts of
            the parallisation approach that is taken with joblib - depending on process timings it can lead to
            not all
        """

        self.results_df.to_csv(f"{Utils.RESULTS_FOLDER}/output_run_{self.run_number}.csv", header='column_names')

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
        self.write_run_results()
