import math
import os, simpy
from random import expovariate, uniform
from sim_tools.time_dependent import NSPPThinning
import pandas as pd
# from random import expovariate
from utils import Utils, SeededDistribution
from class_patient import Patient
# Revised class for HEMS availability
from class_hems_availability import HEMSAvailability
from class_hems import HEMS
from class_ambulance import Ambulance
from datetime import timedelta
import warnings
import numpy as np
from math import floor
warnings.filterwarnings("ignore", category=RuntimeWarning)
# import all distributions
import ast
from numpy.random import SeedSequence
from typing import List, Dict, Tuple
from numpy.random import SeedSequence, default_rng
import random

import logging
logging.basicConfig(filename='log.txt', filemode="w", level=logging.DEBUG, format='')

class DES_HEMS:
    """
        # The DES_HEMS class

        This class contains everything require to undertake a single DES run for multiple patients
        over the specified duration.

        NOTE: The unit of sim is minutes

    """

    def __init__(self,
                run_number: int,
                sim_duration: int,
                warm_up_duration: int,
                sim_start_date: str,
                amb_data: bool,
                random_seed: int,
                demand_increase_percent: float,
                activity_duration_multiplier: float,
                print_debug_messages: bool):

        self.run_number = run_number + 1 # Add 1 so we don't have a run_number 0
        self.sim_duration = sim_duration
        self.warm_up_duration = warm_up_duration
        self.sim_start_date = sim_start_date

        self.random_seed = random_seed
        self.random_seed_sequence = SeedSequence(self.random_seed)

        self.print_debug_messages = print_debug_messages

        self.utils = Utils(
            master_seed=self.random_seed_sequence.spawn(1)[0],
            print_debug_messages=self.print_debug_messages
            )

        self.utils.setup_seeds()

        self.demand_increase_percent = demand_increase_percent

        # Option for sampling calls per day by season or quarter
        self.daily_calls_by_quarter_or_season = 'quarter'

        # Option to include/exclude ambulance service cases in addition to HEMS
        self.amb_data = amb_data
        #self.debug(f"Ambulance data values is {self.amb_data}")

        self.all_results_location = self.utils.ALL_RESULTS_CSV
        self.run_results_location = self.utils.RUN_RESULTS_CSV

        self.env = simpy.Environment()
        self.patient_counter = 0
        self.calls_today = 0
        self.new_day = pd.to_datetime("1900-01-01").date
        self.new_hour = -1

        self.hems_resources = HEMSAvailability(env=self.env,
                                               sim_start_date=sim_start_date,
                                               sim_duration=sim_duration,
                                               print_debug_messages=self.print_debug_messages,
                                               master_seed=self.random_seed_sequence,
                                               utility=self.utils
                                               )

        # Set up empty list to store results prior to conversion to dataframe
        self.results_list = []

        # Set up data frame to capture time points etc. during the simulation
        # We might not need all of these, but simpler to capture them all for now.

        # This might be better as a separate 'data collection' class of some sort.
        # I am thinking that each resource will need its own entry for any given
        # patient to account for dual response (ambulance service and HEMS)
        # stand downs etc.
        self.results_df = None

        self.inter_arrival_times_df = pd.read_csv('distribution_data/inter_arrival_times.csv')

        self.activity_duration_multiplier = activity_duration_multiplier

        # self.seeded_dists = self.utils.build_seeded_distributions(
        #     self.utils.activity_time_distr,
        #     master_seed=self.random_seed
        #     )

    def debug(self, message: str):
        if self.print_debug_messages:
            logging.debug(message)
            #print(message)


    def calls_per_hour(self, quarter: int) -> dict:
        """
            Function to return a dictionary of keys representing the hour of day
            and value representing the number of calls in that hour
        """

        # self.debug(f"There are going to be {self.calls_today} calls today and the current hour is {current_hour}")

        hourly_activity = self.utils.hourly_arrival_by_qtr_probs_df
        hourly_activity_for_qtr = hourly_activity[hourly_activity['quarter'] == quarter][['hour','proportion']]

        calls_in_hours = []

        for i in range(0, self.calls_today):
            hour = pd.Series.sample(hourly_activity_for_qtr['hour'], weights = hourly_activity_for_qtr['proportion'],
                                random_state=self.utils.rngs["calls_per_hour"]).iloc[0]
            #self.debug(f"Chosen hour is {hour}")
            calls_in_hours.append(hour)

        calls_in_hours.sort()

        #self.debug(calls_in_hours)

        d = {}

        hours, counts = np.unique(calls_in_hours, return_counts=True)

        for i in range(len(hours)):
            d[hours[i]] = counts[i]

        return d

    def predetermine_call_arrival(self, current_hour: int, quarter: int) -> list:
        """
            Function to determine the number of calls in
            24 hours and the inter-arrival rate for calls in that period
            Returns a list of times that should be used in the yield timeout statement
            in a patient generator

        """

        hourly_activity = self.utils.hourly_arrival_by_qtr_probs_df
        hourly_activity_for_qtr = hourly_activity[hourly_activity['quarter'] == quarter][['hour','proportion']]

        d = self.calls_per_hour(quarter)

        ia_time = []

        for index, row in hourly_activity_for_qtr.iterrows():
            hour = row['hour']
            if hour >= current_hour and hour in d:
                count = d[hour]
                if count > 0:
                    scale = 60 / count  # mean of exponential = 1 / rate
                    calc_ia_time = self.utils.rngs["predetermine_call_arrival"].exponential(scale=scale)
                    tmp_ia_time = ((hour - current_hour) * 60) + calc_ia_time
                    current_hour += floor(tmp_ia_time / 60)
                    ia_time.append(tmp_ia_time)

        return ia_time


    def generate_calls(self):
        """
            **Call generator**

            Generate calls (and patients) until current time equals sim_duration + warm_up_duration.
            This method calculates number of calls per day and then distributes them according to distributions determined
            from historic data

        """

        while self.env.now < (self.sim_duration + self.warm_up_duration):
            # Get current day of week and hour of day
            [dow, hod, weekday, month, qtr, current_dt] = self.utils.date_time_of_call(self.sim_start_date, self.env.now)

            if(self.new_hour != hod):
                self.new_hour = hod

                #self.debug("new hour")

            # If it is a new day, need to calculate how many calls
            # in the next 24 hours
            if(self.new_day != current_dt.date()):
                # self.debug("It's a new day")
                # self.debug(dow)
                # self.debug(f"{self.new_day} and {current_dt.date}")

                # Now have additional option of determining calls per day by quarter instead of season
                #self.calls_today = int(self.utils.inc_per_day(qtr, self.daily_calls_by_quarter_or_season) * (self.demand_increase_percent))

                # Try with actual sampled values instead
                self.calls_today = int(self.utils.inc_per_day_samples(qtr, self.daily_calls_by_quarter_or_season) * (self.demand_increase_percent))

                # self.debug(f"{current_dt.date()} There will be {self.calls_today} calls today")
                #self.debug(f"{current_dt.date()} There will be {self.calls_today} calls today")

                self.new_day = current_dt.date()

                ia_dict = {}
                ia_dict = self.calls_per_hour(qtr)

                #self.debug(ia_dict)

                # Also run scripts to check HEMS resources to see whether they are starting/finishing service
                yield self.env.process(self.hems_resources.daily_servicing_check(current_dt, hod, qtr))


            if self.calls_today > 0:
                if hod in ia_dict.keys():
                    minutes_elapsed = 0
                    for i in range(0, ia_dict[hod]):
                        if minutes_elapsed < 59:
                            # Determine remaining time and sample a wait time
                            remaining_minutes = 59 - minutes_elapsed
                            # wait_time = random.randint(0, remaining_minutes)
                            wait_time = int(self.utils.rngs["call_iat"].integers(0, remaining_minutes+1))

                            yield self.env.timeout(wait_time)
                            minutes_elapsed += wait_time

                            self.env.process(self.generate_patient(dow, hod, weekday, month, qtr, current_dt))

                            [dow, hod, weekday, month, qtr, current_dt] = self.utils.date_time_of_call(self.sim_start_date, self.env.now)


                next_hr = current_dt.floor('h') + pd.Timedelta('1h')
                yield self.env.timeout(math.ceil(pd.to_timedelta(next_hr - current_dt).total_seconds() / 60))

                    #self.debug('Out of loop')
            else:
                # Skip to tomorrow

                self.debug('Skip to tomorrow')

                [dow, hod, weekday, month, qtr, current_dt] = self.utils.date_time_of_call(self.sim_start_date, self.env.now)
                next_day = current_dt.floor('d') + pd.Timedelta(days=1)
                self.debug("next day is {next_day}")
                yield self.env.timeout(math.ceil(pd.to_timedelta(next_hr - current_dt).total_seconds() / 60))


    def generate_patient(self, dow, hod, weekday, month, qtr, current_dt):

        self.patient_counter += 1

        # Create a new caller/patient
        pt = Patient(self.patient_counter)

        # Update patient instance with time-based values so the current time is known
        pt.day = dow
        pt.hour = hod
        pt.weekday = weekday
        pt.month = month
        pt.qtr = qtr
        pt.current_dt = current_dt

        #self.debug(f"Patient {pt.id} incident date: {pt.current_dt}")

        # Update patient instance with age, sex, AMPDS card, whether they are a HEMS' patient and if so, the HEMS result,
        pt.ampds_card = self.utils.ampds_code_selection(pt.hour)
        #self.debug(f"AMPDS card is {pt.ampds_card}")
        pt.age = self.utils.age_sampling(pt.ampds_card, 115)
        pt.sex = self.utils.sex_selection(pt.ampds_card)
        hems_cc_or_ec = self.utils.care_category_selection(pt.ampds_card)
        pt.hems_cc_or_ec = hems_cc_or_ec
        self.debug(f"{pt.current_dt}: {pt.id} Pt allocated to {pt.hems_cc_or_ec} from AMPDS {pt.ampds_card}")

        pt.pt_outcome = self.utils.pt_outcome_selection(pt.hems_cc_or_ec, int(qtr))
        self.debug(f"{pt.current_dt}: {pt.id} Pt allocated to patient outcome: {pt.pt_outcome}")

        self.add_patient_result_row(pt, "arrival", "arrival_departure")

        if self.amb_data:
            # TODO: We'll need the logic to decide whether it is an ambulance or HEMS case
            # if ambulance data is being collected too.
            self.debug("Ambulance case")
            pt.hems_case = 1 if self.utils.rngs["hems_case"].uniform(0, 1) <= 0.5 else pt.hems_case == 0
        else:
            pt.hems_case = 1

        if pt.hems_case == 1:
            #self.debug(f"Going to callsign_group_selection with hour {pt.hour} and AMPDS {pt.ampds_card}")
            # pt.hems_pref_callsign_group = self.utils.callsign_group_selection(pt.ampds_card)
            pt.hems_pref_callsign_group = self.utils.callsign_group_selection(pt.hems_cc_or_ec)
            #self.debug(f"Callsign is {pt.hems_pref_callsign_group}")

            # Some % of 'REG' calls have a helicopter benefit
            # Default to y for all patients
            # NOTE - this may not be strictly true! Some EC/CC may not have a direct heli benefit
            helicopter_benefit = 'y'
            # Update for REG patients based on historically observed patterns
            with open("distribution_data/proportion_jobs_heli_benefit.txt", "r") as file:
                expected_prop_heli_benefit_jobs = float(file.read().strip())

            if pt.hems_cc_or_ec == 'REG':
                helicopter_benefit = 'y' if self.utils.rngs["helicopter_benefit_from_reg"].uniform(0, 1) <= expected_prop_heli_benefit_jobs else 'n'

            pt.hems_helicopter_benefit = helicopter_benefit
            self.add_patient_result_row(pt, pt.hems_cc_or_ec, "patient_care_category")
            self.add_patient_result_row(pt, pt.hems_helicopter_benefit, "patient_helicopter_benefit")

            #pt.hems_pref_vehicle_type = self.utils.vehicle_type_selection(pt.hems_pref_callsign_group)
            pt.hems_pref_vehicle_type = 'helicopter'
            #pt.hems_pref_callsign_group = '70'
            #pt.hems_helicopter_benefit = 'y'

            self.add_patient_result_row(pt, pt.hems_pref_callsign_group, "resource_preferred_resource_group")
            self.add_patient_result_row(pt, pt.hems_pref_vehicle_type, "resource_preferred_vehicle_type")

            if pt.hems_cc_or_ec == 'REG':
                # Separate (basically the old way of doing things)
                # function to determine HEMS resource based on callsign group, vehicle type and yearly quarter
                hems_res_list: list[HEMS|None, str, HEMS|None] = yield self.hems_resources.allocate_regular_resource(pt)
            else:
                hems_res_list: list[HEMS|None, str, HEMS|None] = yield self.hems_resources.allocate_resource(pt)
                #self.debug(hems_res_list)

            hems_allocation = hems_res_list[0]

            # This will either contain the other resource in a callsign_group and HEMS category (EC/CC) or None
            hems_group_resource_allocation = hems_res_list[2]

            self.add_patient_result_row(pt, hems_res_list[1], "resource_preferred_outcome")

            if hems_allocation is not None:
                #self.debug(f"allocated {hems_allocation.callsign}")

                self.env.process(self.patient_journey(hems_allocation, pt, hems_group_resource_allocation))
            else:
                #self.debug(f"{pt.current_dt} No HEMS resource available - non-DAAT land crew sent")
                self.env.process(self.patient_journey(None, pt, None))


    def patient_journey(self, hems_res: HEMS|None, patient: Patient, secondary_hems_res: HEMS|None):
        """
            Send patient on their journey!
        """

        #self.debug(f"Patient journey triggered for {patient.id}")
        #self.debug(f"patient journey Time: {self.env.now}")
        # self.debug(hems_res.callsign)

        try:
            patient_enters_sim = self.env.now

            # Note that 'add_patient_result_row' does its own check for whether it's currently within
            # the warm-up period, so this does not need to be checked manually when adding result
            # rows in this section
            # not_in_warm_up_period = False if self.env.now < self.warm_up_duration else True

            # Add variables for quick determination of whether certain parts of the process should
            # be included per case
            hems_case = True if patient.hems_case == 1 else False
            hems_avail = True if hems_res is not None else False

            if not hems_avail:
                self.debug(f"Patient {patient.id}: No HEMS available")
                self.add_patient_result_row(patient, "No HEMS available", "queue")
                self.add_patient_result_row(patient, "No Resource Available", "resource_request_outcome")

            # if hems_avail
            else:

                # Record selected vehicle type and callsign group in patient object
                patient.hems_vehicle_type = hems_res.vehicle_type
                patient.hems_registration = hems_res.registration
                patient.callsign = hems_res.callsign

                self.add_patient_result_row(patient, patient.hems_vehicle_type, "selected_vehicle_type")
                self.add_patient_result_row(patient, patient.hems_callsign_group, "selected_callsign_group")

                #patient.hems_result = self.utils.hems_result_by_callsign_group_and_vehicle_type_selection(patient.hems_callsign_group, patient.hems_vehicle_type)
                #self.debug(f"{patient.hems_cc_or_ec} and {patient.hems_helicopter_benefit}")
                #patient.hems_result = self.utils.hems_result_by_care_category_and_helicopter_benefit_selection(patient.hems_cc_or_ec, patient.hems_helicopter_benefit)

                # Determine outcome
                # If we know a resource has been allocated, we can determine the output from historical patterns
                if hems_res:
                    patient.hems_result = self.utils.hems_results_by_patient_outcome_and_time_of_day_and_quarter_and_vehicle_type_and_callsign_group_probs(
                        patient.pt_outcome,
                        int(patient.qtr),
                        patient.hems_vehicle_type,
                        patient.hems_callsign_group,
                        int(patient.hour)
                    )
                else:
                # Default to what is recorded when no resource sent
                    patient.hems_result = "Unknown"

                self.debug(f"{patient.current_dt}: PT_ID:{patient.id} Pt allocated to HEMS result: {patient.hems_result}")
                self.add_patient_result_row(patient, "HEMS Resource Available", "resource_request_outcome")
                self.add_patient_result_row(patient, hems_res.callsign, "resource_use")
                self.debug(f"{patient.current_dt} Patient {patient.id} (preferred callsign group {patient.hems_pref_callsign_group}, preferred resource type {patient.hems_pref_vehicle_type}) sent resource {hems_res.callsign}")
                self.add_patient_result_row(patient, hems_res.callsign, "callsign_group_resource_use")

                # Check if HEMS result indicates that resource stood down before going mobile or en route
                no_HEMS_at_scene = True if patient.hems_result in ["Stand Down"] else False
                # Check if HEMS result indicates no leaving scene/at hospital times
                no_HEMS_hospital = (
                    True if patient.hems_result
                    in ["Stand Down",  "Landed but no patient contact", "Patient Treated but not conveyed by HEMS"]
                    else False
                    )

            #self.debug('Inside hems_avail')
            if self.amb_data:
                ambulance = Ambulance()

            patient.time_in_sim = self.env.now - patient_enters_sim

            if hems_case and hems_avail:
                #self.debug(f"Adding result row with csg {patient.hems_callsign_group}")
                self.add_patient_result_row(patient, "HEMS call start", "queue")

            if self.amb_data:
                self.add_patient_result_row(patient,  "AMB call start", "queue")

            # Allocation time --------------
            # Allocation will always take place if a resource is found

            if hems_case and hems_avail:
                # Calculate min and max permitted times.
                allocation_time = (
                    self.utils.activity_time(patient.hems_vehicle_type, 'time_allocation')
                    * self.activity_duration_multiplier
                    )
                self.add_patient_result_row(patient, allocation_time, 'time_allocation')
                self.add_patient_result_row(patient, "HEMS allocated to call", "queue")
                #self.debug(f"Vehicle type {patient.hems_vehicle_type} and allocation time is {allocation_time}")
                yield self.env.timeout(allocation_time)

            if self.amb_data:
                #self.debug('Ambulance allocation time')
                yield self.env.timeout(180)

            patient.time_in_sim = self.env.now - patient_enters_sim

            # Mobilisation time ---------------

            # Calculate mobile to time at scene (or stood down before)
            if hems_case and hems_avail:
                mobile_time = (
                    self.utils.activity_time(patient.hems_vehicle_type, 'time_mobile')
                    * self.activity_duration_multiplier
                    )
                self.add_patient_result_row(patient, mobile_time, 'time_mobile')
                self.add_patient_result_row(patient,  "HEMS mobile", "queue")
                yield self.env.timeout(mobile_time)

            if self.amb_data:
                # Determine allocation time for ambulance
                # Yield until done
                #self.debug('Ambulance time to going mobile')
                self.debug('Ambulance mobile')
                yield self.env.timeout(1)

            patient.time_in_sim = self.env.now - patient_enters_sim

            # On scene ---------------

            if hems_case and hems_avail and not no_HEMS_at_scene:
                tts_time = (
                    self.utils.activity_time(patient.hems_vehicle_type, 'time_to_scene')
                    * self.activity_duration_multiplier
                    )
                self.add_patient_result_row(patient, tts_time, "time_to_scene")
                self.add_patient_result_row(patient,  "HEMS on scene", "queue")
                yield self.env.timeout(tts_time)

            if self.amb_data:
                    # Determine allocation time for ambulance
                    # Yield until done
                    #self.debug('Ambulance time to scene')
                    yield self.env.timeout(20)

            patient.time_in_sim = self.env.now - patient_enters_sim

            if self.amb_data:
                self.debug('Ambulance stand down en route')


            # Leaving scene ------------

            if hems_case and hems_avail and not no_HEMS_at_scene:
                tos_time = (
                    self.utils.activity_time(patient.hems_vehicle_type, 'time_on_scene')
                    * self.activity_duration_multiplier
                    )
                self.add_patient_result_row(patient, tos_time, "time_on_scene")
                yield self.env.timeout(tos_time)

                if no_HEMS_hospital:
                    self.add_patient_result_row(patient, f"HEMS {patient.hems_result.lower()}", "queue")
                else:
                    self.add_patient_result_row(patient, "HEMS leaving scene", "queue")

            if self.amb_data:
                #self.debug('Ambulance on scene duration')
                yield self.env.timeout(120)
                self.debug('Ambulance leaving scene time')

            patient.time_in_sim = self.env.now - patient_enters_sim

            # Arrived destination time ------------

            if hems_case and hems_avail and not no_HEMS_hospital:
                travel_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_to_hospital') * self.activity_duration_multiplier
                self.add_patient_result_row(patient, travel_time, 'time_to_hospital')
                self.add_patient_result_row(patient, "HEMS arrived destination", "queue")
                yield self.env.timeout(travel_time)

            if self.amb_data:
                #self.debug('Ambulance travel time')
                yield self.env.timeout(30)
                self.debug('Ambulance at destination time')

            patient.time_in_sim = self.env.now - patient_enters_sim

            # Handover time ---------------

            # Not currently available

            # Clear time ------------

            if hems_case and hems_avail:
                clear_time = (
                    self.utils.activity_time(patient.hems_vehicle_type, 'time_to_clear')
                    * self.activity_duration_multiplier
                    )
                self.add_patient_result_row(patient, clear_time, 'time_to_clear')
                self.add_patient_result_row(patient, "HEMS clear", "queue")
                yield self.env.timeout(clear_time)

            if self.amb_data:
                #self.debug('Ambulance clear time')
                yield self.env.timeout(60)

            patient.time_in_sim = self.env.now - patient_enters_sim

            if self.amb_data:
                self.debug('Ambulance clear time')

            #self.debug(f"Depart for patient {patient.id} on run {self.run_number}")

            self.add_patient_result_row(patient, "depart", "arrival_departure")
        finally:
            # Always return the resource at the end of the patient journey.
            if hems_res is not None:
                self.hems_resources.return_resource(hems_res, secondary_hems_res)
                self.add_patient_result_row(patient, hems_res.callsign, "resource_use_end")


    def add_patient_result_row(self,
                               patient: Patient,
                               time_type: str,
                               event_type: str,
                               **kwargs) -> None :
        """
            Convenience function to create a row of data for the results table

        """

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
            "registration"      : patient.hems_registration, # registration of allocated/attending vehicle
            "callsign"          : patient.callsign, # callsign of allocated/attending vehicle
            "callsign_group"    : patient.hems_callsign_group, # callsign group of allocated/attending vehicle
            "vehicle_type"      : patient.hems_vehicle_type, # vehicle type (car/helicopter) of allocated/attending vehicle
            "hems_res_category" : patient.hems_category,
            "ampds_card"        : patient.ampds_card,
            "age"               : patient.age,
            "sex"               : patient.sex,
            "care_cat"          : patient.hems_cc_or_ec,
            "heli_benefit"      : patient.hems_helicopter_benefit,
            "hems_result"       : patient.hems_result,
            "outcome"           : patient.pt_outcome,
            "hems_reg"          : patient.hems_registration
        }

        #self.debug(results)

        # Add any additional items passed in **kwargs
        for key, value in kwargs.items():
             results[key] = value

        if self.env.now >= self.warm_up_duration:
            self.store_patient_results(results)


    def store_patient_results(self, results: dict) -> None:
        """
            Adds a row of data to the Class' `result_df` dataframe
        """

        self.results_list.append(results)


    def convert_results_to_df(self, results: dict) -> None:
        self.results_df = pd.DataFrame(results)
        try:
            self.results_df.set_index("P_ID", inplace=True)
        # TODO - Improve error handling here
        except KeyError:
            pass

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
        self.debug(f"HEMS class initialised with the following: run {self.run_number}, duration {self.sim_duration}, warm-up {self.warm_up_duration}, start date {self.sim_start_date}, demand increase multiplier {self.demand_increase_percent}, activity duration multiplier {self.activity_duration_multiplier}")

        # Start entity generators
        self.env.process(self.generate_calls())

        # Run simulation
        self.env.run(until=(self.sim_duration + self.warm_up_duration))

        # Convert results to a dataframe
        self.convert_results_to_df(results=self.results_list)

        # Write run results to file
        self.write_all_results()
        self.write_run_results()
