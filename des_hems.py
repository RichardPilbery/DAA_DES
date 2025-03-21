import math
import os, simpy
from random import expovariate, uniform
from sim_tools.time_dependent import NSPPThinning
import pandas as pd
# from random import expovariate
from utils import Utils
from class_patient import Patient
from class_hems_availability import HEMSAvailability
from class_hems import HEMS
from class_ambulance import Ambulance
from datetime import timedelta
import warnings
import numpy as np
from math import floor
warnings.filterwarnings("ignore", category=RuntimeWarning)

class DES_HEMS:
    """
        # The DES_HEMS class

        This class contains everything require to undertake a single DES run for multiple patients
        over the specified duration.

        NOTE: The unit of sim is minutes

    """

    def __init__(self,
                run_number: int, sim_duration: int, warm_up_duration: int, sim_start_date: str,
                amb_data: bool, demand_increase_percent: float, activity_duration_multiplier: float):

        self.run_number = run_number + 1 # Add 1 so we don't have a run_number 0
        self.sim_duration = sim_duration
        self.warm_up_duration = warm_up_duration
        self.sim_start_date = sim_start_date

        self.demand_increase_percent = demand_increase_percent

        # Option to include/exclude ambulance service cases in addition to HEMS
        self.amb_data = amb_data
        #print(f"Ambulance data values is {self.amb_data}")

        self.utils = Utils()

        self.all_results_location = self.utils.ALL_RESULTS_CSV
        self.run_results_location = self.utils.RUN_RESULTS_CSV

        self.env = simpy.Environment()
        self.patient_counter = 0
        self.calls_today = 0
        self.new_day = pd.to_datetime("1900-01-01").date

        self.hems_resources = HEMSAvailability(self.env, sim_start_date, sim_duration)

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


    def calc_interarrival_time(self, hour: int, qtr: int, NSPPThin = False):
        """
            Convenience function to return the time between incidents
            using either NSPPThinning or sampling the exponential
            distribution

            Arrivals distribution using NSPPThinning
            HSMA example: https://hsma-programme.github.io/hsma6_des_book/modelling_variable_arrival_rates.html

        """

        if NSPPThin:
           # Determine the inter-arrival time usiong NSPPThinning
            arrivals_dist = NSPPThinning(
                data = self.inter_arrival_times_df[
                    (self.inter_arrival_times_df['quarter'] == qtr)
                ],
                random_seed1 = self.run_number * 112,
                random_seed2 = self.run_number * 999
            )

            return arrivals_dist.sample(hour)

        else:
            # Or just regular exponential distrib.
            inter_time = self.utils.inter_arrival_rate(hour, qtr)
            return expovariate(1.0 / inter_time)


    def calls_per_hour(self, quarter: int) -> dict:
        """
            Function to return a dictionary of keys representing the hour of day
            and value representing the number of calls in that hour
        """

        #print(f"There are going to be {self.calls_today} calls today and the current hour is {current_hour}")

        hourly_activity = self.utils.hourly_arrival_by_qtr_probs_df
        hourly_activity_for_qtr = hourly_activity[hourly_activity['quarter'] == quarter][['hour','proportion']]

        calls_in_hours = []

        for i in range(0, self.calls_today):
            hour = pd.Series.sample(hourly_activity_for_qtr['hour'], weights = hourly_activity_for_qtr['proportion']).iloc[0]
            #print(f"Chosen hour is {hour}")
            calls_in_hours.append(hour)

        calls_in_hours.sort()

        #print(calls_in_hours)

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
            if row['hour'] >= current_hour:
                if row['hour'] in d.keys():
                    calc_ia_time = expovariate(int(d[row['hour']]) / 60)
                    tmp_ia_time = ((row['hour']-current_hour) * 60) + calc_ia_time
                    # Update current hour
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

            # If it is a new day, need to calculate how many calls
            # in the next 24 hours
            if(self.new_day != current_dt.date()):
                # print("It's a new day")
                # print(dow)
                # print(f"{self.new_day} and {current_dt.date}")
                self.calls_today = int(self.utils.inc_per_day(qtr) * (self.demand_increase_percent))

                #print(f"{current_dt.date()} There will be {self.calls_today} calls today")

                self.new_day = current_dt.date()

                ia_dict = {}
                ia_dict = self.calls_per_hour(qtr)

                #print(ia_dict)

                # Also run scripts to check HEMS resources to see whether they are starting/finishing service
                self.hems_resources.daily_servicing_check(current_dt)

            if self.calls_today > 0:
                # Work out how long until next incident
                #print(ia_dict.keys())
                if hod in ia_dict.keys():
                    #print(f"Hour of day is {hod} and there are {ia_dict[hod]} patients to create")
                    for i in range(0, ia_dict[hod]):
                        #print(f"Creating new patient at {current_dt}")
                        self.env.process(self.generate_patient(dow, hod, weekday, month, qtr, current_dt))
                        # Might need to determine spread of jobs during any given hour.
                        yield self.env.timeout(5) # Wait 5 minutes until the next allocation
                        [dow, hod, weekday, month, qtr, current_dt] = self.utils.date_time_of_call(self.sim_start_date, self.env.now)

                next_hr = current_dt.floor('h') + pd.Timedelta('1h')
                yield self.env.timeout(math.ceil(pd.to_timedelta(next_hr - current_dt).total_seconds() / 60))

                    #print('Out of loop')
            else:
                # Skip to tomorrow

                print('Skip to tomorrow')

                [dow, hod, weekday, month, qtr, current_dt] = self.utils.date_time_of_call(self.sim_start_date, self.env.now)
                next_day = current_dt.floor('d') + pd.Timedelta(days=1)
                print("next day is {next_day}")
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

        #print(f"Patient {pt.id} incident date: {pt.current_dt}")

        # Update patient instance with age, sex, AMPDS card, whether they are a HEMS' patient and if so, the HEMS result,
        pt.ampds_card = self.utils.ampds_code_selection(pt.hour)
        #print(f"AMPDS card is {pt.ampds_card}")
        pt.age = self.utils.age_sampling(pt.ampds_card, 115)
        pt.sex = self.utils.sex_selection(pt.ampds_card)
        pt.hems_cc_or_ec = self.utils.care_category_selection(pt.ampds_card)
        #print(f"Pt allocated to {pt.hems_cc_or_ec} from AMPDS {pt.ampds_card}")

        not_in_warm_up_period = False if self.env.now < self.warm_up_duration else True

        if self.amb_data:
            # TODO: We'll need the logic to decide whether it is an ambulance or HEMS case
            # if ambulance data is being collected too.
            print("Ambulance case")
            pt.hems_case = 1 if uniform(0, 1) <= 0.5 else pt.hems_case == 0
        else:
            pt.hems_case = 1

        if pt.hems_case == 1:
            #print(f"Going to callsign_group_selection with hour {pt.hour} and AMPDS {pt.ampds_card}")
            pt.hems_pref_callsign_group = self.utils.callsign_group_selection(int(pt.hour), pt.ampds_card)
            #print(f"Callsign is {pt.hems_pref_callsign_group}")

            pt.hems_pref_vehicle_type = self.utils.vehicle_type_selection(pt.month, pt.hems_pref_callsign_group)
            #print(f"Vehicle type is {pt.hems_pref_vehicle_type}")

            self.add_patient_result_row(pt, pt.hems_pref_callsign_group, "resource_preferred_resource_group")
            self.add_patient_result_row(pt, pt.hems_pref_vehicle_type, "resource_preferred_vehicle_type")

            hems_res_list: list[HEMS|None, int, bool, HEMS|None] = yield self.hems_resources.allocate_resource(pt)
            hems_allocation = hems_res_list[0]

            # This will either contain the other resource in a callsign_group or None
            hems_group_resource_allocation = hems_res_list[3]

            if not_in_warm_up_period:
                msg = "No resource in group available"
                if hems_res_list[1] == 1:
                    msg = "Preferred resource available and allocated"
                elif hems_res_list[1] == 2:
                    msg = "Preferred resource not available but other resource in same group allocated"

                self.add_patient_result_row(pt, msg, "resource_preferred_outcome")

                if hems_allocation != None:
                    if hems_res_list[2]:
                        self.add_patient_result_row(pt, f"{'H' if pt.hems_pref_vehicle_type == 'helicopter' else 'CC'}{pt.hems_pref_callsign_group}", "resource_preferred_service")

            # if hems_res_list[2]:
            #     print(f"Back from allocate resource with {hems_res_list[1]} and {hems_res_list[2]}")

            if hems_allocation != None:
                #print(f"allocated {hems_allocation.callsign}")
                self.add_patient_result_row(pt, hems_allocation.callsign, "resource_use")

                if hems_group_resource_allocation != None:
                    self.add_patient_result_row(pt, hems_group_resource_allocation.callsign, "callsign_group_resource_use")

                self.env.process(self.patient_journey(hems_allocation, pt, hems_group_resource_allocation))
            else:
                #print("No HEMS resource available - non-DAAT land crew sent")
                self.env.process(self.patient_journey(None, pt, None))


    def patient_journey(self, hems_res: HEMS|None, patient: Patient, secondary_hems_res: HEMS|None):
        """
            Send patient on their journey!
        """

        #print(f"Patient journey triggered for {patient.id}")
        #print(f"patient journey Time: {self.env.now}")
        # print(hems_res.callsign)

        patient_enters_sim = self.env.now

        not_in_warm_up_period = False if self.env.now < self.warm_up_duration else True

        if not_in_warm_up_period:
            #print(f"Arrival for patient {patient.id} on run {self.run_number}")
            self.add_patient_result_row(patient, "arrival", "arrival_departure")

        hems_avail = True if hems_res != None else False

        if hems_res != None:

            patient.hems_callsign_group = hems_res.callsign_group
            #print(f"Patient csg is {patient.hems_callsign_group}")
            patient.hems_vehicle_type = hems_res.vehicle_type

            patient.hems_result = self.utils.hems_result_by_callsign_group_and_vehicle_type_selection(patient.hems_callsign_group, patient.hems_vehicle_type)

            # Check if HEMS result indicates that resource stodd down before going mobile or en route
            no_HEMS_at_scene = True if patient.hems_result in ["Stand Down Before Mobile", "Stand Down En Route"] else False

            # Check if HEMS result indicates no leaving scene/at hospital times
            no_HEMS_hospital = True if patient.hems_result in ["Stand Down Before Mobile", "Stand Down En Route", "Landed but no patient contact", "Patient Treated (not conveyed)"] else False

            patient.pt_outcome = self.utils.pt_outcome_selection(patient.hems_result)

            #print(f"Patient outcome is {patient.pt_outcome}")

        else:
            #print("No HEMS available")
            self.add_patient_result_row(patient, "No HEMS available", "queue")
            #self.add_patient_result_row(patient, "depart", "arrival_departure")

        #print('Inside hems_avail')
        if self.amb_data:
            ambulance = Ambulance()

        # Add boolean to determine whether the patient is still within the simulation warm-up
        # period. If so, then we will not record the patient progress
        # SR NOTE: I changed this from strictly less than to <= as it was causing odd behaviour
        # if the first call came in at the start of the simulation in a scenario with 0 warm up
        # time. Alternative could be reverting it to stricly less check but then adding an or check
        # leading to not_in_warm_up_period being True if the warm-up duration is exactly 0.
        # SR NOTE 2 - there is also a check implemented directly in the method add_patient_result_row
        # so I'm not sure the additional check in this instance is actually necessary
        not_in_warm_up_period = False if self.env.now <= self.warm_up_duration else True

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if patient.hems_case == 1 and hems_avail:
                #print(f"Adding result row with csg {patient.hems_callsign_group}")
                self.add_patient_result_row(patient, "HEMS call start", "queue")

            if self.amb_data:
                self.add_patient_result_row(patient,  "AMB call start", "queue")

        # Allocation time --------------

        if patient.hems_case == 1 and hems_avail:
            # Calculate min and max permitted times.
            allocation_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_allocation') * self.activity_duration_multiplier
            #print(f"Vehicle type {patient.hems_vehicle_type} and allocation time is {allocation_time}")
            yield self.env.timeout(allocation_time)

        if self.amb_data:
                #print('Ambulance allocation time')
                yield self.env.timeout(180)

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if patient.hems_case == 1 and hems_avail:
                if patient.hems_result != "Stand Down Before Mobile":
                    self.add_patient_result_row(patient, "HEMS allocated to call", "queue")
                else:
                    self.add_patient_result_row(patient, "HEMS stand down before mobile", "queue")

            if self.amb_data:
                #print('Ambulance time to allocation')
                yield self.env.timeout(5)


        # Mobilisation time ---------------

        # Calculate mobile to time at scene (or stood down before)
        if patient.hems_case == 1 and hems_avail:
            mobile_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_mobile') * self.activity_duration_multiplier
            yield self.env.timeout(mobile_time)

        if self.amb_data:
                # Determine allocation time for ambulance
                # Yield until done
                #print('Ambulance time to going mobile')
                yield self.env.timeout(1)

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if patient.hems_case == 1 and hems_avail:
                self.add_patient_result_row(patient,  "HEMS mobile", "queue")

            if self.amb_data:
                print('Ambulance mobile')

        # On scene ---------------

        if (patient.hems_case == 1 and hems_avail) and not no_HEMS_at_scene:
            tts_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_to_scene') * self.activity_duration_multiplier
            yield self.env.timeout(tts_time)

        if self.amb_data:
                # Determine allocation time for ambulance
                # Yield until done
                #print('Ambulance time to scene')
                yield self.env.timeout(20)

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if (patient.hems_case == 1 and hems_avail):
                if patient.hems_result != "Stand Down En Route":
                    self.add_patient_result_row(patient,  "HEMS on scene", "queue")
                else:
                    self.add_patient_result_row(patient,  "HEMS stand down en route","queue")

            if self.amb_data:
                print('Ambulance stand down en route')


        # Leaving scene ------------

        if (patient.hems_case == 1 and hems_avail) and not no_HEMS_at_scene:
            tos_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_on_scene') * self.activity_duration_multiplier
            yield self.env.timeout(tos_time)

        if self.amb_data:
            #print('Ambulance on scene duration')
            yield self.env.timeout(120)

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if (patient.hems_case == 1 and hems_avail) and not no_HEMS_at_scene:
                if no_HEMS_hospital == False:
                    self.add_patient_result_row(patient, "HEMS leaving scene", "queue")
                else:
                    self.add_patient_result_row(patient, f"HEMS {patient.hems_result.lower()}", "queue")

            if self.amb_data:
                print('Ambulance leaving scene time')


        # Arrived destination time ------------

        if (patient.hems_case == 1 and hems_avail) and no_HEMS_hospital == False:
            travel_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_to_hospital') * self.activity_duration_multiplier
            yield self.env.timeout(travel_time)

        if self.amb_data:
            #print('Ambulance travel time')
            yield self.env.timeout(30)

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if (patient.hems_case == 1 and hems_avail) and no_HEMS_hospital == False:

                self.add_patient_result_row(patient, "HEMS arrived destination", "queue")

        if self.amb_data:
            print('Ambulance at destination time')


        # Handover time ---------------

        # Not currently available

        # Clear time ------------

        if (patient.hems_case == 1 and hems_avail):
            clear_time = self.utils.activity_time(patient.hems_vehicle_type, 'time_to_clear') * self.activity_duration_multiplier
            yield self.env.timeout(clear_time)

            if hems_res != None:
                self.hems_resources.return_resource(hems_res, secondary_hems_res)
                self.add_patient_result_row(patient, hems_res.callsign, "resource_use_end")

        if self.amb_data:
            #print('Ambulance clear time')
            yield self.env.timeout(60)

        patient.time_in_sim = self.env.now - patient_enters_sim

        if not_in_warm_up_period:
            if patient.hems_case == 1 and hems_avail:
                self.add_patient_result_row(patient,"HEMS clear", "queue")

            if self.amb_data:
                print('Ambulance clear time')

        if not_in_warm_up_period:
            #print(f"Depart for patient {patient.id} on run {self.run_number}")
            self.add_patient_result_row(patient, "depart", "arrival_departure")



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
            "callsign_group"    : patient.hems_callsign_group,
            "vehicle_type"      : patient.hems_vehicle_type,
            "ampds_card"        : patient.ampds_card,
            "age"               : patient.age,
            "sex"               : patient.sex,
            "care_cat"          : patient.hems_cc_or_ec,
            "hems_result"       : patient.hems_result,
            "outcome"           : patient.pt_outcome
        }

        #print(results)

        # Add any additional items passed in **kwargs
        for key, value in kwargs.items():
             results[key] = value

        # SR NOTE: I changed this from strictly less than to <= as it was causing odd behaviour
        # if the first call came in at the start of the simulation in a scenario with 0 warm up
        # time. Alternative could be reverting it to stricly less check but then adding an or check
        # leading to not_in_warm_up_period being True if the warm-up duration is exactly 0.
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
        print(f"HEMS class initialised with the following: run {self.run_number}, duration {self.sim_duration}, warm-up {self.warm_up_duration}, start date {self.sim_start_date}, demand increase multiplier {self.demand_increase_percent}, activity duration multiplier {self.activity_duration_multiplier}")

        # Start entity generators
        self.env.process(self.generate_calls())

        # Run simulation
        self.env.run(until=(self.sim_duration + self.warm_up_duration))

        # Convert results to a dataframe
        self.convert_results_to_df(results=self.results_list)

        # Write run results to file
        self.write_all_results()
        self.write_run_results()
