import simpy
from random import expovariate
from utils import Utils
from class_patient import Patient

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

        self.env = simpy.Environment()
        self.patient_counter = 0

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
                    #self.env.process(self.patient_journey(pt))
                    
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

                    # We need a lookup table for mean inter arrival times. A tabulate version of Figure 1 from
                    # the UoR report would be a good starter for 10...
                    inter_time = 10
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
        # patient_enters_sim = self.env.now
    
        # yield self.env.process(60)

        # patient. = self.env.now - patient_enters_sim
        print('Patient is on journey')


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
        # self.write_all_results() 