
from math import floor
from random import betavariate, uniform, choices
from utils import Utils

class Patient:

    def __init__(self, p_id: int):

        # Unique ID for patient
        self.id = p_id 

        # Incident specific data

        # Allocate triage code to patient
        # This might vary by time of day/year etc. so lookup table might be more complicated
        # or could use a simple regression?
        self.triage_code = choices(Utils.TRIAGE_CODE_DISTR.index, weights=Utils.TRIAGE_CODE_DISTR["prob"])[0]

        # Likely to be postcode sector i.e. BS1 9 of BS1 9HJ
        self.postcode = "" 

        # incident location - latitude (2 decimal places)
        self.lat = ""
        self.long = ""

        # ? Include incident times here ?


        # Demographic data

        # Age selection based on distribution (although might need to take account of incident type)
        self.age = floor(betavariate(0.733, 2.82)*100) # Just an example will need updating

        # Calculate patient sex based on AMPDS code
        prop_female = Utils.TRIAGE_CODE_DISTR.sex_female[Utils.TRIAGE_CODE_DISTR.index == self.triage_code].iloc[0]
        self.sex =  "female" if uniform(0, 1) < prop_female else "male"

        #print(f"AMPDS code is {self.triage_code} and prop_female is {prop_female} and sex is {self.sex}")

        # Keep track of cumulatative time
        self.time_in_sim = 0

        # Variables to keep track of time as the patient progresses through the model
        # These are updated at every step
        self.hour = 0
        self.day = "Mon"
        self.month = 1
        self.weekday = "weekday"

        # HEMS/critical care specific items
        self.time_to_cc = 0
        self.cc_conveyed = 0
        self.cc_flown = 0
        self.cc_travelled_with = 0
        # Binary flag to indicate whether patient cared for by HEMS or not
        self.hems = 1
        # Critical care desk staffed
        self.cc_desk = 0

        # Despatched by EOC outside of criteria
        # 0 = no, 1 = P1, 2 = P2, 3 = P3 maybe?
        self.dispatcher_intervention = 0

        self.time_to_first_respone = 0

        # Flag to indicate whether incident has completed
        self.incident_completed = 0