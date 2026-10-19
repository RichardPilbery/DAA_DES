from math import floor


class Patient:
    def __init__(self, p_id: int):
        # Unique ID for patient
        self.id = p_id

        # Incident specific data

        # Allocate triage code to patient
        # This might vary by time of day/year etc. so lookup table might be more complicated
        # or could use a simple regression?
        self.ampds_card = ""

        # Likely to be postcode sector i.e. BS1 9 of BS1 9HJ
        self.postcode = ""

        # incident location - latitude (2 decimal places)
        self.lat = ""
        self.long = ""

        # ? Include incident times here ?

        # Demographic data

        self.age = 0
        self.sex = "female"

        # print(f"AMPDS code is {self.triage_code} and prop_female is {prop_female} and sex is {self.sex}")

        # Keep track of cumulatative time
        self.time_in_sim = 0

        # Variables to keep track of time as the patient progresses through the model
        # These are updated at every step
        self.hour = 0
        self.day = "Mon"
        self.month = 1
        self.qtr = 1
        self.weekday = "weekday"
        self.current_dt = None  # TODO: Initialise as a dt?

        # HEMS/critical care specific items
        self.time_to_cc = 0
        self.cc_conveyed = 0
        self.cc_flown = 0
        self.cc_travelled_with = 0
        # Binary flag to indicate whether it is a 'HEMS job' whether they attend or not
        self.hems_case = -1
        # Binary flag to indicate whether patient cared for by HEMS or not
        self.hems = -1
        self.hems_result = ""
        self.hems_pref_vehicle_type = ""
        self.hems_pref_callsign_group = ""
        # Strings relating to allocated vehicle
        self.hems_vehicle_type = ""
        self.hems_callsign_group = ""
        self.hems_registration = ""
        self.hems_category = ""  # allocated vehicle registration
        self.callsign = ""  # allocated vehicle callsign

        self.pt_outcome = ""

        # Category to denote need for EC/CC or REG (regular) care
        self.hems_cc_or_ec = "REG"

        # Is the helicopter beneficial for this job?
        self.hems_helicopter_benefit = ""

        # Critical care desk staffed
        self.cc_desk = 0

        # Despatched by EOC outside of criteria
        # 0 = no, 1 = P1, 2 = P2, 3 = P3 maybe?
        self.dispatcher_intervention = 0

        self.time_to_first_respone = 0
