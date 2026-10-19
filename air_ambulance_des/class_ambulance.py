from air_ambulance_des.utils import Utils
import pandas as pd


class Ambulance:
    """
    # The Ambulance class

    This class defines an 'Ambulance'; effectively any resource that
    responds to a patient/incident. This includes HEMS, which is a
    child class of Ambulance.


    """

    def __init__(self, ambulance_type="ambulance", callsign="AMBULANCE"):
        self.mobile = ""
        self.as_scene = ""
        self.leaving_scene = ""
        self.at_hospital = ""
        self.clear = ""
        self.stood_down = ""

        self.ambulance_type = ambulance_type
        self.callsign = callsign

    def what_am_i(self):
        print(f"I am {self.ambulance_type}")


# TEST
# Run on python command line

# from class_ambulance import Ambulance
# from class_hems import HEMS

# a = HEMS("CC72", "2021-01-01 00:00:00")
# a.what_am_i()

# b = Ambulance("2021-01-01 00:00:00")
# b.what_am_i()
