"""
DAA_DES: Discrete Event Simulation tools for air ambulance operational modelling.
"""

__version__ = "0.1.0"
__author__ = ["Richard Pilbery", "Sammi Rosser", "Hannah Trebilcock"]

from . import class_ambulance
from . import class_hems_availability
from . import class_hems
from . import class_historic_results
from . import class_patient
from . import class_simulation_inputs
from . import class_simulation_trial_results
from . import des_hems
from . import des_parallel_process
from . import distribution_fit_utils
from . import school_holidays
from . import utils

__all__ = [
    "class_ambulance",
    "class_hems_availability",
    "class_hems",
    "class_historic_results",
    "class_patient",
    "class_simulation_inputs",
    "class_simulation_trial_results",
    "des_hems",
    "des_parallel_process",
    "distribution_fit_utils",
    "school_holidays",
    "utils",
]

"""
.. include:: ../README.md
"""
