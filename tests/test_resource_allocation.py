"""
Testing for the choice of HEMS resource sent to individual patients in the Devon Air Ambulance
Discrete Event Simulation (DES) Model.

Unlike the resource tests in test_unittest_model.py, these call the allocation logic directly for
a single patient, rather than checking the results of a full simulation run.

Planned tests are listed below with a [].
Implemented tests are listed below with a [x].

## REG jobs

[x] Whether a helicopter benefit is known in advance is decided once per patient, not once per
   resource considered
[x] If a helicopter benefit is known in advance, a car is not sent when no helicopter is available
[x] If a helicopter benefit is not known in advance, the preferred callsign group and vehicle type
   are sent
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
import simpy
from numpy.random import SeedSequence

from air_ambulance_des.class_hems_availability import HEMSAvailability
from air_ambulance_des.class_patient import Patient
from air_ambulance_des.utils import Utils


def make_availability():
    """Return the HEMS resources at the start of a simulation, with every helicopter available."""
    utility = Utils(master_seed=SeedSequence(42))
    utility.setup_seeds()

    availability = HEMSAvailability(
        env=simpy.Environment(),
        sim_start_date=datetime(2023, 6, 1, 10),
        sim_duration=60 * 24,
        utility=utility,
    )

    # Remove the randomness of ad hoc helicopter unavailability (e.g. weather)
    availability.utilityClass.sample_ad_hoc_reason = MagicMock(return_value="available")

    return availability


def make_reg_patient(pref_callsign_group, pref_vehicle_type, helicopter_benefit):
    """Return a REG patient whose call arrives at 10am in June, when every resource is on shift."""
    patient = Patient(1)
    patient.hour = 10
    patient.month = 6
    patient.qtr = 2
    patient.hems_cc_or_ec = "REG"
    patient.hems_pref_callsign_group = pref_callsign_group
    patient.hems_pref_vehicle_type = pref_vehicle_type
    patient.hems_helicopter_benefit = helicopter_benefit
    return patient


def set_heli_benefit_known(availability, known):
    """Replace the random number stream deciding whether a helicopter benefit is known in advance."""
    rng = MagicMock()
    rng.uniform.return_value = 0.0 if known else 1.0
    availability.utilityClass.rngs["know_heli_benefit"] = rng
    return rng


##############################################################################
# Begin tests                                                                #
##############################################################################


##################################
# REG jobs                       #
##################################
@pytest.mark.resources
@pytest.mark.parametrize("known", [True, False])
def test_heli_benefit_known_decided_once_per_patient(known):
    availability = make_availability()
    rng = set_heli_benefit_known(availability, known)

    # CC72 is the last resource considered, so every resource is checked before it is chosen
    patient = make_reg_patient(72, "car", helicopter_benefit="y")

    availability.preferred_regular_group_available(patient)

    assert rng.uniform.call_count == 1, (
        f"[FAIL - RESOURCE ALLOCATION - HELI BENEFIT KNOWN] Decided {rng.uniform.call_count} times for one patient, expected once"
    )


@pytest.mark.resources
def test_no_car_sent_when_heli_benefit_known():
    availability = make_availability()
    set_heli_benefit_known(availability, known=True)

    availability.utilityClass.sample_ad_hoc_reason = MagicMock(return_value="weather")

    patient = make_reg_patient(72, "car", helicopter_benefit="y")

    resource, reason = availability.preferred_regular_group_available(patient)

    assert resource is None, (
        f"[FAIL - RESOURCE ALLOCATION - HELI BENEFIT KNOWN] {resource.callsign} sent when no helicopter was available ({reason})"
    )


@pytest.mark.resources
def test_preferred_resource_sent_when_heli_benefit_not_known():
    availability = make_availability()
    set_heli_benefit_known(availability, known=False)

    patient = make_reg_patient(72, "car", helicopter_benefit="y")

    resource, reason = availability.preferred_regular_group_available(patient)

    assert resource is not None and resource.callsign == "CC72", (
        f"[FAIL - RESOURCE ALLOCATION - HELI BENEFIT NOT KNOWN] Expected CC72 to be sent, got {resource.callsign if resource else None} ({reason})"
    )
