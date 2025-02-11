"""Unit testing for the Discrete-Event Simulation (DES) Model.

These check specific parts of the simulation and code, ensuring they work
correctly and as expected.

Many tests inspired by list here:
https://github.com/pythonhealthdatascience/rap_template_python_des/blob/main/tests/test_unittest_model.py

Planned tests are listed below with a [].
Implemented tests are listed below with a [x].

## General Checks

[] Results dataframe is longer when model is run for longer


## Multiple Runs

[] Results dataframe is longer when model conducts more runs
[] Results differ across multiple runs
[] Arrivals differ across multiple runs
[] Test running the model sequentially and in parallel produce identical results when seeds set

## Seeds

[] Model behaves consistently across repeated runs when provided with a seed

## Warm-up period impact

[] Data is not present in results dataframe for warm-up period

## Arrivals

[] All patients who arrive outside of the warm-up period have an entry in the results dataframe
[] Number of arrivals increase if parameter adjusted
[] Number of arrivals decrease if parameter adjusted

## Sensible Resource Use

[] All provided resources get at least some utilisation in a model that
   runs for a sufficient length of time with sufficient demand
[] Utilisation never exceeds 100%
[] Utilisation never drops below 0%
[] No one waits in the model for a resource to become availabile - they leave and are recorded as missed
[] Resources are used in the expected order determined within the model
[] Resources belonging to the same callsign group don't get sent on jobs at the same time
[] Changing helicopter type results in different unavailability results being generated

## Activity during inactive periods

[] Resources do not respond during times they are meant to be off shift
[] Calls do not generate activity if they arrive during times the resource is meant to be off shift
[] Inactive periods correctly change across seasons if set to do so

## Expected responses of metrics under different conditions

[] Utilisation is higher when resource is reduced but demand kept consistent
[] 'Missed' calls are higher when resource is reduced but demand kept consistent

[] Utilisation is lower when resource is reduced but demand decreases
[] 'Missed' calls are lower when resource is reduced but demand decreases

[] Utilisation is higher when resource is kept consistent but demand increases
[] 'Missed' calls are higher when resource is kept consistent but demand increases

[] Utilisation is lower when resource is kept consistent but demand decreases
[] 'Missed' calls are lower when resource is kept consistent but demand decreases


## Failure to run under nonsensical conditions

[] Model does not run with a negative number of resources
[] Model does not run with negative average demand parameter

"""

import pandas as pd
import pytest
from des_parallel_process import removeExistingResults, parallelProcessJoblib




def test_warmup_only():
    """
    Ensures no results are recorded during the warm-up phase.

    This is tested by running the simulation model with only a warm-up period,
    and then checking that results are all zero or empty.
    """
    pass
