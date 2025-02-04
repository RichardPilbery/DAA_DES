"""Unit testing for the Discrete-Event Simulation (DES) Model.

These check specific parts of the simulation and code, ensuring they work
correctly and as expected.

Planned tests are listed below with a [].
Implemented tests are listed below with a [x].

## General Checks

[] Results dataframe is longer when model is run for longer


## Multiple Runs

[] Results dataframe is longer when model conducts more runs
[] Results differ across multiple runs
[] Arrivals differ across multiple runs

## Seeds

[] Model behaves consistently across repeated runs when provided with a seed

## Warm-up period impact

[] Data is not present in results dataframe for warm-up period

## Sensible Resource Use

[] All provided resources get at least some utilisation in a model that
   runs for a sufficient length of time with sufficient demand
[] Utilisation never exceeds 100%
[] Utilisation never drops below 0%

## Activity during inactive periods

[] Calls do not generate activity if they arrive during times the resource is meant to be off shift
[] Resources do not respond during times they are meant to be off shift


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

[] Model does not run with a negative number of resource
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
