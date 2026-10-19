# Tests

This folder contains the pytest test suite for the model and web app. Run all commands from the root folder of the project.

## Running tests

| Command | What it runs |
|---|---|
| `pytest` | Every test. Several tests run long simulations, so this can take a long time. |
| `pytest tests/test_school_holidays.py` | Every test in one file |
| `pytest -m quick` | Tests with a particular marker (see below) |
| `pytest -k demand` | Tests whose names contain a particular string |
| `pytest --html=report.html` | Every test, with an interactive html report |
| `pytest --cov=. --cov-report=html` | Every test, with a code coverage report |

The tests in `test_school_holidays.py`, `test_resource_allocation.py`, `test_call_generation.py` and `test_scenario_importing.py` take seconds, so are a good first check after making changes.

### Markers

Markers group related tests across files. `pytest.ini` holds the most up-to-date list.

| Marker | Tests relating to |
|---|---|
| `quick` | Quickly checking the model runs at a basic level |
| `medium` | Checking the model runs over a longer period than the quick test |
| `warmup` | The warm-up period |
| `calls` | The number of calls/jobs generated |
| `resources` | Resource allocation and unavailability |
| `callsign` | Allocation of callsigns |
| `callsigngroup` | Allocation of callsign groups |
| `jobdurations` | The duration of jobs |
| `reproducibility` | Whether random seeds behave as expected |
| `performance` | Whether basic metrics change as expected when core parameters change |
| `scenarios` | The parameter template and predefined scenarios imported in the web app |
| `holidays` | Reading and checking the school holiday dates used to schedule servicing |
| `backtest` | Whether the results of a fixed scenario have changed since the stored baseline |

## Test files

Each test file starts with a list of the tests it contains, and those that are planned.

| File | Checks |
|---|---|
| `test_unittest_model.py` | Specific parts of the model, such as the warm-up period, demand parameters, and resources not being used off shift or during servicing |
| `test_call_generation.py` | Call generation, including days with no calls (e.g. when demand is reduced) |
| `test_resource_allocation.py` | The choice of resource sent to an individual patient |
| `test_school_holidays.py` | Reading school holiday dates, and that the school holidays in each parameter file are plausible |
| `test_scenario_importing.py` | The parameter template and predefined scenarios, and importing them in the web app |
| `test_reproducibility.py` | Results are the same with the same seed, and different with different seeds |
| `test_against_condition_changes.py` | Results change in the expected direction when parameters change (e.g. adding a helicopter answers more calls) |
| `test_against_reality.py` | Results with the historic parameters are close enough to the historic data in `historical_data` |
| `test_backtest.py` | Results of a fixed scenario haven't changed since the stored baseline (see below) |

Other files and folders:

- `conftest.py` - shared fixtures and command line options. The `simulation_results` fixture runs the model once (10 runs of 4 years with the rotas in `rotas_historic`) and shares the results between the tests that use it.
- `helpers.py` - shared functions used by several test files
- `rotas_historic`, `rotas_test` - parameter files used by particular tests
- `backtest_baseline` - the stored results compared by `test_backtest.py`
- `test_outputs` - csvs written by tests to help investigate failures (not committed)
- `test_logs` - logs from simulation runs during tests

## Side effects of running tests

Many tests run the model the same way as the web app, which reads its parameters from the csvs in `actual_data` and writes its results to `data`. Be aware that:

- Tests that change the parameters in `actual_data` overwrite those csvs while running. Most of them restore the files afterwards, but `conftest.py` also resets `HEMS_ROTA.csv`, `callsign_registration_lookup.csv` and `service_schedules_by_model.csv` to their `_DEFAULT` versions at the end of every test session. **Running tests will undo a scenario you have loaded in the web app.**
- Tests that run the model through `runSim` or `parallelProcessJoblib` delete and rewrite the results in `data` (e.g. `run_results.csv` and `all_results.csv`). `test_backtest.py` restores the `data` folder afterwards, but other tests don't.

## Backtest

`test_backtest.py` checks that the model's results haven't changed unexpectedly. It runs a fixed scenario and compares the results with a stored baseline in `backtest_baseline`. The scenario is:

- the `_DEFAULT` parameter csvs in `actual_data`, which the web app starts with
- the fitted distributions in `distribution_data`
- 5 runs of 730 days from 1st January 2023 at 08:00, with master seed 42

It runs in the same way as the web app, and restores `actual_data` and `data` afterwards. It takes around one to two minutes.

The results compared are:

- calls by month, hour of day and day of week
- jobs and total time on jobs by month and callsign
- missed calls (no HEMS resource available) by month and hour of day
- the number, median and mean duration of each job stage by month and vehicle type

These are totals across all runs, so they are compared exactly (allowing only for tiny floating point differences between machines). The backtest doesn't check whether the results are realistic - `test_against_reality.py` does that.

### When the backtest fails

A failure means the results of the scenario have changed since the baseline was stored. The failure message lists the values that changed, and the full current results are saved to `test_outputs/BACKTEST_CURRENT_<metric>.csv`.

Two other tests help explain why:

- `test_inputs_match_baseline` fails if any input file (the `_DEFAULT` csvs or the files in `distribution_data`) has changed since the baseline was stored. If it passes but the results changed, the change came from the code.
- `test_scenario_matches_baseline` fails if the scenario settings in `test_backtest.py` have changed.

If you **weren't** expecting the results to change, something has changed the model's results unintentionally, and should be investigated before going further.

### Regenerating the baseline

If you **were** expecting the results to change (e.g. after fixing a bug that affects results, refitting the distributions, or changing the default parameters):

1. Run `pytest -m backtest` and check the differences reported are the ones you expected.
2. Regenerate the baseline:

   ```
   pytest -m backtest --update-backtest-baseline
   ```

   This runs the scenario and overwrites the files in `backtest_baseline` with the new results. The backtests are reported as skipped while doing this.
3. Run `pytest -m backtest` again to check it passes.
4. Commit the updated files in `backtest_baseline` in the same commit as the change that caused them, explaining in the commit message why the results changed.

Only regenerate the baseline when the change is expected - regenerating it to make a failing backtest pass hides the change it was meant to catch.

To change the scenario itself, edit `SCENARIO` in `test_backtest.py` and regenerate the baseline.
