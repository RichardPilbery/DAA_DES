# Historical data

This folder contains aggregated historical data (January 2023 onwards, covering two years) derived from the real-world job-level data. It contains no patient-level records.

It is used to:

- visualise the historical range of values and compare them with simulated results (`class_historic_results.py`, `class_simulation_trial_results.py`)
- support the pytest suite and backtests, which check that the model reproduces observed real-world patterns

Most of the files are written by `distribution_fit_utils.py` when distributions are re-fitted from raw data, and are overwritten each time this is done. Months are given as the first day of the month (e.g. `2023-01-01`).

## Call volumes

| File | Contents |
|---|---|
| `historical_monthly_totals_all_calls.csv` | Number of calls (`inc_date`) received per month. |
| `historical_monthly_totals_by_callsign.csv` | Jobs per month by callsign (`CC70`, `CC71`, `CC72`, `H70`, `H71`). |
| `historical_monthly_totals_by_day_of_week.csv` | Calls per month by day of the week. |
| `historical_monthly_totals_by_hour_of_day.csv` | Calls per month by hour of the day (columns `0`-`23`). |
| `historical_daily_calls.csv` | Frequency table: `days` on which `calls_in_day` calls were received. |
| `historical_daily_calls_breakdown.csv` | Calls received on each day of the period (`day`, `calls_in_day`). |
| `historical_jobs_per_month.csv` | Total jobs per month, with the numbers of stand-downs for cars and helicopters. |
| `historical_jobs_per_day_per_callsign.csv` | Frequency table: number of days (`count`) on which each callsign did `jobs_in_day` jobs. Includes days with zero jobs. |
| `historical_care_cat_counts.csv` | Number of jobs by hour of day and care category (`CC`, `EC`; the unnamed first column is a row index). |

## Missed calls

Calls are split into two groups: "HEMS (helo or car) available and sent" and "No HEMS available" (the `callsign_group_simplified` column).

| File | Contents |
|---|---|
| `historical_missed_calls_by_month.csv` | Count of calls in each group per month. |
| `historical_missed_calls_by_hour.csv` | Count of calls in each group by hour of day. |
| `historical_missed_calls_by_quarter_and_hour.csv` | Count of calls in each group by quarter of the year and hour of day. |

## Job durations and utilisation

| File | Contents |
|---|---|
| `historical_job_durations_breakdown.csv` | Long-format, job-level durations in minutes. One row per job and activity (`job_identifier`, `callsign`, `vehicle_type`, `name`, `value`). `name` is one of `time_allocation`, `time_mobile`, `time_to_scene`, `time_on_scene`, `time_to_hospital`, `time_to_clear` or `total_duration`. |
| `historical_median_time_of_activities_by_month_and_resource_type.csv` | Median duration of each activity per month, with separate columns for cars and helicopters (e.g. `median_car_time_on_scene`). |
| `historical_monthly_resource_utilisation.csv` | Per month and callsign, the number of jobs (`n_<callsign>`) and total time spent on jobs (`total_time_<callsign>`, in minutes). |

## calculated/

Files in this subfolder are not raw aggregates of the real data. They are produced by running the simulation with the historical parameters, or by combining historical data with the rota.

| File | Contents |
|---|---|
| `run_params_used.csv` | The parameters (simulation duration, start date, rota seasons, etc.) used for the historical-parameter simulation run. The historical comparison uses `sim_duration_days` to scale totals to a per-year figure. |
| `SIM_hist_params.csv` | Event-level output from every run of the simulation using the historical parameters. This is a very large file (over 3 million rows). |
| `SIM_hist_params_missed_jobs_care_cat_breakdown.csv` | Per-run counts of jobs by care category and whether a HEMS resource was available (`jobs`, `jobs_per_year`). |
| `SIM_hist_params_missed_jobs_care_cat_summary.csv` | The above summarised across runs (average, min, max). |
| `SIM_hist_params_suboptimal_care_cat_sent_summary.csv` | Mean, min and max number of jobs for each combination of the care category of the resource sent (`hems_res_category`) and the care category needed (`care_cat`). Used to show how often a lower-level resource is sent. |
| `SIM_hist_params_suboptimal_vehicle_type_sent_summary.csv` | Mean, min and max number of jobs by vehicle type sent and whether there would have been a helicopter benefit (`heli_benefit`). |
| `theoretical_availability_historical.csv` | Theoretical minutes of availability per month for each callsign, based on the rota. |
| `complete_utilisation_historical.csv` | Per month and callsign, the time used, theoretical availability and the resulting percentage utilisation. |
| `complete_utilisation_historical_summary.csv` | Min, max, mean and median monthly percentage utilisation per callsign. |

Because the `calculated` files depend on the simulation and rota, they should be regenerated if the model logic, the historical parameters or `actual_data/HEMS_ROTA.csv` change.
