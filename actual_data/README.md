# Servicing and rota files

This folder holds the scenario inputs that describe the HEMS resource pool (which vehicles exist, when they are staffed and when they are serviced), plus a few supporting lookups.

Most files come in pairs:

- **`<name>_DEFAULT.csv`**: the default version of the file. It is never read by the model. It is present so the web app (or you, after manual edits) can return to the default values.
- **`<name>.csv`**: the working copy. This is the file the modelling process actually reads, and the file the web app overwrites when scenario parameters are changed.

Colours in the column listings below show which columns link files together (e.g. `registration` is the key joining the registration lookup and the service history).

## Callsign Registration

$${Columns:\color{red}registration, \color{green}callsign, \color{blue}model}$$

Links each physical vehicle (by registration) to the callsign it operates under and its vehicle model. Registrations are lower case (e.g. `g-daas`, `cc70`).

**callsign_registration_lookup_DEFAULT.csv**: Default version of this file. Change to reflect updates to the resource pool that is available to the air ambulance. Present to allow easy return to default values from web app front-end or after manual changes made to registration lookup file.

**callsign_registration_lookup.csv**: This file will be used by the actual modelling process. Changes in this file will be reflected in resource availability calculations.

Every registration listed here should also have a row in the service history, and every model should have a row in the service schedules.

## HEMS Rota

$${Columns:\color{green}callsign,\color{black}category, \color{orange}vehicle\_type, \color{black}callsign\_group, summer\_start, winter\_start, summer\_end, winter\_end}$$

Defines when each resource is staffed. There is one row per callsign and care category covered (a callsign can have several rows, e.g. `H70` is critical care by day and enhanced care by night).

- `category`: the care category the resource covers during that shift (`CC` critical care or `EC` enhanced care).
- `vehicle_type`: `helicopter` or `car`.
- `callsign_group`: the crew/callsign number shared by a helicopter and car that are crewed together (e.g. `H70` and `CC70` are both group `70`).
- `summer_start`, `summer_end`, `winter_start`, `winter_end`: the hour of the day (0-23) the shift starts and ends in each season. If the end hour is earlier than the start hour, the shift runs past midnight (e.g. start 19, end 2).

The dates at which summer and winter begin are set in `rota_start_end_months.csv`.

**HEMS_ROTA_DEFAULT.csv**: Default version of this file. Change to reflect updates to the resource pool that is available to the air ambulance. Present to allow easy return to default values from web app front-end or after manual changes made to HEMS rota file.

**HEMS_ROTA.csv**: This file will be used by the actual modelling process. Changes in this file will be reflected in resource availability calculations.

## Rota Start and End Months

$${Columns:\color{black}what, month}$$

Sets the months in which the summer rota starts and ends. Rows are `summer_start_month` and `summer_end_month` (month numbers, 1-12) and the matching `_string` rows with the month name for display. With the defaults, the summer rota runs from April to October and the winter rota applies for the rest of the year.

**rota_start_end_months_DEFAULT.csv** / **rota_start_end_months.csv**: Default and working copies, as above.

## Service History

$${Columns:\color{red}registration,\color{black}last\_service}$$

The date (`YYYY-MM-DD`) on which each vehicle was last serviced. The simulation uses this, together with the service schedule for the vehicle's model, to work out when each vehicle is next unavailable for servicing.

A value of `0` is treated as missing: the vehicle has no servicing scheduled in the simulation. This is used for the cars, which have no service schedule modelled.

**service_history_DEFAULT.csv** / **service_history.csv**: Default and working copies, as above.

## Service Schedules by Model

$${Columns:\color{blue}model,\color{orange}vehicle\_type,\color{black}service\_schedule\_months,service\_duration\_weeks}$$

How often each vehicle model is serviced (`service_schedule_months`) and how long each service takes (`service_duration_weeks`). A schedule of `0` means the model is not serviced in the simulation.

**service_schedules_by_model.csv**: Not changed by the web app. Additional models of vehicle should be added manually to this list. The models available in this list are the ones that can be chosen in the registration lookup, so a model must be added here before it can be used there.

**service_schedules_by_model_DEFAULT.csv**: Default copy, kept for returning to the original values.

## School Holidays

$${Columns:\color{black}year, start\_date, end\_date}$$

The start and end dates of school holiday periods. Servicing is scheduled to avoid these periods (when staff are most likely to be on leave), and to avoid overlapping with the servicing of other resources.

Dates are in UK format (`dd/mm/yyyy`) in the csvs in this folder. ISO format (`yyyy-mm-dd`) is also accepted, as this is what the web app writes when a scenario is imported. UK format is tried first, so `03/04/2026` is read as 3rd April.

The working file is regenerated from the date range of the historical data by `distribution_fit_utils.py`.

**school_holidays_DEFAULT.csv** / **school_holidays.csv**: Default and working copies, as above.

## Upper Allowable Time Bounds

$${Columns:\color{black}time, min\_value\_mins, max\_value\_mins}$$

The minimum and maximum plausible durations (in minutes) of each stage of a job: `time_allocation`, `time_mobile`, `time_to_scene`, `time_on_scene`, `time_to_hospital` and `time_to_clear`.

These bounds are applied when distributions are fitted to the raw job-level data in `distribution_fit_utils.py`. Values outside the range are treated as data errors or extreme outliers and excluded. Changing them therefore only has an effect when distributions are re-fitted, not when the simulation is run on the existing fitted distributions.

**upper_allowable_time_bounds_DEFAULT.csv** / **upper_allowable_time_bounds.csv**: Default and working copies, as above.
