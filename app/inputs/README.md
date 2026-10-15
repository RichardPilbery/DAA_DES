This folder contains Excel templates for setting the model parameters.

- `parameter_template.xlsx` - the template users can download from the setup page of the app ("Download file template"), fill in, and upload again. It contains the default parameters and is also used by `tests/test_scenario_importing.py`.
- `parameter_template_test.xlsx` - a copy of the template with a more varied set of parameters, for manual testing of the upload process. It is not currently referenced by any code.

Each template contains the following sheets. When a completed template is uploaded, each sheet (other than `scenario_details`, `additional_parameters` and `allowable_values`) overwrites the csv of the same name in the `actual_data` folder.

| Sheet | Contents |
|---|---|
| `scenario_details` | Name, description and notes for the scenario, shown in the app on upload |
| `callsign_registration_lookup` | Registration, callsign, model, vehicle type and callsign group of each resource |
| `HEMS_ROTA` | Care category and summer/winter start and end hours for each callsign |
| `service_schedules_by_model` | Servicing interval (months) and duration (weeks) for each vehicle model |
| `rota_start_end_months` | First and last month of the summer rota |
| `upper_allowable_time_bounds` | Minimum and maximum permitted duration (minutes) of each activity |
| `service_history` | Date of the most recent service for each registration |
| `school_holidays` | Start and end dates of school holidays, which servicing is scheduled around |
| `additional_parameters` | Number of runs, simulation duration, warm-up duration, start date and time, random seed and activity duration multiplier |
| `allowable_values` | Lists of permitted values used for the dropdowns in the other sheets |

Pre-prepared scenarios that use the same structure can be found in the `scenarios` folder at the root of the repository.
