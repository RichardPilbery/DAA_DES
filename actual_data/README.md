# Servicing and rota files

## Callsign Registration

$${Columns:\color{red}registration, \color{green}callsign, \color{blue}model}$$

**callsign_registration_lookup_DEFAULT.csv**: Default version of this file. Change to reflect updates to the resource pool that is available to the air ambulance. Present to allow easy return to default values from web app front-end or after manual changes made to registration lookup file.

**callsign_registration_lookup.csv**: This file will be used by the actual modelling process. Changes in this file will be reflected in resource availability calculations.


## HEMS Rota

$${Columns:\color{green}callsign,\color{black}category, \color{orange}vehicle\_type, \color{black}callsign\_group, summer\_start, winter\_start, summer\_end, winter\_end}$$

**HEMS_ROTA_DEFAULT.csv**: Default version of this file. Change to reflect updates to the resource pool that is available to the air ambulance. Present to allow easy return to default values from web app front-end or after manual changes made to HEMS rota file.

**HEMS_ROTA**: This file will be used by the actual modelling process. Changes in this file will be reflected in resource availability calculations.

## Service History

$${Columns:\color{red}registration,\color{black}last\_service}$$

**service_history.csv**:

## Service Schedules by Model

$${Columns:\color{blue}model,\color{orange}vehicle\_type,\color{black}service\_schedule\_months,service\_duration\_weeks}$$

**service_schedules_by_model.csv**: Not changed by the web app. Additional models of vehicle should be added manually to this list. The models available in this list

# School holidays
*To be filled in*

# Upper allowable time bounds

*To be filled in*
