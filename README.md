# DAA_DES


## Data output items

The model generates a CSV file containing raw data, which can then be wrangled and presented separately, either following a model run(s) or at any time after the model has been run. The table below identifies the column headers and provides a brief description

| Column name               | Description |
| --------------------------| ------------------------------------------------------------------------------ |
| P_ID                      | Patient ID    |
| run_number                | Run number in cases where the model is run repeatedly (e.g. 100x) to enable calculations of confidence intervals etc.|
| time_type                 | Category that elapsed time represents e.g. 'call start', 'on scene', 'leave scene', 'arrive hospital', 'handover', 'time clear'|
| timestamp                 | Elapsed time in seconds since model started running |
| day                       | Day of the week as a string ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun') |
| hour                      | Hour of call as integer between 0–23 |
| weekday                   | String identifying whether the call started on a weekday or weekend |
| month                     | Integer representing the month of the year (1–12) |
| qtr                       | Integer representing the yearly quarter of the call (1–4) |
| callsign                  | String representing callsign of resource (either HEMS or Ambulance service) |
| triage_code               | Call triage outcome represented as one of the AMPDS 'golden' codes (or OTHER) |
| age                       | Integer representing patient age in years |
| sex                       | String representing patient sex ('male', 'female') |
| time_to_first_response    | Integer representing time in minutes to first response (ambulance service or HEMS) |
| time_to_cc                | Integer representing time in minutes to first HEMS response |
| cc_conveyed               | Integer indicating whether HEMS conveyed patient (1 or 0) |
| cc_flown                  | Integer indicating whether HEMS conveyed the patient by air (1 or 0) |
| cc_travelled_with         | Integer indicating whether patient was conveyed by ambulance but HEMS personnel travelled with the patient to hospital (1 or 0) |
| hems                      | Integer indicating whether HEMS attended the incident (1 or 0) |
| cc_desk                   | Integer indicating whether HEMS activation was due to critical care desk dispatch |
| dispatcher_intevention    | Integer indicating whether the ambulance service dispatcher activated HEMS for a called which did not meet auto-dispatch criteria (1 or 0) |