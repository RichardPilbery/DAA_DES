# Devon Air Ambulance Discrete Event Simulation

## DES model logic

The model creates patient episodes and associated outcomes based on the following sequence:

1.  Obtain AMPDS card category based on hour of day
2.  Choose a callsign based on activation criteria, which helicopter (if any) is available, whether helicopter can currently fly (servicing or weather impacts), 
3.  Based on callsign, determine the HEMS result (Stand Down Before Mobile, Stand Down En Route, Landed but no patient contact, Patient Treated (Not Conveyed), Patient Conveyed)
4.  Based on the HEMS result determine the patient outcome (Airlifted, Conveyed by land with DAA, Conveyed by land without DAA, Deceased, Unknown)


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


## API Documentation

[https://richardpilbery.github.io/DAA_DES/](https://richardpilbery.github.io/DAA_DES/)

## Web App

The web app can be run using the command

`streamlit run app/app_main.py`
