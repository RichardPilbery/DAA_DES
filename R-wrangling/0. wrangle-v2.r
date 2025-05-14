library(tidyverse)
library(lubridate)
library(readxl)

df <- read_xlsx('data/DAA_with_missing_2023-2024.xlsx', sheet='Tasked') %>%
  rename(
    inc_date = `Date Time`,
    ampds_card = `AMPDS Card (all sources)`
  )

missed_df <- read_xlsx('data/DAA_with_missing_2023-2024.xlsx', sheet='Missed') %>%
  rename(
    inc_date = `Date/time`,
    ampds_card = `AMPDS Card`
  )

dfa <- bind_rows(df, missed_df)



simple_df <- dfa %>%
  filter(is.na(result) | result != 'Remote advice', inc_date > lubridate::ymd('2023-01-01')) %>% # Removing since unreliable
  # Now going to include everything not in main cad categories as other
  #filter(ampds_card %in% AMPDS_card_with_100_df) %>%
  transmute(
    job_id,
    inc_date,
    #ampds_card, # Won't normally be present - this can be derived from the full AMPDS code
    ampds_card = fct_lump_min(str_pad(ampds_card, 2, "left", "0"), 100),
    ampds_code = "",
    dispatch_type = case_when(
      grepl("Request", `Dispatch type`) ~ "Request",
      `Dispatch type` %in% c('Immediate', 'Interrogate') ~ `Dispatch type`,
      .default = NA_character_
    ),
    callsign,
    vehicle = tolower(Vehicle),
    vehicle_type = case_when(
      str_sub(tolower(callsign), 1, 2) == "cc" ~ "car",
      str_sub(tolower(callsign), 1, 2) == "h7" ~ "helicopter",
      .default = NA_character_
    ),
    callsign_group = case_when(
      callsign %in% c("H70","CC70") ~ "70",
      callsign %in% c("H71","CC71") ~ "71",
      callsign == "CC72" ~ "72",
      .default = "Other"
    ),
    hems_result = case_when(
      patient_result %in% c("Conveyed by land without DAA", "Deceased") ~ "Patient Treated (not conveyed)",
      patient_result == "Conveyed by land with DAA" ~ "Patient Conveyed",
      patient_result == "Airlifted" ~ "Patient Conveyed",
      .default = result
    ),
    # sd_reason, These results look a little suss...not sure how useful it is going to be.
    pt_outcome = if_else(is.na(patient_result), "Unknown", patient_result),
    age = Age,
    sex = Sex,
    HLIDD = if_else(is.na(`HLIDD?`), 'n', tolower(`HLIDD?`)),
    helicopter_benefit = if_else(is.na(Helicopter), "n", tolower(Helicopter)),
    cc_benefit = if_else(is.na(`Critical care`), "n", tolower(`Critical care`)),
    ec_benefit = if_else(is.na(`Enhanced care`), "n", tolower(`Enhanced care`)),
    time_allocation = if_else(Allocation > 120, NA_integer_, Allocation),
    time_mobile = if_else(Mobilisation > 30, NA, Mobilisation),
    time_to_scene = if_else(`Journey to scene` > 60 | `Journey to scene` <= 0, NA_integer_, `Journey to scene`),
    time_on_scene = if_else(`On scene time` > 120 | `On scene time` <= 0, NA_integer_, `On scene time`),
    time_to_hospital = if_else(`Journey to hospital` > 140 | `Journey to hospital` <= 0, NA_integer_, `Journey to hospital`),
    total_minus_all = `Total job duration` - `Journey to hospital` - `On scene time` - `Journey to scene` - Mobilisation,
    time_to_clear = case_when(
      total_minus_all < 0 ~ NA_integer_,
      total_minus_all > 90 ~ NA_integer_,
      .default = total_minus_all
    ), 
    call_cat = NA_character_
  ) %>% filter(is.na(callsign) | callsign %in% c("CC70", "CC71", "CC72", "H70", "H71"))


saveRDS(simple_df, 'clean_daa_import_missing_2023_2024.rds')
simple_df %>% write_csv('clean_daa_import_missing_2023_2024.csv')
