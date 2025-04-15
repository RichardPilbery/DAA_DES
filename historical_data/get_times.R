library(readr)
library(dplyr)
library(tidyr)

df <- readRDS("clean_daa_import_missing_2023_2024.rds")

df %>%
    dplyr::select(callsign, vehicle_type,
    time_allocation, time_mobile,
    time_to_scene, time_on_scene,
    time_to_hospital,
    # total_minus_all, # Not included - duplicate of time_to_clear
    time_to_clear) %>%
    tibble::rowid_to_column("job_identifier") %>%
    dplyr::group_by(job_identifier) %>%
    dplyr::mutate(total_duration = sum(
        time_allocation, time_mobile,
        time_to_scene, time_on_scene,
        time_to_hospital, time_to_clear, na.rm = TRUE)
        ) %>%
    tidyr::pivot_longer(cols = c(
        time_allocation, time_mobile,
        time_to_scene, time_on_scene,
        time_to_hospital, time_to_clear,
        total_duration
    )) %>%
        tidyr::drop_na(callsign,vehicle_type) %>%
        readr::write_csv("historical_job_durations_breakdown.csv")
