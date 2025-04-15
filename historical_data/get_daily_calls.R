library(readr)
library(dplyr)
library(lubridate)

df <- readRDS("clean_daa_import_missing_2023_2024.rds")

calls_in_day_breakdown <- df %>%
    dplyr::mutate(date = lubridate::date(inc_date)) %>%
    dplyr::count(date, name = "calls_in_day")

calls_in_day_breakdown %>%
    select(-date) %>%
    tibble::rowid_to_column("day") %>%
    readr::write_csv("historical_daily_calls_breakdown.csv")

calls_in_day_breakdown %>%
    dplyr::count(calls_in_day, name = "days") %>%
    readr::write_csv("historical_daily_calls.csv")
