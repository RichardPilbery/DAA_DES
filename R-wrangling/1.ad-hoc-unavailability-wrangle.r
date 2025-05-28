# Ad hoc unavailability

library(tidyverse)
library(readxl)
library(lubridate)

df <- read_xlsx('adhoc-unavailability.xlsx', sheet = 'AUG22-JUL24 Instances')


df1 <- df %>%
  transmute(
    aircraft = Aircraft, 
    offline = `Date time offline`,
    online = `Date time online`,
    total_offline = as.numeric(difftime(online, offline, units = "mins")),
    reason = case_when(
      tolower(Reason) %in% c("clinical crew returning on ccc", "clinician absence", "drug/alcohol testing", "pilot training", "pilot absence", "pilot duty hours issue") ~ "crew",
      tolower(Reason) == 'weather' ~ 'weather',
      tolower(Reason) %in% c("aircraft mechanical fault", "maintenance on base", "aircraft not night capable") ~ "aircraft",
      .default = "other"
    ),
    qtr = quarter(offline),
    year = year(offline),
    hour = hour(offline),
    six_hour_bin = cut(hour,
                   breaks = c(-1, 5, 11, 17, 23),
                   labels = c("00-05", "06-11", "12-17", "18-23"))
  ) %>% filter(reason != "other") # only 7 cases


df1 %>%
  group_by(qtr, reason, aircraft, year) %>%
  summarise(
    total_time = sum(total_offline)
  ) %>% #write_csv('split-by-qtr-reason')
  ungroup() %>%
  ggplot(aes(x = qtr, y=total_time, fill = reason )) +
  geom_col() +
  facet_grid(rows = vars(aircraft), cols = vars(year))


df1 %>%
  #filter(between(offline, ymd('2023-01-01'), ymd('2023-12-31'))) %>%
  group_by(hour, reason, aircraft, qtr) %>%
  summarise(
    n = n(),
    total_time = sum(total_offline)
  ) %>% #write_csv('split-by-hour.csv')
  ungroup() %>% 
  ggplot(aes(x = hour, y = total_time, fill=reason)) +
  geom_col() +
  facet_grid(rows = vars(aircraft), cols = vars(qtr))



df1 %>%
  #filter(between(offline, ymd('2023-01-01'), ymd('2023-12-31'))) %>%
  group_by(six_hour_bin, reason, aircraft, qtr) %>%
  summarise(
    n = n(),
    total_time = sum(total_offline)
  ) %>% #write_csv('split-by-hour.csv')
  ungroup() %>% 
  ggplot(aes(x = six_hour_bin, y = total_time, fill=reason)) +
  geom_col() +
  facet_grid(rows = vars(aircraft), cols = vars(qtr))


  