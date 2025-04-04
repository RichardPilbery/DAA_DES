library(readr)
library(bupaverse)
library(processanimateR)
library(processmapR)
library(DiagrammeR)
library(ggplot2)
library(htmlwidgets)

data <- readr::read_csv("visualisation/event_log.csv")

activity_log <- data %>%
    filter(run_number==1) %>%
    bupaR::convert_timestamps("timestamp_dt", ymd_hms) %>%
    bupaR::eventlog(
        case_id = "P_ID",
        activity_id = "time_type",
        activity_instance_id = "activity_id",
        lifecycle_id = "lifecycle_id",
        timestamp = "timestamp_dt",
        resource_id = "callsign"
        )

### Frequency Maps

#### Absolute (counts)

activity_log %>%
    process_map(frequency("absolute"), render=FALSE) %>%
    processmapR::export_graph("visualisation/absolute_frequency.svg",
                         file_type = "svg")

##### Absolute case

activity_log %>%
    process_map(frequency("absolute-case"), render=FALSE) %>%
    processmapR::export_graph("visualisation/absolute_case.svg",
                         file_type = "svg")

#### Relative
activity_log %>%
    process_map(frequency("relative"), render=FALSE) %>%
    processmapR::export_graph("visualisation/relative_case.svg",
                         file_type = "svg")


### Performance maps

#### Mean Times
activity_log %>%
    process_map(performance(), render=FALSE) %>%
    processmapR::export_graph("visualisation/performance_mean.svg",
                         file_type = "svg")


#### Max Times
activity_log %>%
    process_map(performance(FUN = max), render=FALSE) %>%
    processmapR::export_graph("visualisation/performance_max.svg",
                         file_type = "svg")

### Common Routes
activity_log %>%
    trace_explorer(n_traces = 10) %>%
    plot()

ggsave("visualisation/trace_explorer.svg")


### Activity Presence
activity_log %>%
    activity_presence() %>%
    plot()

ggsave("visualisation/activity_presence.svg")

### Processing Time
activity_log %>%
    processing_time("resource-activity", units = "mins") %>%
    plot()

ggsave("visualisation/processing_time_resource_activity.svg")

activity_log %>%
    processing_time("activity", units = "mins") %>%
    plot()

ggsave("visualisation/processing_time_activity.svg")

### Idle Time
activity_log %>%
    idle_time("resource", units = "mins") %>%
    plot()

ggsave("visualisation/idle_time_resource.svg")

## Animated Maps
activity_log %>%
    animate_process() %>%
    saveWidget("visualisation/anim_process.html", selfcontained = FALSE)


## ==== Resource level pathing ==== ##
activity_log_2 <- data %>%
    filter(run_number==1) %>%
    bupaR::convert_timestamps("timestamp_dt", ymd_hms) %>%
    bupaR::eventlog(
        case_id = "P_ID",
        activity_id = "callsign",
        activity_instance_id = "activity_id",
        lifecycle_id = "lifecycle_id",
        timestamp = "timestamp_dt",
        resource_id = "callsign"
        )

activity_log_2 %>%
    process_map(frequency("relative"), render=FALSE) %>%
    processmapR::export_graph("visualisation/relative_resource_level.svg",
                         file_type = "svg")

activity_log_2 %>%
    animate_process() %>%
    saveWidget("visualisation/anim_resource_level.html", selfcontained = FALSE)
