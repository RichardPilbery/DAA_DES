# resource_plotting.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# If DAA_COLORSCHEME is defined elsewhere, you might need to import it
# For example:
# from your_constants_module import DAA_COLORSCHEME
# Or, you can pass it as an argument to the function.
# For this example, let's assume it's passed as an argument.

def display_resource_use_exploration(resource_use_events_only_df, results_all_runs_df, DAA_COLORSCHEME):
    """
    Displays the resource use exploration section including dataframes and plots.

    Args:
        resource_use_events_only_df (pd.DataFrame): DataFrame containing only resource_use and resource_use_end events.
        results_all_runs_df (pd.DataFrame): DataFrame containing all run results, used for missed_job_events.
        DAA_COLORSCHEME (dict): Dictionary defining the color scheme for plots.
    """

    st.subheader("Resource Use")

    # Accounting for odd bug being seen in streamlit community cloud
    # This check might be more robust if done before calling this function,
    # but keeping it here to match original logic if resource_use_events_only_df is passed directly.
    if 'P_ID' not in resource_use_events_only_df.columns:
        resource_use_events_only_df = resource_use_events_only_df.reset_index()

    # The @st.fragment decorator is used to group widgets and outputs
    # that should be treated as a single unit for rerun behavior.
    # If you want this behavior, keep it. Otherwise, it can be removed
    # if the function is called within a fragment in the main app.
    # For this refactoring, we'll keep it to ensure similar behavior.
    @st.fragment
    def resource_use_exploration_plots_fragment():

        run_select_ruep = st.selectbox("Choose the run to show",
                                    resource_use_events_only_df["run_number"].unique(),
                                    key="ruep_run_select" # Added a key for uniqueness
                                    )

        # colour_by_cc_ec = st.toggle("Colour the plot by CC/EC/REG patient benefit",
        #                             value=True, key="ruep_color_toggle") # Added a key

        show_outline = st.toggle("Show an outline to help debug overlapping calls",
                                value=False, key="ruep_outline_toggle") # Added a key

        with st.expander("Click here to see the timings of resource use"):
            st.dataframe(
                resource_use_events_only_df[resource_use_events_only_df["run_number"] == run_select_ruep]
            )

            st.dataframe(
                resource_use_events_only_df[resource_use_events_only_df["run_number"] == run_select_ruep]
                [['callsign', 'callsign_group', 'registration']]
                .value_counts()
            )

            st.dataframe(resource_use_events_only_df[resource_use_events_only_df["run_number"] == run_select_ruep]
                        [["P_ID", "time_type", "timestamp_dt", "event_type"]]
                        .melt(id_vars=["P_ID", "time_type", "event_type"],
                                value_vars="timestamp_dt").drop_duplicates())

            resource_use_wide = (resource_use_events_only_df[resource_use_events_only_df["run_number"] == run_select_ruep]
                                [["P_ID", "time_type", "timestamp_dt", "event_type", "registration", "care_cat"]].drop_duplicates()
                                .pivot(columns="event_type", index=["P_ID","time_type", "registration", "care_cat"], values="timestamp_dt").reset_index())

            # get the number of resources and assign them a value
            resources = resource_use_wide.time_type.unique()
            resources = np.concatenate([resources, ["No Resource Available"]])
            resource_dict = {resource: index for index, resource in enumerate(resources)}

            missed_job_events = results_all_runs_df[
                (results_all_runs_df["run_number"] == run_select_ruep) & # Filter by selected run first
                (results_all_runs_df["event_type"] == "resource_request_outcome") &
                (results_all_runs_df["time_type"] == "No Resource Available")
            ].copy() # Use .copy() to avoid SettingWithCopyWarning if further modifications are made

            # Check if 'P_ID' is in columns, if not, reset_index (bug handling from original)
            if 'P_ID' not in missed_job_events.columns and not missed_job_events.empty:
                 missed_job_events = missed_job_events.reset_index()

            missed_job_events = missed_job_events[["P_ID", "time_type", "timestamp_dt", "event_type", "registration", "care_cat"]].drop_duplicates()
            missed_job_events["event_type"] = "resource_use"

            missed_job_events_end = missed_job_events.copy()
            missed_job_events_end["event_type"] = "resource_use_end"
            missed_job_events_end["timestamp_dt"] = pd.to_datetime(missed_job_events_end["timestamp_dt"]) + timedelta(minutes=5)

            missed_job_events_full = pd.concat([missed_job_events, missed_job_events_end])
            missed_job_events_full["registration"] = "No Resource Available" # Explicitly set for these events

            if not missed_job_events_full.empty:
                missed_job_events_full_wide = missed_job_events_full.pivot(columns="event_type", index=["P_ID","time_type", "registration", "care_cat"], values="timestamp_dt").reset_index()
                resource_use_wide = pd.concat([resource_use_wide, missed_job_events_full_wide]).reset_index(drop=True)
            else:
                # Ensure columns match if missed_job_events_full_wide is empty
                # This might need more robust handling based on expected columns
                pass

            resource_use_wide["y_pos"] = resource_use_wide["time_type"].map(resource_dict)

            resource_use_wide["resource_use_end"] = pd.to_datetime(resource_use_wide["resource_use_end"])
            resource_use_wide["resource_use"] = pd.to_datetime(resource_use_wide["resource_use"])

            resource_use_wide["duration"] = resource_use_wide["resource_use_end"] - resource_use_wide["resource_use"]
            resource_use_wide["duration_seconds"] = (resource_use_wide["resource_use_end"] - resource_use_wide["resource_use"]).dt.total_seconds()*1000
            resource_use_wide["duration_minutes"] = resource_use_wide["duration_seconds"] / 1000 / 60
            resource_use_wide["duration_minutes"] = resource_use_wide["duration_minutes"].round(1)

            resource_use_wide["callsign_group"] = resource_use_wide["time_type"].str.extract(r"(\d+)") # Added r for raw string

            resource_use_wide = resource_use_wide.sort_values(["callsign_group", "time_type"])

            st.dataframe(resource_use_wide)

            ######################################
            # Load in the servicing schedule df
            ######################################
            try:
                service_schedule = pd.read_csv("data/service_dates.csv")
            except FileNotFoundError as e:
                st.error(f"Error loading service schedule data: {e}. Please ensure 'data/service_dates.csv' exists.")
                return # Stop execution if files are not found
            try:
                callsign_lookup = pd.read_csv("actual_data/callsign_registration_lookup.csv")
            except FileNotFoundError as e:
                st.error(f"Error loading callsign lookup data: {e}. Please ensure 'actual_data/callsign_registration_lookup.csv' exists.")
                return # Stop execution if files are not found

            service_schedule = service_schedule.merge(callsign_lookup, on="registration") # Specify merge key if different

            service_schedule["service_end_date"] = pd.to_datetime(service_schedule["service_end_date"])
            service_schedule["service_start_date"] = pd.to_datetime(service_schedule["service_start_date"])
            service_schedule["duration_seconds"] = ((service_schedule["service_end_date"] - service_schedule["service_start_date"]) + timedelta(days=1)).dt.total_seconds()*1000
            service_schedule["duration_days"] = (service_schedule["duration_seconds"] / 1000) / 60 / 60 / 24

            # Map y_pos using the comprehensive resource_dict
            service_schedule["y_pos"] = service_schedule["callsign"].map(resource_dict)
            # Filter out entries that couldn't be mapped if necessary, or handle NaNs
            service_schedule = service_schedule.dropna(subset=['y_pos'])

            # Ensure resource_use_wide is not empty before accessing .min()/.max()
            if not resource_use_wide.empty and 'resource_use' in resource_use_wide.columns:
                    min_date = resource_use_wide.resource_use.min()
                    max_date = resource_use_wide.resource_use.max()
                    service_schedule = service_schedule[(service_schedule["service_start_date"] <= max_date) &
                                                    (service_schedule["service_end_date"] >= min_date)]
            else: # Handle case where resource_use_wide might be empty or missing column
                    service_schedule = pd.DataFrame(columns=service_schedule.columns) # Empty df with same columns


            st.dataframe(service_schedule)

        # Create figure
        resource_use_fig = go.Figure()

        # Add horizontal bars using actual datetime values
        # Ensure unique callsigns are taken from the sorted resource_use_wide for consistent y-axis order
        unique_time_types_sorted = resource_use_wide.time_type.unique()

        for idx, callsign in enumerate(unique_time_types_sorted):
            callsign_df = resource_use_wide[resource_use_wide["time_type"]==callsign]
            service_schedule_df = service_schedule[service_schedule["callsign"]==callsign]

            # Add in hatched boxes showing the servicing periods
            if not service_schedule_df.empty:
                resource_use_fig.add_trace(go.Bar(
                    x=service_schedule_df["duration_seconds"],
                    y=service_schedule_df["y_pos"],
                    base=service_schedule_df["service_start_date"],
                    orientation="h",
                    width=0.6,
                    marker_pattern_shape="x",
                    marker=dict(color="rgba(63, 63, 63, 0.30)",
                                line=dict(color="black", width=1)),
                    name=f"Servicing = {callsign}",
                    customdata=service_schedule_df[['callsign','duration_days','service_start_date', 'service_end_date', 'registration']],
                    hovertemplate="Servicing %{customdata[0]} (registration %{customdata[4]}) lasting %{customdata[1]:.1f} days<br>(%{customdata[2]|%a %-e %b %Y} to %{customdata[3]|%a %-e %b %Y})<extra></extra>"
                ))

            # if colour_by_cc_ec: # Logic for this needs DAA_COLORSCHEME and potentially cc_ec_reg_colour_lookup
            #     if not callsign_df.empty and 'care_cat' in callsign_df.columns:
            #         cc_ec_status = callsign_df["care_cat"].values[0] # This might need adjustment if multiple care_cat per callsign
            #         # cc_ec_reg_colour_lookup would also be needed here
            #         # marker_val = dict(color=list(DAA_COLORSCHEME.values())[cc_ec_reg_colour_lookup[cc_ec_status]])
            #     else:
            #         marker_val = dict(color=list(DAA_COLORSCHEME.values())[idx % len(DAA_COLORSCHEME)])
            # else: # Fallback or default coloring
            if show_outline:
                marker_val=dict(color=list(DAA_COLORSCHEME.values())[idx % len(DAA_COLORSCHEME)], # Use modulo for safety
                                line=dict(color="#FFA400", width=0.2))
            else:
                marker_val = dict(color=list(DAA_COLORSCHEME.values())[idx % len(DAA_COLORSCHEME)]) # Use modulo

            # Add in boxes showing the duration of individual calls
            if not callsign_df.empty:
                resource_use_fig.add_trace(go.Bar(
                    x=callsign_df["duration_seconds"],
                    y=callsign_df["y_pos"],
                    base=callsign_df["resource_use"],
                    orientation="h",
                    width=0.4,
                    marker=marker_val,
                    name=callsign,
                    customdata=callsign_df[['resource_use','resource_use_end','time_type', 'duration_minutes', 'registration', 'care_cat']],
                    hovertemplate="Response to %{customdata[5]} call from %{customdata[2]}<br>(registration %{customdata[4]}) lasting %{customdata[3]:.1f} minutes<br>(%{customdata[0]|%a %-e %b %Y %H:%M} to %{customdata[1]|%a %-e %b %Y %H:%M})<extra></extra>"
                ))

        # Layout tweaks
        resource_use_fig.update_layout(
            title_text="Resource Use Over Time", # Changed from title
            barmode='overlay',
            xaxis=dict(
                title_text="Time", # Changed from title
                type="date",
            ),
            yaxis=dict(
                title_text="Callsign", # Changed from title
                tickmode="array",
                tickvals=list(resource_dict.values()),
                ticktext=list(resource_dict.keys()), # These should be the sorted unique callsigns
                autorange = "reversed"
            ),
            showlegend=True,
            height=700
        )

        resource_use_fig.update_xaxes(rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ]))
        )
        # Ensure the output directory exists
        # import os
        # os.makedirs("app/fig_outputs", exist_ok=True)
        # resource_use_fig.write_html("app/fig_outputs/resource_use_fig.html",full_html=False, include_plotlyjs='cdn')

        st.plotly_chart(
            resource_use_fig,
            use_container_width=True # Added for better responsiveness
        )

    # Call the fragment function to render its content
    resource_use_exploration_plots_fragment()
