import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Ensure this folder is in sys.path
import _processing_functions
import pandas as pd
import plotly.express as px

import _vehicle_calculation
import plotly.graph_objects as go

import streamlit as st

from _app_utils import DAA_COLORSCHEME, iconMetricContainer

def make_utilisation_model_dataframe(path="../data/run_results.csv",
                                     params_path="../data/run_params_used.csv",
                                     rota_path="../actual_data/HEMS_ROTA.csv",
                                     callsign_path="../actual_data/callsign_registration_lookup.csv",
                                     service_path="../data/service_dates.csv",
                                     ):
    df = pd.read_csv(path)
    params_df = pd.read_csv(params_path)
    n_runs = len(df["run_number"].unique())

    daily_availability, total_avail_minutes = (
        _vehicle_calculation.calculate_available_hours_v2(
            params_df,
            rota_data=pd.read_csv(rota_path),
            service_data=pd.read_csv(service_path),
            callsign_data=pd.read_csv(callsign_path),
            long_format_df=False
                    )
                )


    # print("==_utilisation_result_calculation - make_utilisation_model_dataframe - run_vehicle_calculation.calculate_available_hours_v2==")
    # print("daily_availability")
    # print(daily_availability)

    # print("total_avail_minutes")
    # print(total_avail_minutes)

    # total_avail_minutes["callsign"] = total_avail_minutes["callsign"].str.replace("CC", "C")

    # print("df")
    # print(df)
    # Add callsign column if not already present in the dataframe passed to the function
    if 'callsign' not in df.columns:
        df = _processing_functions.make_callsign_column(df)

    # Restrict to only events in the event log where resource use was starting or ending
    resource_use_only = df[df["event_type"].isin(["resource_use", "resource_use_end"])].copy()

    del df

    # print("==resource_use_only==")
    # print(resource_use_only)
    # Pivot to wide-format dataframe with one row per patient/call
    # and columns for start and end types
    resource_use_wide = (
        resource_use_only[["P_ID", "run_number", "event_type", "timestamp_dt",
                           "callsign_group", "vehicle_type", "callsign"]]
        .pivot(index=["P_ID", "run_number", "callsign_group", "vehicle_type", "callsign"],
               columns="event_type", values="timestamp_dt").reset_index()
               )

    del resource_use_only

    # print("==resource_use_wide - initial==")
    # print(resource_use_wide)
    # If utilisation end date is missing then set to end of model
    # as we can assume this is a call that didn't finish before the model did
    resource_use_wide = _processing_functions.fill_missing_values(
        resource_use_wide, "resource_use_end",
        _processing_functions.get_param("sim_end_date", params_df)
        )

    # If utilisation start time is missing, then set to start of model + warm-up time (if relevant)
    # as can assume this is a call that started before the warm-up period elapsed but finished
    # after the warm-up period elapsed
    # TODO: need to add in a check to ensure this only happens for calls at the end of the model,
    # not due to errors elsewhere that could fail to assign a resource end time
    resource_use_wide = _processing_functions.fill_missing_values(
        resource_use_wide, "resource_use",
        _processing_functions.get_param("warm_up_end_date", params_df)
        )

    # Calculate number of minutes the attending resource was in use on each call
    resource_use_wide["resource_use_duration"] = _processing_functions.calculate_time_difference(
        resource_use_wide, 'resource_use', 'resource_use_end', unit='minutes'
        )

    # ============================================================ #
    # Calculage per-run utilisation,
    # stratified by callsign and vehicle type (car/helicopter)
    # ============================================================ #
    utilisation_df_per_run = (
        resource_use_wide.groupby(['run_number', 'vehicle_type', 'callsign'])
        [["resource_use_duration"]]
        .sum()
        )

    # Join with df of how long each resource was available for in the sim
    # We will for now assume this is the same across each run
    utilisation_df_per_run = utilisation_df_per_run.reset_index(drop=False).merge(
        total_avail_minutes, on="callsign", how="left"
        )

    utilisation_df_per_run["perc_time_in_use"] = (
        utilisation_df_per_run["resource_use_duration"].astype(float) /
        # float(_processing_functions.get_param("sim_duration", params_df))
        utilisation_df_per_run["total_available_minutes_in_sim"].astype(float)
        )

    # Add column of nicely-formatted values to make printing values more streamlined
    utilisation_df_per_run["PRINT_perc"] = utilisation_df_per_run["perc_time_in_use"].apply(
        lambda x: f"{x:.1%}")

    # print("==utilisation_df_per_run==")
    # print(utilisation_df_per_run)

    # ============================================================ #
    # Calculage averge utilisation across simulation,
    # stratified by callsign group
    # ============================================================ #
    # print("==resource_use_wide==")
    # print(resource_use_wide)

    utilisation_df_per_run_by_csg = (
        resource_use_wide.groupby(['callsign_group'])
        [["resource_use_duration"]]
        .sum()
        )

    utilisation_df_per_run_by_csg["resource_use_duration"] = (
        utilisation_df_per_run_by_csg["resource_use_duration"] /
        n_runs
        )

    utilisation_df_per_run_by_csg = utilisation_df_per_run_by_csg.reset_index()

    # print("==utilisation_df_per_run_by_csg==")
    # print(utilisation_df_per_run_by_csg)

    total_avail_minutes_per_csg = total_avail_minutes.groupby('callsign_group').head(1).drop(columns='callsign')
    # print("==utilisation_df_per_run_by_csg==")
    # print(utilisation_df_per_run_by_csg)

    # print("==total_avail_minutes_per_csg")
    # print(total_avail_minutes_per_csg)
    total_avail_minutes_per_csg['callsign_group'] =  total_avail_minutes_per_csg['callsign_group'].astype('float')

    utilisation_df_per_run_by_csg = utilisation_df_per_run_by_csg.merge(
        total_avail_minutes_per_csg, on="callsign_group", how="left"
        )

    utilisation_df_per_run_by_csg["perc_time_in_use"] = (
        utilisation_df_per_run_by_csg["resource_use_duration"].astype(float) /
        # float(_processing_functions.get_param("sim_duration", params_df))
        utilisation_df_per_run_by_csg["total_available_minutes_in_sim"].astype(float)
        )

    utilisation_df_per_run_by_csg["PRINT_perc"] = utilisation_df_per_run_by_csg["perc_time_in_use"].apply(
        lambda x: f"{x:.1%}"
        )


    # ============================================================ #
    # Calculage averge utilisation across simulation,
    # stratified by callsign and vehicle type (car/helicopter)
    # ============================================================ #
    utilisation_df_overall = (
        utilisation_df_per_run.groupby(['callsign', 'vehicle_type'])
        [["resource_use_duration"]]
        .sum()
        )

    utilisation_df_overall["resource_use_duration"] = (
        utilisation_df_overall["resource_use_duration"] /
        n_runs
        )

    utilisation_df_overall = utilisation_df_overall.reset_index(drop=False).merge(
        total_avail_minutes, on="callsign", how="left"
        )

    utilisation_df_overall["perc_time_in_use"] = (
        utilisation_df_overall["resource_use_duration"].astype(float) /
        # float(_processing_functions.get_param("sim_duration", params_df))
        utilisation_df_overall["total_available_minutes_in_sim"].astype(float)
    )

    # Add column of nicely-formatted values to make printing values more streamlined
    utilisation_df_overall["PRINT_perc"] = utilisation_df_overall["perc_time_in_use"].apply(
        lambda x: f"{x:.1%}"
        )


    # Return tuple of values
    return (resource_use_wide, utilisation_df_overall,
            utilisation_df_per_run, utilisation_df_per_run_by_csg)


def make_SIMULATION_utilisation_variation_plot(utilisation_df_per_run,
                                               car_colour="blue",
                                               helicopter_colour="red",
                                               use_poppins=False):
    """
    Creates a box plot to visualize the variation in resource utilization
    across all simulation runs.

    Parameters
    ----------
    utilisation_df_per_run : pandas.DataFrame
        A DataFrame containing utilization data per simulation run, generated by
        make_utilisation_model_dataframe().
        It must include the columns:
        - "callsign": Identifier for the resource.
        - "perc_time_in_use": Percentage of time the resource was in use.
        - "vehicle_type": Type of vehicle (e.g., "Car", "Helicopter").

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly box plot showing the distribution of utilization percentages
        for each resource, grouped by vehicle type.

    Notes
    -----
    - The `vehicle_type` column values are capitalized for consistency.
    - The x-axis values are formatted as percentages with no decimal places.
    - A custom color scheme is applied based on vehicle type:
      - "Car" is mapped to `DAA_COLORSCHEME["blue"]`.
      - "Helicopter" is mapped to `DAA_COLORSCHEME["red"]`.

    Example
    -------
    >>> fig = make_SIMULATION_utilisation_variation_plot(utilisation_df)
    >>> fig.show()
    """
    utilisation_df_per_run = utilisation_df_per_run.reset_index()
    utilisation_df_per_run["vehicle_type"] = utilisation_df_per_run["vehicle_type"].str.title()

    fig = (px.box(utilisation_df_per_run,
        x="perc_time_in_use",
        y="callsign",
        color="vehicle_type",
        title="Variation in Resource Utilisation Across All Simulation Runs",
                    labels={
                     "callsign": "Callsign",
                     "perc_time_in_use": "Average Percentage of Available<br>Time Spent in Use",
                     "vehicle_type": "Vehicle Type"
                 },
                 color_discrete_map={'Car': DAA_COLORSCHEME[car_colour],
                                     "Helicopter": DAA_COLORSCHEME[helicopter_colour]})
        .update_layout(
                    xaxis={
                        "tickformat": ".0%"  # Formats as percentage with no decimal places
                    }))

    # TODO: Add indications of good/bad territory for utilisation levels

    if use_poppins:
        return fig.update_layout(font=dict(family="Poppins", size=18, color="black"))
    else:
        return fig

def make_SIMULATION_utilisation_summary_plot(utilisation_df_overall,
                                             historical_utilisation_df_summary,
                                               car_colour="blue",
                                               helicopter_colour="red",
                                               use_poppins=False):
    """
    Creates a bar plot to summarize the average resource utilization
    across all simulation runs.

    Parameters
    ----------
    utilisation_df_overall : pandas.DataFrame
        A DataFrame containing overall utilization data, generated by
        make_utilisation_model_dataframe()..
        It must include the columns:
        - "callsign": Identifier for the resource.
        - "perc_time_in_use": Average percentage of time the resource was in use.
        - "vehicle_type": Type of vehicle (e.g., "Car", "Helicopter").

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly bar plot showing the average utilization percentage
        for each resource, grouped by vehicle type.

    Notes
    -----
    - The `vehicle_type` column values are capitalized for consistency.
    - The y-axis values are formatted as percentages with no decimal places.
    - A custom color scheme is applied based on vehicle type:
      - "Car" is mapped to `DAA_COLORSCHEME["blue"]`.
      - "Helicopter" is mapped to `DAA_COLORSCHEME["red"]`.

    Example
    -------
    >>> fig = make_SIMULATION_utilisation_summary_plot(utilisation_df)
    >>> fig.show()
    """
    utilisation_df_overall = utilisation_df_overall.reset_index()
    utilisation_df_overall["vehicle_type"] = utilisation_df_overall["vehicle_type"].str.title()
    utilisation_df_overall["perc_time_formatted"] = utilisation_df_overall["perc_time_in_use"].apply(lambda x: f"Simulated: {x:.1%}")

    # Create base bar chart
    fig = px.bar(
        utilisation_df_overall,
        y="perc_time_in_use",
        x="callsign",
        color="vehicle_type",
        opacity=0.5,
        text="perc_time_formatted",
        title="Average Resource Utilisation Across All Simulation Runs",
        labels={
            "callsign": "Callsign",
            "perc_time_in_use": "Average Percentage of Available<br>Time Spent in Use",
            "vehicle_type": "Vehicle Type"
        },
        color_discrete_map={
            'Car': DAA_COLORSCHEME.get(car_colour, "blue"), # Use .get for safety
            'Helicopter': DAA_COLORSCHEME.get(helicopter_colour, "red")
        },
        # barmode='group' is default when color is used, which is good.
    )

    # Place actual label at the bottom of the bar
    fig.update_traces(textposition='inside', insidetextanchor='start')

    fig.update_layout(
        bargap=0.4,
        yaxis_tickformat=".0%",
        xaxis_type="category",  # Explicitly set x-axis to category
        # Ensure categories are ordered as per the sorted DataFrame
        # If callsigns are purely numeric but should be treated as categories, ensure they are strings
        xaxis={'categoryorder':'array', 'categoryarray': sorted(utilisation_df_overall['callsign'].unique())}
    )

    # Get the unique, sorted callsigns as they will appear on the x-axis
    # This order is now determined by the 'categoryarray' in layout or default sorting.
    x_axis_categories = sorted(utilisation_df_overall['callsign'].unique())

    # Define the width of the historical markers (box and line) relative to category slot
    # A category slot is 1 unit wide (e.g., from -0.5 to 0.5 around the category's integer index).
    # We'll make the markers 80% of this width.
    historical_marker_width_fraction = 0.3  # Half-width, so total width is 0.8

    # Iterate through the callsigns in the order they appear on the axis
    for num_idx, callsign_str in enumerate(x_axis_categories):
        if callsign_str in historical_utilisation_df_summary.index:
            row = historical_utilisation_df_summary.loc[callsign_str]
            min_val = row["min"] / 100.0  # Convert percentage to 0-1 scale
            max_val = row["max"] / 100.0
            mean_val = row["mean"] / 100.0

            # Calculate x-positions for the historical markers
            # num_idx is the integer position of the category (0, 1, 2, ...)
            x_pos_start = num_idx - historical_marker_width_fraction
            x_pos_end = num_idx + historical_marker_width_fraction

            # --- Min/Max shaded rectangle for historical range ---
            fig.add_shape(
                type="rect",
                xref="x", yref="y",  # Refer to data coordinates
                x0=x_pos_start, y0=min_val,
                x1=x_pos_end, y1=max_val,
                fillcolor=DAA_COLORSCHEME.get("historical_box_fill", "rgba(0,0,0,0.08)"),
                line=dict(color="rgba(0,0,0,0)"), # No border for the box
                layer="below"  # Draw below the main bars
            )

            # --- Mean horizontal line for historical mean ---
            fig.add_shape(
                type="line",
                xref="x", yref="y",
                x0=x_pos_start, y0=mean_val,
                x1=x_pos_end, y1=mean_val,
                line=dict(
                    dash="dot",
                    color=DAA_COLORSCHEME.get("charcoal", "black"),
                    width=2
                ),
                layer="below" # Draw below main bars
            )

            # --- Mean value label (text annotation) ---
            # Using go.Scatter for text annotation, positioned at the center of the category
            fig.add_trace(go.Scatter(
                x=[callsign_str],  # Use the category name for x
                y=[mean_val + (0.01 if max_val < 0.95 else -0.01)], # Adjust y to avoid overlap with top
                text=[f"Historical: {mean_val * 100:.0f}%"], # Simpler label
                mode="text",
                textfont=dict(
                    color=DAA_COLORSCHEME.get("charcoal", "black"),
                    size=10
                ),
                hoverinfo="skip",
                showlegend=False
            ))

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    if use_poppins:
        fig.update_layout(font=dict(family="Poppins", size=18, color=DAA_COLORSCHEME.get("charcoal", "black")))
    else: # Apply a default font for better appearance
        fig.update_layout(font=dict(family="Arial, sans-serif", size=12, color=DAA_COLORSCHEME.get("charcoal", "black")))

    return fig

def make_RWC_utilisation_dataframe(
        historical_df_path="../historical_data/historical_monthly_resource_utilisation.csv",
        rota_path="../tests/rotas_historic/HISTORIC_HEMS_ROTA.csv",
        callsign_path="../tests/rotas_historic/HISTORIC_callsign_registration_lookup.csv",
        service_path="../tests/rotas_historic/HISTORIC_service_dates.csv"):

    historical_utilisation_df = pd.read_csv(historical_df_path)

    def calculate_theoretical_time(
        historical_df,
        rota_df,
        service_df,
        callsign_df,
        long_format_df=True):

        # Pull in relevant rota, registration and servicing data
        rota_df = rota_df.merge(callsign_df, on="callsign")
        service_df = service_df.merge(callsign_df, on="registration")
        # print("==calculate_theoretical_time - rota_df after merging with callsign_df==")
        # print(rota_df)

        # Convert date columns to datetime format
        historical_df['month'] = pd.to_datetime(historical_df['month'])

        # Create a dummy dataframe with every date in the range represented
        # We'll use this to make sure days with 0 activity get factored in to the calculations
        date_range = pd.date_range(start=historical_df['month'].min(),
                        end=pd.offsets.MonthEnd().rollforward(historical_df['month'].max()),
                        freq='D')
        daily_df = pd.DataFrame({'date': date_range})

        # print("==historical_df==")
        # print(historical_df)

        service_df['service_start_date'] = pd.to_datetime(service_df['service_start_date'])
        service_df['service_end_date'] = pd.to_datetime(service_df['service_end_date'])

        def is_summer(date_obj):
            return current_date.month in [4,5,6,7,8,9]

        # Initialize columns in df_availability for each unique callsign
        for callsign in rota_df['callsign'].unique():
            daily_df[callsign] = 0 # Initialize with 0 minutes

        daily_df = daily_df.set_index('date')

        # Iterate through each date in our availability dataframe
        for date_idx, current_date in enumerate(daily_df.index):
            is_current_date_summer = is_summer(current_date)

            # Iterate through each resource's rota entry
            for _, rota_entry in rota_df.iterrows():
                callsign = rota_entry['callsign']
                start_hour_col = 'summer_start' if is_current_date_summer else 'winter_start'
                end_hour_col = 'summer_end' if is_current_date_summer else 'winter_end'

                start_hour = rota_entry[start_hour_col]
                end_hour = rota_entry[end_hour_col]

                # --- Calculate minutes for the current_date ---
                minutes_for_callsign_on_date = 0

                # Scenario 1: Shift is fully within one day (e.g., 7:00 to 19:00)
                if start_hour < end_hour:
                    # Check if this shift is active on current_date (it always is in this logic,
                    # as we are calculating for the current_date based on its rota)
                    minutes_for_callsign_on_date = (end_hour - start_hour) * 60
                # Scenario 2: Shift spans midnight (e.g., 19:00 to 02:00)
                elif start_hour > end_hour:
                    # Part 1: Minutes from start_hour to midnight on current_date
                    minutes_today = (24 - start_hour) * 60
                    minutes_for_callsign_on_date += minutes_today

                    # Part 2: Minutes from midnight to end_hour on the *next* day
                    # These minutes need to be added to the *next day's* total for this callsign
                    if date_idx + 1 < len(daily_df): # Ensure there is a next day in our df
                        next_date = daily_df.index[date_idx + 1]
                        minutes_on_next_day = end_hour * 60
                        daily_df.loc[next_date, callsign] = daily_df.loc[next_date, callsign] + minutes_on_next_day

                daily_df.loc[current_date, callsign] += minutes_for_callsign_on_date

        theoretical_availability = daily_df.copy().reset_index()
        theoretical_availability["month"] = theoretical_availability["date"].dt.strftime('%Y-%m-01')
        theoretical_availability = theoretical_availability.drop(columns=["date"])
        theoretical_availability= theoretical_availability.set_index("month")
        theoretical_availability = theoretical_availability.groupby('month').sum()
        theoretical_availability = theoretical_availability.reset_index()

        # print("==_utilisation_result_calculation.py - make_RWC_utilisation_dataframe - theoretical availability df==")
        # print(theoretical_availability_df)

        theoretical_availability.to_csv("historical_data/calculated/theoretical_availability_historical.csv", index=False)

        if long_format_df:
            theoretical_availability_df = (
                theoretical_availability
                .melt(id_vars="month")
                .rename(columns={"value":"theoretical_availability", "variable": "callsign"})
                )

            theoretical_availability_df['theoretical_availability'] = (
                theoretical_availability_df['theoretical_availability'].astype('float')
                )

            theoretical_availability_df = theoretical_availability_df.fillna(0.0)

        return theoretical_availability_df

    theoretical_availability_df = calculate_theoretical_time(
        historical_df=historical_utilisation_df,
        rota_df=pd.read_csv(rota_path),
        callsign_df=pd.read_csv(callsign_path),
        service_df=pd.read_csv(service_path),
        long_format_df=True
    )

    # print("==theoretical_availability_df==")
    # print(theoretical_availability_df)
    theoretical_availability_df['month'] = pd.to_datetime(theoretical_availability_df['month'])

    # theoretical_availability_df.to_csv("historical_data/calculated/theoretical_availability_historical.csv")

    historical_utilisation_df_times = (
        historical_utilisation_df.set_index('month')
        .filter(like='total_time').reset_index()
        )

    historical_utilisation_df_times.columns = [
        x.replace("total_time_","")
        for x in historical_utilisation_df_times.columns
        ]

    historical_utilisation_df_times = (
        historical_utilisation_df_times.melt(id_vars="month")
        .rename(columns={"value":"usage", "variable": "callsign"})
        )

    historical_utilisation_df_times = historical_utilisation_df_times.fillna(0)

    # print(historical_utilisation_df_times)
    # print(theoretical_availability_df)

    historical_utilisation_df_complete = pd.merge(
        left=historical_utilisation_df_times,
        right=theoretical_availability_df,
        on=["callsign", "month"],
        how="left"
    )

    historical_utilisation_df_complete["percentage_utilisation"] = (
        historical_utilisation_df_complete["usage"] /
        historical_utilisation_df_complete["theoretical_availability"]
        )

    historical_utilisation_df_complete["percentage_utilisation_display"] = (
        historical_utilisation_df_complete["percentage_utilisation"].apply(lambda x: f"{x:.1%}")
        )

    historical_utilisation_df_complete.to_csv("historical_data/calculated/complete_utilisation_historical.csv")

    historical_utilisation_df_summary = (
        historical_utilisation_df_complete
        .groupby('callsign')['percentage_utilisation']
        .agg(['min', 'max', 'mean', 'median'])*100
        ).round(1)

    historical_utilisation_df_summary.to_csv("historical_data/calculated/complete_utilisation_historical_summary.csv")

    # print("==historical_utilisation_df_complete==")
    # print(historical_utilisation_df_complete)

    # print("==historical_utilisation_df_summary==")
    # print(historical_utilisation_df_summary)

    return historical_utilisation_df_complete, historical_utilisation_df_summary

def get_hist_util_fig(historical_utilisation_df_summary, callsign="H70", average="mean"):
    return historical_utilisation_df_summary[historical_utilisation_df_summary.index==callsign][average].values[0]

def make_RWC_utilisation_plot(historical_df_path="../historical_data/historical_monthly_resource_utilisation.csv"):
    historical_utilisation_df_complete, historical_utilisation_df_summary = (
                make_RWC_utilisation_dataframe(
                    historical_df_path=historical_df_path
                    )
                )
    fig = px.box(historical_utilisation_df_complete, x="percentage_utilisation", y="callsign")

    return fig

def make_SIMULATION_utilisation_headline_figure(vehicle_type, utilisation_df_overall):
    """
    Options:
        - helicopter
        - solo car
        - helicopter backup car
    """

    if vehicle_type == "helicopter":
        return utilisation_df_overall[utilisation_df_overall["vehicle_type"] == "helicopter"].mean(numeric_only=True)['perc_time_in_use']

    else:
        # assume anything with >= 1 entries in a callsign group is helicopter + backup car
        # NOTE: This assumption may not hold forever! It assumes
        vehicles_per_callsign_group = utilisation_df_overall.groupby('callsign_group').count()[['callsign']]

        if vehicle_type == "solo car":
            car_only = vehicles_per_callsign_group[vehicles_per_callsign_group['callsign'] == 1]
            return utilisation_df_overall[utilisation_df_overall["callsign_group"].isin(car_only.reset_index().callsign_group.values)].mean(numeric_only=True)['perc_time_in_use']

        elif vehicle_type == "helicopter backup car":
            backupcar_only = vehicles_per_callsign_group[vehicles_per_callsign_group['callsign'] == 2]
            utilisation_df_overall = utilisation_df_overall[utilisation_df_overall["vehicle_type"] == "car"]
            return utilisation_df_overall[utilisation_df_overall["callsign_group"].isin(backupcar_only.reset_index().callsign_group.values)].mean(numeric_only=True)['perc_time_in_use']

        else:
            print("Invalid vehicle type entered. Please use 'helicopter', 'solo car' or 'helicopter backup car'")


def prep_util_df_from_call_df(call_df):

    call_df['timestamp_dt'] = pd.to_datetime(call_df['timestamp_dt'], format='ISO8601')
    call_df['month_start'] = call_df['timestamp_dt'].dt.to_period('M').dt.to_timestamp()

    # print("==prep_util_df_from_call_df: call_df==")
    # print(call_df)

    jobs_counts_by_callsign_monthly_sim = call_df[~call_df['callsign'].isna()]

    # print("==jobs_counts_by_callsign_monthly_sim - prior to aggregation==")
    # print(jobs_counts_by_callsign_monthly_sim)

    jobs_counts_by_callsign_monthly_sim['callsign_group'] = jobs_counts_by_callsign_monthly_sim["callsign"].str.extract(r'(\d+)')

    jobs_counts_by_callsign_monthly_sim = (
        jobs_counts_by_callsign_monthly_sim
        .groupby(['run_number', 'month_start', 'callsign', 'callsign_group', 'vehicle_type'])['P_ID']
        .count().reset_index().rename(columns={'P_ID': 'jobs'})
        )

    # print("==jobs_counts_by_callsign_monthly_sim==")
    # print(jobs_counts_by_callsign_monthly_sim)

    all_combinations = pd.MultiIndex.from_product([
        jobs_counts_by_callsign_monthly_sim["month_start"].unique(),
        jobs_counts_by_callsign_monthly_sim["run_number"].unique(),
        jobs_counts_by_callsign_monthly_sim["callsign"].unique()
    ], names=["month_start", "run_number", "callsign"])

    # Reindex the dataframe to include missing callsigns
    jobs_counts_by_callsign_monthly_sim = jobs_counts_by_callsign_monthly_sim.set_index(
        ["month_start", "run_number", "callsign"]
    ).reindex(all_combinations, fill_value=0).reset_index()

    jobs_counts_by_callsign_monthly_sim['callsign_group'] = jobs_counts_by_callsign_monthly_sim['callsign'].str.extract(r'(\d+)')
    jobs_counts_by_callsign_monthly_sim['vehicle_type'] = jobs_counts_by_callsign_monthly_sim['callsign'].apply(lambda x: 'car' if "C" in x else 'helicopter')

    # Compute total jobs per callsign_group per month
    jobs_counts_by_callsign_monthly_sim["total_jobs_per_group"] = jobs_counts_by_callsign_monthly_sim.groupby(["month_start", "callsign_group", "run_number"])["jobs"].transform("sum")

    # Compute percentage of calls per row
    jobs_counts_by_callsign_monthly_sim["percentage_of_group"] = (jobs_counts_by_callsign_monthly_sim["jobs"] / jobs_counts_by_callsign_monthly_sim["total_jobs_per_group"]) * 100

    # Handle potential division by zero (if total_jobs_per_group is 0)
    jobs_counts_by_callsign_monthly_sim["percentage_of_group"] = jobs_counts_by_callsign_monthly_sim["percentage_of_group"].fillna(0)

    sim_averages = jobs_counts_by_callsign_monthly_sim.groupby(["callsign_group", "callsign", "vehicle_type"])[["percentage_of_group"]].mean().reset_index()

    return sim_averages


def make_SIMULATION_stacked_callsign_util_plot(call_df):
    # print("==make_SIMULATION_stacked_callsign_util_plot - call_df==")
    # print(call_df)

    sim_averages = prep_util_df_from_call_df(call_df)

    fig = px.bar(
        sim_averages,
        x="percentage_of_group",
        y="callsign_group",
        color="callsign",
        height=300
    )

    # Update axis labels and legend title
    fig.update_layout(
        yaxis=dict(
            title="Callsign Group",
            tickmode="linear",  # Ensures ticks appear at regular intervals
            dtick=1  # Set tick spacing to 1 unit
        ),
        xaxis=dict(
            title="Utilisation % within Callsign Group"
        ),
        legend_title="Callsign"
    )

    return fig

def create_UTIL_rwc_plot(call_df,
                        real_data_path="../historical_data/historical_jobs_per_month_by_callsign.csv"):

    #############
    # Prep real-world data
    #############
    jobs_by_callsign = pd.read_csv(real_data_path)
    jobs_by_callsign["month"] = pd.to_datetime(jobs_by_callsign["month"], dayfirst=True)

    jobs_by_callsign_long = jobs_by_callsign.melt(id_vars="month").rename(columns={"variable":"callsign", "value":"jobs"})

    all_combinations = pd.MultiIndex.from_product([
        jobs_by_callsign_long["month"].unique(),
        jobs_by_callsign_long["callsign"].unique()
    ], names=["month", "callsign"])

    # Reindex the dataframe to include missing callsigns
    jobs_by_callsign_long = jobs_by_callsign_long.set_index(
        ["month", "callsign"]
    ).reindex(all_combinations, fill_value=0).reset_index()

    jobs_by_callsign_long["callsign_group"] = jobs_by_callsign_long["callsign"].str.extract(r"(\d+)")
    jobs_by_callsign_long = jobs_by_callsign_long[~jobs_by_callsign_long["callsign_group"].isna()]
    jobs_by_callsign_long["vehicle_type"] = jobs_by_callsign_long["callsign"].apply(lambda x: "car" if "CC" in x else "helicopter")

    # Compute total jobs per callsign_group per month
    jobs_by_callsign_long["total_jobs_per_group"] = jobs_by_callsign_long.groupby(["month", "callsign_group"])["jobs"].transform("sum")

    # Compute percentage of calls per row
    jobs_by_callsign_long["percentage_of_group"] = (jobs_by_callsign_long["jobs"] / jobs_by_callsign_long["total_jobs_per_group"]) * 100

    # Handle potential division by zero (if total_jobs_per_group is 0)
    jobs_by_callsign_long["percentage_of_group"] = jobs_by_callsign_long["percentage_of_group"].fillna(0)


    ###############
    # Prep sim data
    ###############

    sim_averages = prep_util_df_from_call_df(call_df)

    # print("==utilisation_result_calculation.py - create_UTIL_rwc_plot - sim_averages==")
    # print(sim_averages)

    fig = go.Figure()

    # Bar chart (Simulation Averages)
    for idx, vehicle in enumerate(sim_averages["vehicle_type"].unique()):
        filtered_data = sim_averages[sim_averages["vehicle_type"] == vehicle]

        fig.add_trace(go.Bar(
            y=filtered_data["percentage_of_group"],
            x=filtered_data["callsign_group"],
            name=f"Simulated - {vehicle}",
            marker=dict(
            color=list(DAA_COLORSCHEME.values())[idx]),
            width=0.3,
            opacity=0.6,  # Same opacity for consistency
            text=["Simulated:<br>{:.1f}%".format(val) for val in filtered_data["percentage_of_group"]],  # Correct way            textposition="inside",  # Places text just above the x-axis
            insidetextanchor="start",  # Anchors text inside the bottom of the bar
            textfont=dict(
                        color='white'
                    ),
        ))

        fig.update_layout(
            title="<b>Comparison of Allocated Resources by Callsign Group</b><br>Simulation vs Historical Data",
            yaxis_title="Percentage of Jobs in Callsign Group<br>Tasked to Callsign",
            xaxis_title="Callsign Group",
            barmode='group',
            legend_title="Vehicle Type",
            height=600
        )

    for callsign in jobs_by_callsign_long["callsign"].unique():

        filtered_data = jobs_by_callsign_long[
            jobs_by_callsign_long["callsign"] == callsign
        ].groupby(["callsign_group", "vehicle_type", "callsign"])[["percentage_of_group"]].mean().reset_index()

        if filtered_data["callsign_group"].values[0] in ["70", "71"]:

            expected_x = float(filtered_data["callsign_group"].values[0])
            y_value = filtered_data['percentage_of_group'].values[0]
            expected_y =y_value - 1  # Position for the line

            if filtered_data["vehicle_type"].values[0] == "car":
                x_start = expected_x - 0.4
                x_end = expected_x
            else:
                x_start = expected_x
                x_end = expected_x + 0.4

            # Add dashed line
            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[expected_y, expected_y],
                    mode="lines",
                    name=f"Expected Level - {callsign}",
                    showlegend=False,
                    hoverinfo="all",
                    line=dict(dash='dash', color=DAA_COLORSCHEME['charcoal'])
                )
            )

            # Add text annotation above the line
            fig.add_trace(
                go.Scatter(
                    x=[(x_start + x_end) / 2],  # Center the text horizontally
                    y=[expected_y + 5],  # Slightly above the line
                    text=[f"Historical:<br>{y_value:.1f}%"],
                    mode="text",
                    textfont=dict(
                        color='black'
                    ),
                    showlegend=False  # Don't show in legend
                )
            )

    min_x = min(sim_averages["callsign_group"].astype('int').values)
    max_x = max(sim_averages["callsign_group"].astype('int').values)

    tick_vals = list(range(min_x, max_x + 1))  # Tick positions at integer values

    fig.update_layout(
        xaxis=dict(
            title = dict(font=dict(size=20)),
            tickfont = dict(size=25),
            tickmode='array',
            tickvals=tick_vals,  # Ensure ticks are at integer positions
            range=[min_x - 0.5, max_x + 0.5]  # Extend range to start 0.5 units earlier
        ),
        yaxis=dict(
            ticksuffix="%",
            title = dict(
                dict(font=dict(size=15))),
                   tickfont = dict(size=20),
                     range=[0, 100]),

        legend = dict(font = dict(size = 15)),
        legend_title = dict(font = dict(size = 20))
    )

    return fig




def create_callsign_group_split_rwc_plot(
        historical_data_path="historical_data/historical_monthly_totals_by_callsign.csv",
        run_data_path="data/run_results.csv",
        x_is_callsign_group=False
        ):

    jobs_by_callsign = pd.read_csv(historical_data_path)
    jobs_by_callsign["month"] = pd.to_datetime(jobs_by_callsign["month"], format="ISO8601")
    jobs_by_callsign["quarter"] = jobs_by_callsign["month"].dt.quarter
    jobs_by_callsign = jobs_by_callsign.melt(id_vars=["month","quarter"]).rename(columns={'variable':'callsign', 'value': 'jobs'})
    jobs_by_callsign["callsign_group"] = jobs_by_callsign["callsign"].str.extract(r"(\d+)")
    jobs_by_callsign_group_hist = jobs_by_callsign.groupby(['callsign_group', 'quarter'])["jobs"].sum().reset_index()
    # Group by quarter and compute total jobs per quarter
    quarter_totals_hist = jobs_by_callsign_group_hist.groupby('quarter')['jobs'].transform('sum')
    # Calculate proportion
    jobs_by_callsign_group_hist['proportion'] = jobs_by_callsign_group_hist['jobs'] / quarter_totals_hist
    jobs_by_callsign_group_hist['what'] = 'Historical'

    jobs_by_callsign_sim = pd.read_csv(run_data_path)
    jobs_by_callsign_sim = jobs_by_callsign_sim[jobs_by_callsign_sim["event_type"]=="resource_use"][["run_number", "P_ID", "time_type", "qtr"]]
    jobs_by_callsign_sim["callsign_group"] = jobs_by_callsign_sim["time_type"].str.extract(r"(\d+)")
    jobs_by_callsign_group_sim = jobs_by_callsign_sim.groupby(['qtr','callsign_group']).size().reset_index().rename(columns={'qtr': 'quarter', 0:'jobs'})
    # Group by quarter and compute total jobs per quarter
    quarter_totals_sim = jobs_by_callsign_group_sim.groupby('quarter')['jobs'].transform('sum')
    # Calculate proportion
    jobs_by_callsign_group_sim['proportion'] = jobs_by_callsign_group_sim['jobs'] / quarter_totals_sim
    jobs_by_callsign_group_sim['what'] = 'Simulated'

    full_df_callsign_group_counts = pd.concat([jobs_by_callsign_group_hist, jobs_by_callsign_group_sim])

    if not x_is_callsign_group:
        fig = px.bar(
            full_df_callsign_group_counts,
            title="Historical vs Simulated Split of Jobs Between Callsign Groups",
            color="callsign_group",
            y="proportion",
            x="what",
            barmode="stack",
            labels={"what": "", "proportion": "Percent of Jobs in Quarter",
                    "callsign_group": "Callsign Group"},
            facet_col="quarter",
            text=full_df_callsign_group_counts['proportion'].map(lambda p: f"{p:.0%}")  # format as percent
        )

        fig.update_layout(yaxis_tickformat=".0%")


        fig.update_traces(textposition='inside')  # You can also try 'auto' or 'outside'
        return fig

    else:
        fig = px.bar(
            full_df_callsign_group_counts,
            title="Historical vs Simulated Split of Jobs Between Callsign Groups",
            color="what",
            y="proportion",
            x="callsign_group",
            barmode="group",
            labels={"what": "", "proportion": "Percent of Jobs in Quarter",
                    "callsign_group": "Callsign Group"},
            facet_col="quarter",
            text=full_df_callsign_group_counts['proportion'].map(lambda p: f"{p:.0%}")  # format as percent
        )

        fig.update_layout(yaxis_tickformat=".0%")


        fig.update_traces(textposition='inside')  # You can also try 'auto' or 'outside'
        return fig

# --- Helper function to display vehicle metric ---
def display_vehicle_utilisation_metric(st_column, callsign_to_display, vehicle_type_label, icon_unicode,
                                    sim_utilisation_df, hist_summary_df,
                                    util_calc_module, current_quarto_string):
    """
    Displays the utilisation metrics for a given vehicle in a specified Streamlit column.
    Returns the updated quarto_string.
    """
    with st_column:
        with iconMetricContainer(key=f"{vehicle_type_label.lower()}_util_{callsign_to_display}", icon_unicode=icon_unicode, type="symbols"):
            matched_sim = sim_utilisation_df[sim_utilisation_df['callsign'] == callsign_to_display]
            if not matched_sim.empty:
                sim_util_fig = matched_sim['PRINT_perc'].values[0]
                sim_util_display = f"{sim_util_fig}"
            else:
                sim_util_fig = "N/A"
                sim_util_display = "N/A"

            current_quarto_string += f"\n\nAverage simulated {callsign_to_display} Utilisation was {sim_util_fig}\n\n"

            st.metric(f"Average Simulated {callsign_to_display} Utilisation",
                    sim_util_display,
                    border=True)

        # Get historical data
        hist_util_value = util_calc_module.get_hist_util_fig(
            hist_summary_df, callsign_to_display, "mean"
        )
        hist_util_value_display = f"{hist_util_value}%" if isinstance(hist_util_value, (int, float)) else hist_util_value

        hist_util_caption = f"*The historical average utilisation of {callsign_to_display} was {hist_util_value_display}*\n\n"
        current_quarto_string += hist_util_caption
        current_quarto_string += "\n\n---\n\n"
        st.caption(hist_util_caption)

    return current_quarto_string
