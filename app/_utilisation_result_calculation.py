import _processing_functions
import pandas as pd
import plotly.express as px
from _app_utils import DAA_COLORSCHEME
import _vehicle_calculation

def make_utilisation_model_dataframe(path="../data/run_results.csv",
                                     params_path="../data/run_params_used.csv"):
    df = pd.read_csv(path)
    params_df = pd.read_csv(params_path)
    n_runs = len(df["run_number"].unique())

    # First get the dataframe of true availability hours
    # TODO: Incorporate servicing unavailability into this
    daily_availability, total_avail_hours, total_avail_minutes = (
        _vehicle_calculation.calculate_available_hours(
            params_df, rota_path="../data/hems_rota_used.csv"
        )
    )

    del daily_availability, total_avail_hours

    # Add callsign column if not already present in the dataframe passed to the function
    if 'callsign' not in df.columns:
        df = _processing_functions.make_callsign_column(df)

    # Restrict to only events in the event log where resource use was starting or ending
    resource_use_only = df[df["event_type"].isin(["resource_use", "resource_use_end"])]

    del df

    # Pivot to wide-format dataframe with one row per patient/call
    # and columns for start and end types
    resource_use_wide = (
        resource_use_only[["P_ID", "run_number", "event_type", "timestamp_dt",
                           "callsign_group", "vehicle_type", "callsign"]]
        .pivot(index=["P_ID", "run_number", "callsign_group", "vehicle_type", "callsign"],
               columns="event_type", values="timestamp_dt").reset_index()
               )

    del resource_use_only

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
        total_avail_minutes, on="callsign"
        )

    utilisation_df_per_run["perc_time_in_use"] = (
        utilisation_df_per_run["resource_use_duration"].astype(float) /
        # float(_processing_functions.get_param("sim_duration", params_df))
        utilisation_df_per_run["total_available_minutes_in_sim"].astype(float)
        )

    # Add column of nicely-formatted values to make printing values more streamlined
    utilisation_df_per_run["PRINT_perc"] = utilisation_df_per_run["perc_time_in_use"].apply(
        lambda x: f"{x:.1%}")

    # ============================================================ #
    # Calculage averge utilisation across simulation,
    # stratified by callsign group
    # ============================================================ #
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

    total_avail_minutes_per_csg = total_avail_minutes.groupby('callsign_group').head(1).drop(columns='callsign')
    total_avail_minutes_per_csg['callsign_group'] =  total_avail_minutes_per_csg['callsign_group'].astype('float')

    utilisation_df_per_run_by_csg = utilisation_df_per_run_by_csg.merge(
        total_avail_minutes_per_csg, on="callsign_group"
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
        total_avail_minutes, on="callsign"
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


def make_SIMULATION_utilisation_variation_plot(utilisation_df_per_run):
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
                 color_discrete_map={'Car': DAA_COLORSCHEME["blue"],
                                     "Helicopter": DAA_COLORSCHEME["red"]})
        .update_layout(
                    xaxis={
                        "tickformat": ".0%"  # Formats as percentage with no decimal places
                    }))

    # TODO: Add indications of good/bad territory for utilisation levels
    return fig

def make_SIMULATION_utilisation_summary_plot(utilisation_df_overall):
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

    fig = (px.bar(utilisation_df_overall,
                    y="perc_time_in_use",
                    x="callsign",
                    color="vehicle_type",
                    title="Average Resource Utilisation Across All Simulation Runs",
                    labels={
                     "callsign": "Callsign",
                     "perc_time_in_use": "Average Percentage of Available<br>Time Spent in Use",
                     "vehicle_type": "Vehicle Type"
                 },
                 color_discrete_map={'Car': DAA_COLORSCHEME["blue"],
                                     "Helicopter": DAA_COLORSCHEME["red"]})
            .update_layout(
                yaxis={
                    "tickformat": ".0%"  # Formats as percentage with no decimal places
                })
                )
    # TODO: Add indications of good/bad territory for utilisation levels
    return fig

def make_RWC_utilisation_dataframe(utilisation_df):
    pass


def make_RWC_utilisation_plot():
    pass

def make_SIMULATION_utilisation_headline_figure(callsign):
    pass

def make_SIMULATION_fleet_utilisation_headline_figure():
    pass
