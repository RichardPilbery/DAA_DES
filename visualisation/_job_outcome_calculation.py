"""
File containing all calculations and visualisations arelating to the patient and job outcomes/
results.

- Stand-down rates
- Patient outcomes

Covers variation within the simulation, and comparison with real world data.
"""


import pandas as pd
import plotly.express as px
import textwrap


def get_care_cat_counts(results_path="data/run_results.csv",
                        show_proportions=False):
    run_results = pd.read_csv(results_path)

    care_cat_by_hour = run_results[run_results["time_type"]=="arrival"][["P_ID", "run_number", "care_cat", "hour"]].reset_index().groupby(["hour", "care_cat"]).size().reset_index(name="count")
    # Calculate total per hour
    total_per_hour = care_cat_by_hour.groupby("hour")["count"].transform("sum")
    # Add proportion column
    care_cat_by_hour["proportion"] = care_cat_by_hour["count"] / total_per_hour

    title= "Calls in simulation by hour of day with EC/CC/Regular Care Category"

    if not show_proportions:
        fig = px.bar(care_cat_by_hour, x="hour", y="count", color="care_cat", title=title)
        return fig

    # if show_proportions
    else:
        fig = px.bar(care_cat_by_hour, x="hour", y="proportion", color="care_cat", title=title)
        return fig


def get_preferred_outcome_by_hour(results_path="data/run_results.csv", show_proportions=False):
    run_results = pd.read_csv(results_path)
    resource_preferred_outcome_by_hour = run_results[run_results["event_type"]=="resource_preferred_outcome"][["P_ID", "run_number", "care_cat", "time_type", "hour"]].reset_index().groupby(["time_type", "hour"]).size().reset_index(name="count")
    # Calculate total per hour
    total_per_hour = resource_preferred_outcome_by_hour.groupby("hour")["count"].transform("sum")
    # Add proportion column
    resource_preferred_outcome_by_hour["proportion"] = resource_preferred_outcome_by_hour["count"] / total_per_hour

    if not show_proportions:
        fig = px.bar(resource_preferred_outcome_by_hour, x="hour", y="count", color="time_type")

    else:
        fig = px.bar(resource_preferred_outcome_by_hour, x="hour", y="proportion", color="time_type")

    return fig


def get_facet_plot_preferred_outcome_by_hour(results_path="data/run_results.csv"):
    run_results = pd.read_csv(results_path)
    resource_preferred_outcome_by_hour = run_results[run_results["event_type"]=="resource_preferred_outcome"][["P_ID", "run_number", "care_cat", "time_type", "hour"]].reset_index().groupby(["time_type", "hour"]).size().reset_index(name="count")
    # Calculate total per hour
    total_per_hour = resource_preferred_outcome_by_hour.groupby("hour")["count"].transform("sum")
    # Add proportion column
    resource_preferred_outcome_by_hour["proportion"] = resource_preferred_outcome_by_hour["count"] / total_per_hour

    resource_preferred_outcome_by_hour["time_type"] = resource_preferred_outcome_by_hour["time_type"].apply(
    lambda x: textwrap.fill(x, width=25).replace("\n", "<br>")
)

    fig = px.bar(resource_preferred_outcome_by_hour, x="hour", y="proportion", facet_col="time_type",
                 facet_col_wrap=4, height=800, facet_col_spacing=0.05, facet_row_spacing=0.13)

    return fig



def plot_patient_outcomes(group_cols="vehicle_type", outcome_col="hems_result",
                          plot_counts=False,
                          return_fig=True, run_df_path="data/run_results.csv"):
    df = pd.read_csv(run_df_path)

    patient_outcomes_df = df[df["time_type"] == "HEMS call start"][["P_ID", "run_number", "heli_benefit", "care_cat", "vehicle_type", "hems_result", "outcome"]].reset_index(drop=True)

    def calculate_grouped_proportions(df, group_cols, outcome_col):
        """
        Calculate counts and proportions of an outcome column grouped by one or more columns.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        group_cols (str or list of str): Column(s) to group by (e.g., 'care_cat', 'vehicle_type').
        outcome_col (str): The name of the outcome column (e.g., 'hems_result').

        Returns:
        pd.DataFrame: A DataFrame with counts and proportions of outcome values per group.
        """
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        count_df = df.value_counts(group_cols + [outcome_col]).reset_index().sort_values(group_cols + [outcome_col])
        count_df.rename(columns={0: "count"}, inplace=True)

        # Calculate the total count per group (excluding the outcome column)
        total_per_group = count_df.groupby(group_cols)["count"].transform("sum")
        count_df["proportion"] = count_df["count"] / total_per_group

        return count_df

    patient_outcomes_df_grouped_counts = calculate_grouped_proportions(patient_outcomes_df, group_cols, outcome_col)

    if return_fig:
        if plot_counts:
            y="count"
        else:
            y="proportion"

        fig = px.bar(patient_outcomes_df_grouped_counts, color=group_cols, y=y, x=outcome_col, barmode="group")

        return fig
    else:
        return patient_outcomes_df_grouped_counts
