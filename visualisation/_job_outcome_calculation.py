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
import numpy as np


def get_care_cat_counts_plot_sim(results_path="data/run_results.csv",
                        show_proportions=False):
    run_results = pd.read_csv(results_path)

    # Amend care category to reflect the small proportion of regular jobs assumed to have
    # a helicopter benefit
    run_results.loc[
        (run_results['heli_benefit'] == 'y') & (run_results['care_cat'] == 'REG'),
        'care_cat'
    ] = 'REG - helicopter benefit'

    care_cat_by_hour = (run_results[
        run_results["event_type"]=="patient_helicopter_benefit"]
        [["P_ID", "run_number", "care_cat", "hour"]].reset_index()
        .groupby(["hour", "care_cat"]).size()
        .reset_index(name="count")
        )
    # Calculate total per hour
    total_per_hour = care_cat_by_hour.groupby("hour")["count"].transform("sum")
    # Add proportion column
    care_cat_by_hour["proportion"] = care_cat_by_hour["count"] / total_per_hour

    title= "Care Category of calls in simulation by hour of day with EC/CC/Regular - Heli Benefit/Regular"

    if not show_proportions:
        fig = px.bar(care_cat_by_hour, x="hour", y="count", color="care_cat", title=title,
            category_orders={
                "care_cat": ["CC", "EC", "REG - helicopter benefit", "REG"]
            })

    # if show_proportions
    else:
        fig = px.bar(care_cat_by_hour, x="hour", y="proportion", color="care_cat", title=title,
            category_orders={
                "care_cat": ["CC", "EC", "REG - helicopter benefit", "REG"]
            })

    fig.update_layout(xaxis=dict(dtick=1))

    return fig

def get_care_cat_counts_plot_historic(historic_df_path="historical_data/historical_care_cat_counts.csv",
                        show_proportions=False):

    care_cat_by_hour_historic = pd.read_csv(historic_df_path)

    total_per_hour = care_cat_by_hour_historic.groupby("hour")["count"].transform("sum")
    # Add proportion column
    care_cat_by_hour_historic["proportion"] = care_cat_by_hour_historic["count"] / total_per_hour

    title = "Care Category of calls in historical data by hour of day with EC/CC/Regular - Heli Benefit/Regular"

    if not show_proportions:
        fig = px.bar(care_cat_by_hour_historic,
            x="hour", y="count", color="care_category",
            title=title,
            category_orders={
                "care_category": ["CC", "EC", "REG - helicopter benefit", "REG", "Unknown - DAA resource did not attend"]
            }
        )
    else:
        fig = px.bar(care_cat_by_hour_historic,
            x="hour", y="proportion", color="care_category",
            title=title,
            category_orders={
                "care_category": ["CC", "EC", "REG - helicopter benefit", "REG", "Unknown - DAA resource did not attend"]
            }
        )

    fig.update_layout(xaxis=dict(dtick=1))

    return fig

def get_care_cat_proportion_table(
        results_path="data/run_results.csv",
        historic_df_path="historical_data/historical_care_cat_counts.csv"):

    historical_value_counts_by_hour = pd.read_csv(historic_df_path)

    historical_counts_simple = (
        historical_value_counts_by_hour
        .groupby('care_category')['count'].sum()
        .reset_index()
        .rename(columns={'count': 'Historic Job Counts', 'care_category': 'Care Category'})
        )

    run_results = pd.read_csv(results_path)

    # Amend care category to reflect the small proportion of regular jobs assumed to have
    # a helicopter benefit
    run_results.loc[
        (run_results['heli_benefit'] == 'y') & (run_results['care_cat'] == 'REG'),
        'care_cat'
    ] = 'REG - helicopter benefit'

    care_cat_counts_sim = (run_results[
        run_results["event_type"]=="patient_helicopter_benefit"]
        [["P_ID", "run_number", "care_cat"]].reset_index()
        .groupby(["care_cat"]).size()
        .reset_index(name="count")
        )

    full_counts = historical_counts_simple.merge(
            care_cat_counts_sim
            .rename(columns={'care_cat': 'Care Category', 'count':'Simulated Job Counts'}),
            how="outer", on="Care Category"
        )

    # Calculate proportions by column
    full_counts = full_counts[full_counts["Care Category"] != "Unknown - DAA resource did not attend"].copy()

    full_counts["Historic Percentage"] = full_counts["Historic Job Counts"] / full_counts["Historic Job Counts"].sum()
    full_counts["Simulated Percentage"] = full_counts["Simulated Job Counts"] / full_counts["Simulated Job Counts"].sum()

    full_counts["Historic Percentage"] = full_counts["Historic Percentage"].apply(lambda x: f"{x:.1%}")
    full_counts["Simulated Percentage"] = full_counts["Simulated Percentage"].apply(lambda x: f"{x:.1%}")

    full_counts["Care Category"] = pd.Categorical(full_counts["Care Category"], ["CC", "EC", "REG - helicopter benefit", "REG"])

    return full_counts.sort_values('Care Category')

def get_preferred_outcome_by_hour(results_path="data/run_results.csv", show_proportions=False):
    run_results = pd.read_csv(results_path)
    resource_preferred_outcome_by_hour =(
        run_results[run_results["event_type"]=="resource_preferred_outcome"]
        [["P_ID", "run_number", "care_cat", "time_type", "hour"]].reset_index()
        .groupby(["time_type", "hour"]).size()
        .reset_index(name="count")
        )
    # Calculate total per hour
    total_per_hour = resource_preferred_outcome_by_hour.groupby("hour")["count"].transform("sum")
    # Add proportion column
    resource_preferred_outcome_by_hour["proportion"] = (
        resource_preferred_outcome_by_hour["count"] /
        total_per_hour
        )

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
    resource_preferred_outcome_by_hour["proportion"] = (
        resource_preferred_outcome_by_hour["count"] /
        total_per_hour
        )

    resource_preferred_outcome_by_hour["time_type"] = resource_preferred_outcome_by_hour["time_type"].apply(
    lambda x: textwrap.fill(x, width=25).replace("\n", "<br>")
)

    fig = px.bar(resource_preferred_outcome_by_hour, x="hour", y="proportion", facet_col="time_type",
                 facet_col_wrap=4, height=800, facet_col_spacing=0.05, facet_row_spacing=0.13)

    return fig



def plot_patient_outcomes(df, group_cols="vehicle_type",
                          outcome_col="hems_result",
                          plot_counts=False,
                          return_fig=True, run_df_path="data/run_results.csv"):
    df = pd.read_csv(run_df_path)

    patient_outcomes_df = (
        df[df["time_type"] == "HEMS call start"]
        [["P_ID", "run_number", "heli_benefit", "care_cat", "vehicle_type", "hems_result", "outcome"]]
        .reset_index(drop=True)
        .copy()
        )

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

    patient_outcomes_df_grouped_counts = calculate_grouped_proportions(
        patient_outcomes_df, group_cols, outcome_col
     )

    if return_fig:
        if plot_counts:
            y="count"
        else:
            y="proportion"

        fig = px.bar(patient_outcomes_df_grouped_counts,
                     color=group_cols, y=y, x=outcome_col,
                     barmode="group"
                     )

        return fig
    else:
        return patient_outcomes_df_grouped_counts
