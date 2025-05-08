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

    if not show_proportions:
        fig = px.bar(care_cat_by_hour, x="hour", y="count", color="care_cat")
        return fig

    # if show_proportions
    else:
        fig = px.bar(care_cat_by_hour, x="hour", y="proportion", color="care_cat")
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

    fig = px.bar(resource_preferred_outcome_by_hour, x="hour", y="proportion", facet_col="time_type", facet_col_wrap=4, height=800, facet_col_spacing=0.05, facet_row_spacing=0.1)

    return fig
