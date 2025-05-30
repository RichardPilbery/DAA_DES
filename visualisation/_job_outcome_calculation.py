"""
File containing all calculations and visualisations arelating to the patient and job outcomes/
results.

- Stand-down rates
- Patient outcomes

Covers variation within the simulation, and comparison with real world data.
"""


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
        ).copy()

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
        ).copy()

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
        ).copy()

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

    resource_preferred_outcome_by_hour = (
        run_results[run_results["event_type"]=="resource_preferred_outcome"]
        [["P_ID", "run_number", "care_cat", "time_type", "hour"]]
        .reset_index()
        .groupby(["time_type", "hour"])
        .size()
        .reset_index(name="count")
        ).copy()

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


                # ------- Calculate missed jobs in simulation --------- #

                # resource_requests = (
                #     results_all_runs[results_all_runs["event_type"] == "resource_request_outcome"]
                #     .copy()
                #     )

                # resource_requests["care_cat"] = (
                #     resource_requests.apply(
                #         lambda x: "REG - Helicopter Benefit"
                #         if x["heli_benefit"]=="y" and x["care_cat"]=="REG"
                #         else x["care_cat"], axis=1))

                # missed_jobs_care_cat_summary = (
                #     resource_requests[["care_cat", "time_type"]]
                #     .value_counts().reset_index(name="jobs")
                #     .sort_values(["care_cat", "time_type"]).copy()
                #     )

                # missed_jobs_care_cat_summary["jobs_average"] = (
                #     missed_jobs_care_cat_summary["jobs"] / st.session_state.number_of_runs_input)

                # missed_jobs_care_cat_summary["jobs_per_year_average"] = (
                #     (missed_jobs_care_cat_summary["jobs_average"] /
                #     float(st.session_state.sim_duration_input)) * 365
                #     ).round(0)

def get_missed_call_df(results_all_runs,
                       run_length_days, what="summary"):
    # Filter for relevant events

    resource_requests = (
        results_all_runs[results_all_runs["event_type"] == "resource_request_outcome"]
        .copy()
    )

    # Recode care_cat when helicopter benefit applies
    resource_requests["care_cat"] = resource_requests.apply(
        lambda x: "REG - Helicopter Benefit"
        if x["heli_benefit"] == "y" and x["care_cat"] == "REG"
        else x["care_cat"],
        axis=1
    )

    # Group by care_cat, time_type, and run_number to get jobs per run
    jobs_per_run = (
        resource_requests.groupby(["care_cat", "time_type", "run_number"])
        .size()
        .reset_index(name="jobs")
    )

    if what=="breakdown":
        jobs_per_run["jobs_per_year"] = (jobs_per_run["jobs"]/ run_length_days) * 365
        return jobs_per_run


    elif what=="summary":

        # Then aggregate to get average, min, and max per group
        missed_jobs_care_cat_summary = (
            jobs_per_run.groupby(["care_cat", "time_type"])
            .agg(
                jobs_average=("jobs", "mean"),
                jobs_min=("jobs", "min"),
                jobs_max=("jobs", "max")
            )
            .reset_index()
        )

        # Add annualised average per year
        missed_jobs_care_cat_summary["jobs_per_year_average"] = (
            (missed_jobs_care_cat_summary["jobs_average"] / run_length_days) * 365
        ).round(0)

        missed_jobs_care_cat_summary["jobs_per_year_min"] = (
            (missed_jobs_care_cat_summary["jobs_min"] / run_length_days) * 365
        ).round(0)

        missed_jobs_care_cat_summary["jobs_per_year_max"] = (
            (missed_jobs_care_cat_summary["jobs_max"] / run_length_days) * 365
        ).round(0)

        return missed_jobs_care_cat_summary

    else:
        raise("Invalid option passed. Allowed options for the *what* parameter are 'summary' or 'breakdown'")


def plot_missed_calls_boxplot(df_sim_breakdown, df_hist_breakdown, what="breakdown",
                              historical_yearly_missed_calls_estimate=None):

    full_df = pd.concat([df_sim_breakdown, df_hist_breakdown])

    full_df_no_resource_avail = full_df[full_df["time_type"]=="No Resource Available"]

    if what == "breakdown":
        category_order = ["CC", "EC", "REG - Helicopter Benefit", "REG"]

        fig = px.box(
            full_df_no_resource_avail,
            x="jobs_per_year",
            y="care_cat",
            color="what",
            points="all",  # or "suspectedoutliers"
            boxmode='group',
            height=800,
            labels={
                "jobs_per_year": "Estimated Average Jobs per Year",
                "care_cat": "Care Category",
                "what": "Simulation Results vs Simulated Historical Data"
            },
            category_orders={"care_cat": category_order},
        )

        fig.update_layout(
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center')
        )

    if what == "summary":

        full_df_no_resource_avail_per_run = (
            full_df_no_resource_avail
            .groupby(["run_number", "what"])[['jobs_per_year']]
            .sum().reset_index()
            )

        # Compute data bounds for x-axis
        x_min = full_df_no_resource_avail_per_run["jobs_per_year"].min()
        x_max = full_df_no_resource_avail_per_run["jobs_per_year"].max()
        padding = 0.20 * (x_max - x_min)
        x_range = [x_min - padding, x_max + padding]

        fig = px.box(
            full_df_no_resource_avail_per_run,
            x="jobs_per_year",
            y="what",
            color="what",
            points="all",
            boxmode='group',
            height=400
        )

        # Update x-axis range
        fig.update_layout(xaxis_range=x_range, showlegend=False)

        if historical_yearly_missed_calls_estimate is not None:
            # Add the dotted vertical line
            fig.add_vline(
                x=historical_yearly_missed_calls_estimate,
                line_dash="dot",
                line_color="black",
                line_width=2,
                annotation_text=f"Historical Estimate: {historical_yearly_missed_calls_estimate:.0f}",
                annotation_position="top"
            )

        # Step 1: Compute Q1 and Q3
        q_df = (
            full_df_no_resource_avail_per_run
            .groupby("what")["jobs_per_year"]
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .reset_index()
            .rename(columns={0.25: "q1", 0.5: "median", 0.75: "q3"})
        )

        # Step 2: Calculate IQR and upper whisker cap
        q_df["iqr"] = q_df["q3"] - q_df["q1"]
        q_df["upper_whisker_cap"] = q_df["q3"] + 1.5 * q_df["iqr"]

        # Step 3: Find the max non-outlier per group
        max_non_outliers = (
            full_df_no_resource_avail_per_run
            .merge(q_df[["what", "upper_whisker_cap"]], on="what")
        )
        max_non_outliers = (
            max_non_outliers[max_non_outliers["jobs_per_year"] <= max_non_outliers["upper_whisker_cap"]]
            .groupby("what")["jobs_per_year"]
            .max()
            .reset_index()
            .rename(columns={"jobs_per_year": "max_non_outlier"})
        )

        # Step 4: Merge with median data
        annot_df = pd.merge(q_df[["what", "median"]], max_non_outliers, on="what")

        # Step 5: Add annotations just to the right of the whisker
        for _, row in annot_df.iterrows():
            fig.add_annotation(
                x=row["max_non_outlier"] + padding * 0.1,
                y=row["what"],
                text=f"Median: {row['median']:.0f}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=0,
                font=dict(size=12, color="gray"),
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
            )


    return fig


def get_prediction_cc_patients_sent_ec_resource(run_results, run_duration_days):
    counts_df = run_results[run_results["event_type"]=="resource_use"][["run_number", 'hems_res_category', "care_cat"]].value_counts().reset_index()

    counts_df_summary = counts_df.groupby(["hems_res_category", "care_cat"])["count"].agg(["mean", "min", "max"]).reset_index()

    row_of_interest = counts_df_summary[(counts_df_summary["hems_res_category"]!="CC") & (counts_df_summary["care_cat"]=="CC")]

    return (row_of_interest["mean"].values[0]/run_duration_days)*365, (row_of_interest["min"].values[0]/run_duration_days)*365, (row_of_interest["max"].values[0]/run_duration_days)*365

def get_prediction_heli_benefit_patients_sent_car(run_results, run_duration_days):
    counts_df = run_results[run_results["event_type"]=="resource_use"][["run_number", "vehicle_type", "heli_benefit"]].value_counts().reset_index()

    counts_df_summary = counts_df.groupby(["vehicle_type", "heli_benefit"])["count"].agg(["mean", "min", "max"]).reset_index()

    row_of_interest = counts_df_summary[(counts_df_summary["vehicle_type"]=="car") & (counts_df_summary["heli_benefit"]=="y")]

    return (row_of_interest["mean"].values[0]/run_duration_days)*365, (row_of_interest["min"].values[0]/run_duration_days)*365, (row_of_interest["max"].values[0]/run_duration_days)*365
