import streamlit as st
st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

import platform
import os
# Data processing imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
# Plotting
import plotly.express as px
import platform
import plotly.graph_objects as go

import gc

from scipy.stats import ks_2samp

import subprocess

import streamlit.components.v1 as components

import _app_utils
from _app_utils import DAA_COLORSCHEME, iconMetricContainer, file_download_confirm, \
                        get_text, get_text_sheet, to_military_time, format_sigfigs

# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Simulation imports
from des_parallel_process import runSim, parallelProcessJoblib, collateRunResults
from utils import Utils

from _state_control import setup_state

from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.metric_cards import style_metric_cards

import visualisation._job_count_calculation as _job_count_calculation
import visualisation._vehicle_calculation as _vehicle_calculation
import visualisation._utilisation_result_calculation as _utilisation_result_calculation
import visualisation._job_time_calcs as _job_time_calcs
import visualisation._process_analytics as _process_analytics
import visualisation._job_outcome_calculation as _job_outcome_calculation

setup_state()

# Pull in required font
poppins_script = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
"""

quarto_string = ""

text_df = get_text_sheet("model")

# Avoid reading from utils due to odd issues it seems to be introducing
# TODO: Explore why this is happening in more detail
# u = Utils()
# rota = u.HEMS_ROTA
#rota = pd.read_csv("actual_data/HEMS_ROTA.csv")
# SERVICING_SCHEDULE = pd.read_csv('actual_data/service_schedules_by_model.csv')

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("Run a Simulation")

with col2:
    st.image("app/assets/daa-logo.svg", width=200)

with st.sidebar:
    generate_downloadable_report = st.toggle("Generate a Downloadable Summary of Results", False,
                                             help="This will generate a downloadable report. This can slow down the running of the model, so turn this off if you don't need it.")

    debug_messages = st.toggle("Turn on debugging messages", False,
                               help="This will turn on display of messages in the developer terminal")

    _app_utils.summary_sidebar(quarto_string=quarto_string)

with stylable_container(key="run_buttons",
            css_styles=f"""
                    button {{
                            background-color: {DAA_COLORSCHEME['blue']};
                            color: white;
                            border-color: white;
                        }}
                        """
            ):
    button_run_pressed = st.button(
        "Click this button to run the simulation with the selected parameters",
        icon=":material/play_circle:")

if not st.session_state["visited_setup_page"]:
    if not button_run_pressed:
        with stylable_container(key="warning_buttons",
            css_styles=f"""
                    button {{
                            background-color: {DAA_COLORSCHEME['orange']};
                            color: {DAA_COLORSCHEME['charcoal']};
                            border-color: white;
                        }}
                        """
            ):
            if st.button("**Warning**\n\nYou haven't set up any parameters - default parameters will be used!\n\nClick this button to go to the parameter page, or click the blue button above\n\nif you are happy to use the default model parameters",
                        icon=":material/warning:"):
                    st.switch_page("setup.py")

if button_run_pressed:

    gc.collect()

    progress_text = "Simulation in progress. Please wait."
    # This check is a way to guess whether it's running on
    # Streamlit community cloud
    if platform.processor() == '':
        my_bar = st.progress(0, text=progress_text)

    with st.spinner('Simulating the system...'):

        # If running on community cloud, parallelisation will not work
        # so run instead using the runSim function sequentially
        if platform.processor() == '':
            print("Running sequentially")
            results = []

            for run in range(st.session_state.number_of_runs_input):
                run_results = runSim(
                        run = run,
                        total_runs = st.session_state.number_of_runs_input,
                        sim_duration = float(st.session_state.sim_duration_input * 24 * 60),
                        warm_up_time = float(st.session_state.warm_up_duration * 60),
                        sim_start_date = datetime.combine(
                            datetime.strptime(st.session_state.sim_start_date_input, '%Y-%m-%d').date(),
                            datetime.strptime(st.session_state.sim_start_time_input, '%H:%M').time(),
                            ),
                        amb_data=st.session_state.amb_data,
                        demand_increase_percent=float(st.session_state.overall_demand_mult)/100.0,
                        activity_duration_multiplier=float(st.session_state.activity_duration_multiplier),
                        print_debug_messages=debug_messages,
                        master_seed=st.session_state.master_seed
                    )

                results.append(
                    run_results
                    )

                my_bar.progress((run+1)/st.session_state.number_of_runs_input, text=progress_text)

            # Turn into a single dataframe when all runs complete
            results_all_runs = pd.concat(results)

            my_bar.empty()

        # If running locally, use parallel processing function to speed up execution significantly
        else:
            print("Running in parallel")
            print(f"st.session_state.overall_demand_mult: {st.session_state.overall_demand_mult}")
            print(f"st.session_state.sim_duration_input: {st.session_state.sim_duration_input}")
            parallelProcessJoblib(
                        total_runs = st.session_state.number_of_runs_input,
                        sim_duration = float(st.session_state.sim_duration_input * 24 * 60),
                        warm_up_time = float(st.session_state.warm_up_duration * 60),
                        sim_start_date = datetime.combine(
                            datetime.strptime(st.session_state.sim_start_date_input, '%Y-%m-%d').date(),
                            datetime.strptime(st.session_state.sim_start_time_input, '%H:%M').time(),
                            ),
                        amb_data = st.session_state.amb_data,
                        demand_increase_percent=float(st.session_state.overall_demand_mult)/100.0,
                        activity_duration_multiplier=float(st.session_state.activity_duration_multiplier),
                        print_debug_messages=debug_messages,
                        master_seed=st.session_state.master_seed

            )
            collateRunResults()
            results_all_runs = pd.read_csv("data/run_results.csv")


        tab_names = [
            "Simulation Results Summary",
            "Key Visualisations",
            "Comparing Model with Historic Data",
            "Additional Outputs",
            "Download Output"
            ]



        if st.session_state.create_animation_input:
            tab_names.append("Animation")
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                tab_names
            )
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                tab_names
            )

        with tab5:
            report_message = st.empty()

            report_message.info("Generating Report...")


        def get_job_count_df():
            return _job_count_calculation.make_job_count_df(params_path="data/run_params_used.csv",
                                                            path="data/run_results.csv")

        def get_params_df():
            return pd.read_csv("data/run_params_used.csv")

        with tab1:
            quarto_string += "# Key Metrics\n\n"

            averaged_string = f"All Metrics are averaged across {st.session_state.number_of_runs_input} simulation runs"

            quarto_string += "*"
            quarto_string += averaged_string
            quarto_string += "*\n\n"

            st.info(averaged_string)

            report_message = st.empty()

            historical_utilisation_df_complete, historical_utilisation_df_summary = (
                _utilisation_result_calculation.make_RWC_utilisation_dataframe(
                    historical_df_path="historical_data/historical_monthly_resource_utilisation.csv",
                    rota_path="actual_data/HEMS_ROTA.csv",
                    callsign_path="actual_data/callsign_registration_lookup.csv",
                    service_path="data/service_dates.csv"
                    )
                )

            print(historical_utilisation_df_summary)

            t1_col1, t1_col2 = st.columns(2)

            with t1_col1:
                perc_unattended, perc_unattended_normalised = _vehicle_calculation.get_perc_unattended_string_normalised(results_all_runs)

                quarto_string += "## Calls Not Attended\n\n"

                quarto_string += f"Across these runs of the simulation, on average a DAAT Resource was unable to attend {perc_unattended} calls\n\n"

                with iconMetricContainer(key="nonattend_metric", icon_unicode="e61f", family="outline"):
                    st.metric("Average Number of Calls DAAT Resource Couldn't Attend",
                            perc_unattended,
                            border=True)
                    missed_calls_hist_string = _job_count_calculation.plot_historical_missed_jobs_data(format="string")
                    st.caption(f"**{perc_unattended_normalised}**")
                    st.caption(f"*This compares to an average of {missed_calls_hist_string:.1f}% of calls missed historically*")

                    missed_calls_description = get_text("missed_calls_description", text_df)

                    st.caption(missed_calls_description)

                    quarto_string += missed_calls_description

                    # with st.expander("View Breakdown"):
                    #     outcome_df = _vehicle_calculation.resource_allocation_outcomes(results_all_runs)
                    #     outcome_df["Count"] = (outcome_df["Count"]/st.session_state.number_of_runs_input).round(0)
                    #     outcome_df.rename(columns={'Count':'Mean Calls per Simulation Run'}, inplace=True)
                    #     st.dataframe(outcome_df)

            with t1_col2:
                quarto_string += "\n\n## Resource Utilisation"
                resource_use_wide, utilisation_df_overall, utilisation_df_per_run, utilisation_df_per_run_by_csg = (
                    _utilisation_result_calculation.make_utilisation_model_dataframe(
                    path="data/run_results.csv",
                    params_path="data/run_params_used.csv",
                    service_path="data/service_dates.csv",
                    callsign_path="actual_data/callsign_registration_lookup.csv",
                    rota_path="actual_data/HEMS_ROTA.csv"
                ))

                print(utilisation_df_overall)

                t1_col_2_a, t1_col_2_b = st.columns(2)
                with t1_col_2_a:
                    with iconMetricContainer(key="helo_util", icon_unicode="f60c", type="symbols"):
                        h70_util_fig = utilisation_df_overall[utilisation_df_overall['callsign']=='H70']['PRINT_perc'].values[0]

                        quarto_string += f"\n\nAverage simulated H70 Utilisation was {h70_util_fig}\n\n"

                        st.metric("Average Simulated H70 Utilisation",
                                h70_util_fig,
                                border=True)

                    h70_hist = _utilisation_result_calculation.get_hist_util_fig(
                        historical_utilisation_df_summary, "H70", "mean"
                    )

                    h70_hist_util_fig = f"*The historical average utilisation of H70 was {h70_hist}%*\n\n"
                    quarto_string += h70_hist_util_fig

                    quarto_string += f"\n\n---\n\n"

                    st.caption(h70_hist_util_fig)

                with t1_col_2_b:
                    with iconMetricContainer(key="helo_util", icon_unicode="f60c", type="symbols"):
                        h71_util_fig = utilisation_df_overall[utilisation_df_overall['callsign']=='H71']['PRINT_perc'].values[0]

                        quarto_string += f"\n\nAverage simulated H71 Utilisation was {h71_util_fig}\n\n"

                        st.metric("Average Simulated H71 Utilisation",
                                h71_util_fig,
                                border=True)

                    h71_hist = _utilisation_result_calculation.get_hist_util_fig(
                        historical_utilisation_df_summary, "H71", "mean"
                    )
                    h71_hist_util_fig = f"*The historical average utilisation of H71 was {h71_hist}%*\n\n"
                    quarto_string += h71_hist_util_fig
                    st.caption(h71_hist_util_fig)

                    quarto_string += f"\n\n---\n\n"

                st.caption(get_text("helicopter_utilisation_description", text_df))

            st.divider()

            cars = ["CC70", "CC71", "CC72"]

            car_metric_cols = st.columns(len(cars))

            # historical_utilisation_df_summary.index = historical_utilisation_df_summary.index.str.replace("CC", "C")

            for idx, col in enumerate(car_metric_cols):
                with col:
                    car_callsign = cars[idx]

                    with iconMetricContainer(key="car_util", icon_unicode="eb3c", type="symbols"):
                        print(utilisation_df_overall)
                        car_util_fig = utilisation_df_overall[utilisation_df_overall['callsign']==car_callsign]['PRINT_perc'].values[0]

                        quarto_string += f"\n\nAverage simulated {car_callsign} utilisation was {car_util_fig}\n\n"

                        st.metric(f"Average Simulated {car_callsign} Utilisation",
                                car_util_fig,
                                border=True)

                    car_util_hist = _utilisation_result_calculation.get_hist_util_fig(
                        historical_utilisation_df_summary, car_callsign, "mean"
                    )
                    car_util_fig_hist = f"*The historical average utilisation of {car_callsign} was {car_util_hist}%*\n\n"

                    quarto_string += car_util_fig_hist

                    quarto_string += f"\n\n---\n\n"

                    st.caption(car_util_fig_hist)


            t1_col3, t1_col4 = st.columns(2)

        with tab2:
            tab_2_1, tab_2_2, tab_2_3, tab_2_4 = st.tabs(["'Missed' Calls", "Resource Utilisation", "Split of Jobs by Callsign Group", "CC and EC Benefit"])

            with tab_2_1:
                @st.fragment
                def missed_jobs():
                    show_proportions_per_hour = st.toggle("Show as proportion of jobs missed per hour", value=False )
                    by_quarter = st.toggle("Stratify results by quarter", value=False)
                    st.plotly_chart(_job_count_calculation.plot_missed_jobs(
                        show_proportions_per_hour=show_proportions_per_hour,
                        by_quarter=by_quarter
                        ))

                missed_jobs()

                st.caption("""
## What is this plot showing?

This chart shows how often helicopter emergency medical services (HEMS) were either available and sent or unavailable during each hour of the day. It compares simulated data (used for testing or planning purposes) with historical data (what actually happened in the past).

- The top chart shows the simulated job counts by hour.

- The bottom chart shows the historical job counts by hour.

## What do the colours mean?

Each bar is split into:

- Dark blue: When a HEMS vehicle (either helicopter or car) was available and sent to a job.

- Light blue: When no HEMS was available for a job received during that time period.

If more of the bar is light blue, this means that there were more jobs in that hour that were not responded to by a HEMS resource due to no HEMS resource being available at the time.

## Using this plot for model quality assurance

If the default historical parameters are being used, this plot can be used to judge if the simulation is mirroring reality well.
In this case, we would be looking for two things to be consistent across the top and bottom plots:

- the overall pattern of bar heights per hour (reflecting the total number of jobs being received each hour)
- the split between dark and light blue per hour (reflecting how often a resource is or is not available to respond to a job received in that hour)

""")

            with tab_2_2:
                @st.fragment
                def create_utilisation_rwc_plot():
                    call_df = get_job_count_df()

                    fig_utilisation = _utilisation_result_calculation.create_UTIL_rwc_plot(
                        call_df,
                        real_data_path="historical_data/historical_monthly_totals_by_callsign.csv"
                        )

                    fig_utilisation.write_html("app/fig_outputs/fig_utilisation.html")#, post_script = poppins_script)#,full_html=False, include_plotlyjs='cdn')

                    st.plotly_chart(
                        fig_utilisation
                    )

                create_utilisation_rwc_plot()

                historical_monthly_totals_df = pd.read_csv("historical_data/historical_monthly_totals_by_callsign.csv")
                historical_monthly_totals_df["month"] = pd.to_datetime(historical_monthly_totals_df["month"], format="%Y-%m-%d")

                st.caption(f"""
This plot shows the split within a callsign group of resources that are sent on jobs.
Bars within a callsign group will sum to 100%.

Dotted lines indicate the average historical allocation seen of resources within a callsign group,
averaged over {len(historical_monthly_totals_df)} months, drawing on data
from {historical_monthly_totals_df.month.min().strftime("%B %Y")}
to {historical_monthly_totals_df.month.max().strftime("%B %Y")}.

If the simulation is using the default parameters, we would expect the dotted lines to be roughly level with the top of the
relevant bars - though being out by a few % is not too unusual due to the natural variation that occurs across
simulation runs.

If the simulation is not using the default parameters, we would not expect the output to match the historical data, but you may
    wish to consider the historical split as part of your decision making.
                """)

                with tab_2_3:
                    @st.fragment
                    def plot_callsign_group_split():
                        x_is_callsign_group = st.toggle("Plot callsign group on the horizontal axis",
                                                        value=False)

                        st.plotly_chart(
                            _utilisation_result_calculation.create_callsign_group_split_rwc_plot(
                            x_is_callsign_group=x_is_callsign_group
                            )
                        )

                    plot_callsign_group_split()

                with tab_2_4:

                    @st.fragment
                    def plot_cc_ec_split():
                        show_proportions_care_cat_plot = st.toggle("Show Proportions", False)

                        st.plotly_chart(
                            _job_outcome_calculation.get_care_cat_counts(
                                show_proportions=show_proportions_care_cat_plot
                                )
                        )

                    plot_cc_ec_split()



        with tab3:

            # tab_3_1, tab_3_2, tab_3_3, tab_3_4, tab_3_5 = st.tabs([
            tab_3_1, tab_3_2, tab_3_3, tab_3_4, tab_3_5, tab_3_6 = st.tabs([
                "Jobs per Month",
                "Jobs by Hour of Day",
                "Jobs by Day of Week",
                "Jobs per Day - Distribution",
                "Job Durations - Overall",
                "Job Durations - Split"
                ])

            with tab_3_1:
                @st.fragment
                def plot_monthly_jobs():
                    call_df = get_job_count_df()

                    mj_1, mj_2 = st.columns(2)

                    show_real_data = mj_1.toggle(
                        "Compare with Real Data",
                        value=True,
                        disabled=False)

                    show_individual_runs = mj_2.toggle("Show Individual Simulation Runs", value=False)

                    if show_real_data:
                        historical_view_method = st.radio(
                            "Choose Historical Data Display Method",
                            ["Range", "Individual Lines"],
                            horizontal=True
                            )
                        if historical_view_method == "Range":
                            show_historical_individual_years = False
                        else:
                            show_historical_individual_years = True
                    else:
                        show_historical_individual_years = False

                    fig_monthly_calls = _job_count_calculation.plot_monthly_calls(
                            call_df,
                            show_individual_runs=show_individual_runs,
                            use_poppins=True,
                            show_historical=show_real_data,
                            show_historical_individual_years=show_historical_individual_years,
                            historical_monthly_job_data_path="historical_data/historical_monthly_totals_all_calls.csv",
                            job_count_col="inc_date"
                            )

                    _job_count_calculation.plot_monthly_calls(
                            call_df,
                            show_individual_runs=show_individual_runs,
                            use_poppins=False,
                            show_historical=show_real_data,
                            show_historical_individual_years=show_historical_individual_years,
                            historical_monthly_job_data_path="historical_data/historical_monthly_totals_all_calls.csv",
                            job_count_col="inc_date"
                            ).write_html("app/fig_outputs/fig_monthly_calls.html",full_html=False, include_plotlyjs='cdn')#, post_script = poppins_script)


                    return st.plotly_chart(
                        fig_monthly_calls
                    )

                plot_monthly_jobs()
                st.caption("""
Note that only full months in the simulation are included in this plot.
Partial months are excluded for ease of interpretation.
                           """)

            with tab_3_2:

                @st.fragment
                def plot_jobs_per_hour():
                    call_df = get_job_count_df()
                    params_df = get_params_df()
                    help_jph = get_text("help_jobs_per_hour", text_df)
                    jph_1, jph_2, jph_3, jph_4 = st.columns(4)

                    display_historic_jph = jph_1.toggle(
                        "Display Historic Data",
                        value=True
                        )
                    average_per_month = jph_2.toggle(
                        "Display Average Calls Per Month",
                        value=True,
                        help= help_jph
                        )

                    display_advanced = jph_3.toggle("Display Advanced Plot", value=False)

                    if not display_advanced:
                        display_error_bars_bar = jph_4.toggle("Display Variation")
                    else:
                        display_error_bars_bar = False

                    fig_hour_of_day = _job_count_calculation.plot_hourly_call_counts(
                        call_df, params_df,
                        average_per_month=average_per_month,
                        box_plot=display_advanced,
                        show_error_bars_bar=display_error_bars_bar,
                        use_poppins=True,
                        show_historical=display_historic_jph,
                        historical_data_path="historical_data/historical_monthly_totals_by_hour_of_day.csv"
                        )

                    _job_count_calculation.plot_hourly_call_counts(
                        call_df, params_df,
                        average_per_month=average_per_month,
                        box_plot=display_advanced,
                        show_error_bars_bar=display_error_bars_bar,
                        use_poppins=False,
                        show_historical=display_historic_jph,
                        historical_data_path="historical_data/historical_monthly_totals_by_hour_of_day.csv"
                        ).write_html("app/fig_outputs/fig_hour_of_day.html",full_html=False, include_plotlyjs='cdn')#, post_script = poppins_script)


                    st.plotly_chart(
                        fig_hour_of_day
                    )

                plot_jobs_per_hour()

            with tab_3_3:

                @st.fragment
                def plot_jobs_per_day():
                    call_df = get_job_count_df()
                    params_df = get_params_df()
                    # help_jph = get_text("help_jobs_per_hour", text_df)
                    jpd_1, jpd_2, jpd_3, jpd_4 = st.columns(4)

                    display_historic_jph_pd = jpd_1.toggle(
                        "Display Historic Data",
                        value=True,
                         key="historic_pd"
                        )

                    average_per_month_pd = jpd_2.toggle(
                        "Display Average Calls Per Day",
                        value=True,
                        # help= help_jph,
                        key="average_pd"

                        )

                    display_advanced_pd = jpd_3.toggle("Display Advanced Plot", value=False,
                                                        key="advanced_pd")

                    if not display_advanced_pd:
                        display_error_bars_bar_pd = jpd_4.toggle("Display Variation",
                                                        key="variation_pd")
                    else:
                        display_error_bars_bar_pd = False

                    fig_day_of_week = _job_count_calculation.plot_daily_call_counts(
                        call_df, params_df,
                        average_per_month=average_per_month_pd,
                        box_plot=display_advanced_pd,
                        show_error_bars_bar=display_error_bars_bar_pd,
                        use_poppins=True,
                        show_historical=display_historic_jph_pd,
                        historical_data_path="historical_data/historical_monthly_totals_by_day_of_week.csv"
                        )

                    _job_count_calculation.plot_daily_call_counts(
                        call_df, params_df,
                        average_per_month=average_per_month_pd,
                        box_plot=display_advanced_pd,
                        show_error_bars_bar=display_error_bars_bar_pd,
                        use_poppins=False,
                        show_historical=display_historic_jph_pd,
                        historical_data_path="historical_data/historical_monthly_totals_by_day_of_week.csv"
                        ).write_html("app/fig_outputs/fig_day_of_week.html",full_html=False, include_plotlyjs='cdn')#, post_script = poppins_script)


                    st.plotly_chart(
                        fig_day_of_week
                    )


                plot_jobs_per_day()

            #######################################
            # Histogram of calls received per day #
            #######################################

            with tab_3_4:
                @st.fragment()
                def plot_days_with_job_count_hist():
                    call_df = get_job_count_df()
                    call_df["day"] = pd.to_datetime(call_df["timestamp_dt"]).dt.date
                    daily_call_counts = (
                        call_df.groupby(['run_number', 'day'])['P_ID'].agg("count")
                        .reset_index().rename(columns={"P_ID": "Calls per Day"})
                        )

                    historical_daily_calls = pd.read_csv(
                        "historical_data/historical_daily_calls_breakdown.csv"
                        )

                    # Create histogram with two traces
                    call_count_hist = go.Figure()

                    # Simulated data
                    call_count_hist.add_trace(go.Histogram(
                        x=daily_call_counts["Calls per Day"],
                        name="Simulated",
                        histnorm='percent',
                        xbins=dict( # bins used for histogram
                            start=0.0,
                            end=max(daily_call_counts["Calls per Day"])+1,
                            size=1.0
                        ),
                        opacity=0.75
                    ))

                    # Historical data
                    call_count_hist.add_trace(go.Histogram(
                        x=historical_daily_calls["calls_in_day"],
                        xbins=dict( # bins used for histogram
                            start=0.0,
                            end=max(historical_daily_calls["calls_in_day"])+1,
                            size=1.0
                        ),
                        name="Historical",
                        histnorm='percent',
                        opacity=0.75
                    ))

                    # Update layout
                    call_count_hist.update_layout(
                        title="Distribution of Jobs Per Day: Simulated vs Historical",
                        barmode='overlay',
                        bargap=0.03,
                        xaxis=dict(
                            tickmode='linear',
                            tick0=0,
                            dtick=1
                        )
                    )

                    # Save and display
                    call_count_hist.write_html("app/fig_outputs/daily_calls_dist_histogram.html",
                                               full_html=False, include_plotlyjs='cdn')

                    call_count_hist.update_layout(font=dict(family="Poppins", size=18, color="black"))

                    st.plotly_chart(call_count_hist)

                    st.caption("""
This plot looks at the number of days across all repeats of the simulation where each given number of calls was observed (i.e. on how many days was one call received, two calls, three calls, and so on).
                           """)

                    statistic, p_value = ks_2samp(daily_call_counts["Calls per Day"],
                                                  historical_daily_calls["calls_in_day"])

                    if p_value > 0.05:
                        st.success(f"""There is no statistically significant difference between
                                 the distributions of call data from historical data and the
                                 simulation (p = {format_sigfigs(p_value)})

                                 This means that the pattern of calls produced by the simulation
                                 matches the pattern seen in the real-world data —
                                 for example, the frequency or variability of daily calls
                                 is sufficiently similar to what has been observed historically.
                                 """)
                    else:
                        ks_text_string_sig = f"""
There is a statistically significant difference between the
distributions of call data from historical data and
the simulation (p = {format_sigfigs(p_value)}).

This means that the pattern of calls produced by the simulation
does not match the pattern seen in the real-world data —
for example, the frequency or variability of daily calls
may be different.

The simulation may need to be adjusted to better
reflect the patterns of demand observed historically.

"""

                        if statistic < 0.1:
                            st.info(ks_text_string_sig + f"""Although the difference is
                                    statistically significant, the actual magnitude
                                    of the difference (D = {format_sigfigs(statistic)}) is small.
                                    This suggests the simulation's call volume pattern is reasonably
                                    close to reality.
                                    """)

                        elif statistic < 0.2:
                            st.warning(ks_text_string_sig + f"""The KS statistic (D = {format_sigfigs(statistic)})
                                    indicates a moderate difference in
                                    distribution. You may want to review the simulation model to
                                    ensure it adequately reflects real-world variability.
                                    """)

                        else:
                                st.error(ks_text_string_sig + f"""The KS statistic (D = {format_sigfigs(statistic)})
                                   suggests a large difference in call volume patterns.
                                   The simulation may not accurately reflect historical
                                   demand and may need adjustment.
                                    """)




                plot_days_with_job_count_hist()

            ##############################################
            # Historical Job Durations - Overall Summary #
            ##############################################

            with tab_3_5:

                @st.fragment
                def create_job_duration_plot():
                    historical_time_df = _job_time_calcs.get_historical_times_breakdown(
                                'historical_data/historical_job_durations_breakdown.csv'
                                )

                    simulated_job_time_df = _job_time_calcs.get_total_times_model(
                                get_summary=False,
                                path="data/run_results.csv",
                                params_path="data/run_params_used.csv",
                                rota_path="actual_data/HEMS_ROTA.csv",
                                service_path="data/service_dates.csv",
                                callsign_path="actual_data/callsign_registration_lookup.csv"
                                )

                    plot_violin=st.toggle("Violin Plot?", value=False)

                    # Create plot for inclusion in streamlit
                    fig_job_durations_historical =  _job_time_calcs.plot_historical_job_duration_vs_simulation_overall(
                            historical_activity_times=historical_time_df,
                            utilisation_model_df=simulated_job_time_df,
                            use_poppins = True,
                            write_to_html = True,
                            html_output_filepath = "app/fig_outputs/fig_job_durations_historical.html",
                            violin=plot_violin
                            )

                    # Include job durations plot in streamlit app
                    st.plotly_chart(
                    fig_job_durations_historical
                        )

                    st.caption("""
    This plot looks at the total amount of time each resource was in use during the simulation.

    All simulated points are represented in the box plots.

    The blue bars give an indication of the historical averages. We would expect the median - the
    central horizontal line within the box portion of the box plots - to fall within the blue box for
    each resource type, and likely to be fairly central within that region.
    """)

                    historical_time_df_cars_only = historical_time_df[historical_time_df["vehicle_type"] == "car"]
                    historical_time_df_helos_only = historical_time_df[historical_time_df["vehicle_type"] == "helicopter"]

                    simulated_job_time_df_cars_only = simulated_job_time_df[simulated_job_time_df["vehicle_type"] == "car"]
                    simulated_job_time_df_helos_only = simulated_job_time_df[simulated_job_time_df["vehicle_type"] == "helicopter"]

                    _job_time_calcs.calculate_ks_for_job_durations(
                        historical_data_series=historical_time_df_helos_only[historical_time_df_helos_only["name"]=="total_duration"]["value"],
                        simulated_data_series=simulated_job_time_df_helos_only["resource_use_duration"],
                        what="helicopters"
                        )

                    _job_time_calcs.calculate_ks_for_job_durations(
                        historical_data_series=historical_time_df_cars_only[historical_time_df_cars_only["name"]=="total_duration"]["value"],
                        simulated_data_series=simulated_job_time_df_cars_only["resource_use_duration"],
                        what="cars"
                        )

                create_job_duration_plot()

            ############################
            # Historical Job Durations - Breakdown #
            ############################

            with tab_3_6:
                st.plotly_chart(
                    _job_time_calcs.plot_time_breakdown()
                )

                st.caption("""
This chart is comparing how long different stages of emergency jobs take in real life (called Historical) versus how long they take in a computer simulation (called Simulated).

The idea is to check if the simulation is realistic by seeing if it behaves similarly to what actually happened in the past.

Each job has several stages:

- Time allocation: Time from when the call was made to when a vehicle was assigned.
- Time mobile: Time from assignment to when the vehicle started moving.
- Time to scene: Travel time to the scene.
- Time on scene: Time spent at the scene.
- Time to hospital: Travel time to the hospital (if applicable).
- Time to clear: Time from hospital drop-off (or leaving the scene, if no patient transport undertaken) to when the vehicle is ready for the next job.

These stages are shown for two types of vehicles:

- Cars (top row) - including both helicopter backup cars and standalone vehicles
- Helicopters (bottom row)

## How to Read the Boxes

- Each blue box shows the range of times for that job stage—how long it usually takes.
- The dark blue boxes are the simulated times, and the light blue ones are the historical (real) times.
- Taller boxes or longer “whiskers” (lines) mean more **variation** in how long that stage takes.
- If the boxes and whiskers for simulated and historical data overlap a lot, that means the simulation is doing a good job of copying reality.
                           """)

        with tab4:

            st.caption("""
This tab contains visualisations to help model authors do additional checks into the underlying functioning of the model.

Most users will not need to look at the visualisations in this tab.
            """)

            tab_4_1, tab_4_2, tab_4_3, tab_4_4 = st.tabs(["Debug Resources", "Debug Events",
                                                          "Process Analytics", "Process Analytics - Resources"])

            with tab_4_1:
                st.subheader("Resource Use")

                resource_use_events_only = results_all_runs[
                    (results_all_runs["event_type"] == "resource_use") |
                    (results_all_runs["event_type"] == "resource_use_end")].reset_index(drop=True).copy()

                # Accounting for odd bug being seen in streamlit community cloud
                if 'P_ID' not in resource_use_events_only.columns:
                    resource_use_events_only = resource_use_events_only.reset_index()


                @st.fragment
                def resource_use_exploration_plots():

                    run_select_ruep = st.selectbox("Choose the run to show",
                            resource_use_events_only["run_number"].unique()
                        )

                    # colour_by_cc_ec = st.toggle("Colour the plot by CC/EC/REG patient benefit",
                    #                          value=True)

                    show_outline = st.toggle("Show an outline to help debug overlapping calls",
                                             value=False)



                    with st.expander("Click here to see the timings of resource use"):
                        st.dataframe(
                            resource_use_events_only[resource_use_events_only["run_number"] == run_select_ruep]
                            )

                        st.dataframe(
                            resource_use_events_only[resource_use_events_only["run_number"] == run_select_ruep]
                            [['callsign', 'callsign_group', 'registration']]
                            .value_counts()
                            )

                        st.dataframe(resource_use_events_only[resource_use_events_only["run_number"] == run_select_ruep]
                                        [["P_ID", "time_type", "timestamp_dt", "event_type"]]
                                        .melt(id_vars=["P_ID", "time_type", "event_type"],
                                        value_vars="timestamp_dt").drop_duplicates())

                        resource_use_wide = (resource_use_events_only[resource_use_events_only["run_number"] == run_select_ruep]
                            [["P_ID", "time_type", "timestamp_dt", "event_type", "registration", "care_cat"]].drop_duplicates()
                            .pivot(columns="event_type", index=["P_ID","time_type", "registration", "care_cat"], values="timestamp_dt").reset_index())

                        # get the number of resources and assign them a value
                        resources = resource_use_wide.time_type.unique()
                        resources = np.concatenate([resources, ["No Resource Available"]])
                        resource_dict = {resource: index for index, resource in enumerate(resources)}

                        missed_job_events = results_all_runs[
                            (results_all_runs["event_type"] == "resource_request_outcome") &
                            (results_all_runs["time_type"] == "No Resource Available")
                            ].reset_index(drop=True).copy()

                        missed_job_events = missed_job_events[missed_job_events["run_number"]==run_select_ruep][["P_ID", "time_type", "timestamp_dt", "event_type", "registration", "care_cat"]].drop_duplicates()
                        missed_job_events["event_type"] = "resource_use"

                        missed_job_events_end = missed_job_events.copy()
                        missed_job_events_end["event_type"] = "resource_use_end"
                        missed_job_events_end["timestamp_dt"] = pd.to_datetime(missed_job_events_end["timestamp_dt"]) + timedelta(minutes=5)

                        missed_job_events_full = pd.concat([missed_job_events, missed_job_events_end])
                        missed_job_events_full["registration"] = "No Resource Available"

                        missed_job_events_full_wide = missed_job_events_full.pivot(columns="event_type", index=["P_ID","time_type", "registration", "care_cat"], values="timestamp_dt").reset_index()

                        resource_use_wide = pd.concat([resource_use_wide, missed_job_events_full_wide]).reset_index(drop=True)

                        resource_use_wide["y_pos"] = resource_use_wide["time_type"].map(resource_dict)

                        # Convert time types to numerical values
                        # resource_use_wide["y_pos"] = resource_use_wide["time_type"].map(resource_dict)

                        # Compute duration for each resource use
                        resource_use_wide["resource_use_end"] = pd.to_datetime(resource_use_wide["resource_use_end"])
                        resource_use_wide["resource_use"] = pd.to_datetime(resource_use_wide["resource_use"])

                        # Convert time to numeric (e.g., seconds since the first event)
                        time_origin = resource_use_wide["resource_use"].min()  # Set the reference time
                        # resource_use_wide["resource_use_numeric"] = (resource_use_wide["resource_use"] - time_origin).dt.total_seconds()
                        # resource_use_wide["resource_use_end_numeric"] = (resource_use_wide["resource_use_end"] - time_origin).dt.total_seconds()

                        # Compute duration
                        resource_use_wide["duration"] = resource_use_wide["resource_use_end"] - resource_use_wide["resource_use"]

                        # Compute duration as seconds and multiply by 1000 (to account for how datetime axis
                        # is handled in plotly
                        resource_use_wide["duration_seconds"] = (resource_use_wide["resource_use_end"] - resource_use_wide["resource_use"]).dt.total_seconds()*1000
                        resource_use_wide["duration_minutes"] = resource_use_wide["duration_seconds"] / 1000 / 60
                        resource_use_wide["duration_minutes"] = resource_use_wide["duration_minutes"].round(1)

                        resource_use_wide["callsign_group"] = resource_use_wide["time_type"].str.extract("(\d+)")

                        resource_use_wide = resource_use_wide.sort_values(["callsign_group", "time_type"])

                        st.dataframe(resource_use_wide)

                        ######################################
                        # Load in the servicing schedule df
                        ######################################
                        service_schedule = pd.read_csv("data/service_dates.csv")
                        service_schedule = service_schedule.merge(pd.read_csv("actual_data/callsign_registration_lookup.csv"))
                        # Convert to appropriate datatypes
                        service_schedule["service_end_date"] = pd.to_datetime(service_schedule["service_end_date"])
                        service_schedule["service_start_date"] = pd.to_datetime(service_schedule["service_start_date"])
                        # Calculate duration for plotting and for hover text
                        service_schedule["duration_seconds"] = (service_schedule["service_end_date"] - service_schedule["service_start_date"]).dt.total_seconds()*1000
                        service_schedule["duration_days"] = (service_schedule["duration_seconds"] / 1000) / 60 / 60 / 24
                        # Match y position of the resource in the rest of the plot
                        print(f"resource_dict: {resource_dict}")
                        service_schedule["y_pos"] = service_schedule["callsign"].map(resource_dict)
                        # Limit the dataframe to only contain servicing
                        service_schedule = service_schedule[(service_schedule["service_start_date"] <= resource_use_wide.resource_use.max()) &
                                        (service_schedule["service_end_date"] >= resource_use_wide.resource_use.min())]

                        st.dataframe(service_schedule)

                    # # Add cc/ec/reg lookup
                    # cc_ec_reg_colour_lookup = {
                    #     'CC': 0,
                    #     'EC': 1,
                    #     'REG': 2
                    # }

                    # Create figure
                    resource_use_fig = go.Figure()

                    # Add horizontal bars using actual datetime values
                    for idx, callsign in enumerate(resource_use_wide.time_type.unique()):
                        callsign_df = resource_use_wide[resource_use_wide["time_type"]==callsign]
                        # print(f"==callsign_df - {callsign} - for resource use debugging plot==")
                        # print(callsign_df.head(5))

                        service_schedule_df = service_schedule[service_schedule["callsign"]==callsign]

                        # Add in hatched boxes showing the servicing periods
                        if len(service_schedule_df) > 0:
                            resource_use_fig.add_trace(go.Bar(
                                x=service_schedule_df["duration_seconds"],  # Duration (Timedelta)
                                y=service_schedule_df["y_pos"],
                                base=service_schedule_df["service_start_date"],  # Start time as actual datetime
                                orientation="h",
                                width=0.6,
                                marker_pattern_shape="x",
                                marker=dict(color="rgba(63, 63, 63, 0.30)",
                                            line=dict(color="black", width=1)
                                            ),
                                name=f"Servicing = {callsign}",
                                customdata=service_schedule_df[['callsign','duration_days','service_start_date', 'service_end_date', 'registration']],
                                hovertemplate="Servicing %{customdata[0]} (registration %{customdata[4]}) lasting %{customdata[1]} days (%{customdata[2]|%a %-e %b %Y} to %{customdata[3]|%a %-e %b %Y})<extra></extra>"
                            ))

                        # if colour_by_cc_ec:
                        #     cc_ec_status = callsign_df["care_cat"].values[0]
                        #     marker_val = dict(color=list(DAA_COLORSCHEME.values())[cc_ec_reg_colour_lookup[cc_ec_status]])

                        if show_outline:
                            marker_val=dict(color=list(DAA_COLORSCHEME.values())[idx],
                                        line=dict(color="#FFA400", width=0.2))
                        else:
                            marker_val = dict(color=list(DAA_COLORSCHEME.values())[idx])

                        # Add in boxes showing the duration of individual calls
                        resource_use_fig.add_trace(go.Bar(
                            x=callsign_df["duration_seconds"],  # Duration (Timedelta)
                            y=callsign_df["y_pos"],
                            base=callsign_df["resource_use"],  # Start time as actual datetime
                            orientation="h",
                            width=0.4,
                            marker=marker_val,
                            name=callsign,
                            customdata=callsign_df[['resource_use','resource_use_end','time_type', 'duration_minutes', 'registration', 'care_cat']],
                            hovertemplate="Response to %{customdata[5]} call from %{customdata[2]} (registration %{customdata[4]}) lasting %{customdata[3]} minutes (%{customdata[0]|%a %-e %b %Y %H:%M} to %{customdata[1]|%a %-e %b %Y %H:%M})<extra></extra>"
                            #customdata=callsign_df[['resource_use','resource_use_end','time_type', 'duration_minutes']],
                            #hovertemplate="Response from %{customdata[2]} lasting %{customdata[3]} minutes (%{customdata[0]|%a %-e %b %Y %H:%M} to %{customdata[1]|%a %-e %b %Y %H:%M})<extra></extra>"

                        ))

                    # Layout tweaks
                    resource_use_fig.update_layout(
                        title="Resource Use Over Time",
                        barmode='overlay',
                        xaxis=dict(
                            title="Time",
                            type="date",  # Ensures proper datetime scaling with zoom
                            # tickformat="%b %d, %Y",  # Default formatting (months & days)
                        ),
                        yaxis=dict(
                            title="Callsign",
                            tickmode="array",
                            tickvals=list(resource_dict.values()),
                            ticktext=list(resource_dict.keys()),
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

                    resource_use_fig.write_html("app/fig_outputs/resource_use_fig.html",full_html=False, include_plotlyjs='cdn')#, post_script = poppins_script)

                    st.plotly_chart(
                        resource_use_fig
                    )

                resource_use_exploration_plots()

                st.caption("""
This visual shows the resource use of each resource throughout the simulation.

Grey hatched boxes indicate the time the resource was away for servicing.

- For H70 (g-daas), it is assumed that H71 (g-daan) will be reallocated the callsign H70 during the
service period for g-daas. Therefore, for the H70 line, we would expect calls to continue being allocated
to H70 during its service period, **but we would expect H71 to consequently show no activity in that period.**

- For the servicing of H71 (g-daan), it is assumed that g-daan will be unavailable during that period
and no callsign reallocation will occur, so we would anticipate no activity occurring for H71 during that period.

CC70 and CC71 are backup vehicles, for use in the event that their associated helicopter cannot fly
for any reason (pilot unavailability, servicing, etc.).

It should be the case that resources from the same callsign group (H70 & CC70, H71 & CC71) cannot ever be allocated
to a job at the same time, as it is assumed that a single crew is available for each callsign group.

Unavailability of cars due to servicing is not modelled; cars are assumed to always be available.

*The handles at the bottom of the plot can be used to zoom in to a shorter period of time, allowing
you to more clearly see patterns of resource use. The '1m, 6m, YTD, 1y' buttons at the top of the plot
can also be used to adjust the chosen time period. Double click on the plot or click on the 'reset axes'
button at the top right - which will only appear when hovering over the plot - to reset to looking at
the overall time period.*
            """)

                st.subheader("Jobs per Day - By Callsign")

                st.plotly_chart(_job_count_calculation.plot_jobs_per_callsign())

                st.subheader("Minutes per day on Shift")

                daily_availability_df = (
                    pd.read_csv("data/daily_availability.csv")
                    .melt(id_vars="month")
                    .rename(columns={"value":"theoretical_availability", "variable": "callsign"})
                    )

                st.plotly_chart(
                    px.bar(daily_availability_df, x="month", y="theoretical_availability", facet_row="callsign")
                )

            st.subheader("Jobs Outcome by Category/Preference")

            @st.fragment
            def plot_preferred_outcome_by_hour():
                show_proportions_job_outcomes_by_hour = st.toggle("Show Proportions", False, key="show_proportions_job_outcomes_by_hour")
                st.plotly_chart(_job_outcome_calculation.get_preferred_outcome_by_hour(show_proportions=show_proportions_job_outcomes_by_hour))

            plot_preferred_outcome_by_hour()

            st.plotly_chart(_job_outcome_calculation.get_facet_plot_preferred_outcome_by_hour())

            with tab_4_2:
                st.subheader("Event Overview")

                # @st.fragment
                # def event_overview_plot():
                #     runs_to_display_eo = st.multiselect("Choose the runs to display", results_all_runs["run_number"].unique(), default=1)

                #     events_over_time_df = results_all_runs[results_all_runs["run_number"].isin(runs_to_display_eo)]

                #     # Fix to deal with odd community cloud indexing bug
                #     if 'P_ID' not in events_over_time_df.columns:
                #         events_over_time_df = events_over_time_df.reset_index()

                #     events_over_time_df['time_type'] = events_over_time_df['time_type'].astype('str')

                #     fig = px.scatter(
                #             events_over_time_df,
                #             x="timestamp_dt",
                #             y="time_type",
                #             # facet_row="run_number",
                #             # showlegend=False,
                #             color="time_type",
                #             height=800,
                #             title="Events Over Time - By Run"
                #             )

                #     fig.update_traces(marker=dict(size=3, opacity=0.5))

                #     fig.update_layout(yaxis_title="", # Remove y-axis label
                #                       yaxis_type='category',
                #                       showlegend=False)
                #     # Remove facet labels
                #     fig.for_each_annotation(lambda x: x.update(text=""))

                #     st.plotly_chart(
                #         fig,
                #             use_container_width=True
                #         )

                # event_overview_plot()

                # Fix to deal with odd community cloud indexing bug
                if 'P_ID' not in results_all_runs.columns:
                    results_all_runs = results_all_runs.reset_index()


                st.plotly_chart(
                        px.line(
                            results_all_runs[results_all_runs["time_type"]=="arrival"],
                            x="timestamp_dt",
                            y="P_ID",
                            color="run_number",
                            height=800,
                            title="Cumulative Arrivals Per Run"),
                            use_container_width=True
                        )

                st.subheader("Event Counts")
                st.write(f"Period: {st.session_state.sim_duration_input} days")

                event_counts_df =  (pd.DataFrame(
                        results_all_runs[["run_number", "time_type"]].value_counts()).reset_index()
                        .pivot(index="run_number", columns="time_type", values="count")
                )
                event_counts_long = event_counts_df.reset_index(drop=False).melt(id_vars="run_number")

                @st.fragment
                def event_funnel_plot():

                    hems_events_initial = ["arrival", "HEMS call start", "HEMS allocated to call", "HEMS mobile",
                                    # "HEMS stood down en route",
                                    "HEMS on scene",
                                    # "HEMS patient treated (not conveyed)",
                                    "HEMS leaving scene",
                                    "HEMS arrived destination",
                                    "HEMS clear"]

                    hems_events = st.multiselect("Choose the events to show",
                                                 event_counts_long["time_type"].unique(),
                                                 hems_events_initial)

                    run_select = st.multiselect("Choose the runs to show",
                                                 event_counts_long["run_number"].unique(),
                                                 1)

                    return st.plotly_chart(
                            px.funnel(
                                event_counts_long[(event_counts_long["time_type"].isin(hems_events)) &
                                                  (event_counts_long["run_number"].isin(run_select))  ],
                                facet_col="run_number",
                                x="value",
                                y="time_type",
                                category_orders={"time_type": hems_events[::-1]}

                        )
                    )

                event_funnel_plot()

                @st.fragment
                def patient_viz():
                    st.subheader("Per-patient journey exploration")

                    patient_filter = st.selectbox("Select a patient", results_all_runs.P_ID.unique())

                    tab_list =  st.tabs([f"Run {i+1}" for i in range(st.session_state.number_of_runs_input)])

                    for idx, tab in enumerate(tab_list):
                        p_df = results_all_runs[
                                    (results_all_runs.P_ID==patient_filter) &
                                    (results_all_runs.run_number==idx+1)]

                        p_df['time_type'] = p_df['time_type'].astype('str')

                        fig = px.scatter(
                                p_df,
                                x="timestamp_dt",
                                y="time_type",
                                color="time_type")

                        fig.update_layout(yaxis_type='category')

                        tab.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"p_viz_{patient_filter}_{idx}"
                        )

                patient_viz()

            with tab_4_3:
                _process_analytics.create_event_log("data/run_results.csv")

                print("Current working directory:", os.getcwd())

                # This check is a way to guess whether it's running on
                # Streamlit community cloud
                if platform.processor() == '':
                    try:
                        process1 = subprocess.Popen(["Rscript", "app/generate_bupar_outputs.R"],
                                                    stdout=subprocess.PIPE,
                                                    stderr=subprocess.PIPE,
                                                    text=True,
                                                    cwd="app")

                    except:
                        # Get absolute path to the R script
                        script_path = Path(__file__).parent / "generate_bupar_outputs.R"
                        st.write(f"Trying path: {script_path}" )

                        process1 = subprocess.Popen(["Rscript", str(script_path)],
                                                    stdout=subprocess.PIPE,
                                                    stderr=subprocess.PIPE,
                                                    text=True)

                else:
                    result = subprocess.run(["Rscript", "app/generate_bupar_outputs.R"],
                                            capture_output=True, text=True)
                try:
                    st.subheader("Process - Absolute Frequency")
                    st.image("visualisation/absolute_frequency.svg")
                except:
                    st.warning("Process maps could not be generated")

                try:
                    # st.html("visualisation/anim_process.html")
                    components.html("visualisation/anim_process.html")
                except:
                    st.warning("Animated Process maps could not be generated")

                try:
                    # st.subheader("Process - Absolute Cases")
                    # st.image("visualisation/absolute_case.svg")

                    st.subheader("Performance - Average (Mean) Transition and Activity Times")
                    st.image("visualisation/performance_mean.svg")

                    st.subheader("Performance - Maximum Transition and Activity Times")
                    st.image("visualisation/performance_max.svg")

                    st.subheader("Activity - Processing Time - activity")
                    st.image("visualisation/processing_time_activity.svg")

                    st.subheader("Activity - Processing Time - Resource/Activity")
                    st.image("visualisation/processing_time_resource_activity.svg")
                except:
                    st.warning("Process maps could not be generated")


            with tab_4_4:
                try:
                    st.subheader("Activities - by Resource")
                    st.image("visualisation/relative_resource_level.svg")
                except:
                    st.warning("Animated process maps could not be generated")

                try:
                    # st.html("visualisation/anim_resource_level.html")
                    components.html("visualisation/anim_resource_level.html")
                except:
                    st.warning("Animated process maps could not be generated")


        with tab5:
            if generate_downloadable_report:
                try:
                    with open("app/fig_outputs/quarto_text.txt", "w") as text_file:
                        text_file.write(quarto_string)

                    msg = _app_utils.generate_quarto_report(run_quarto_check=False)

                    # print(msg)

                    if msg == "success":
                        report_message.success("Report Available for Download")

                except:
                    ## error message
                    report_message.error(f"Report cannot be generated - please speak to a developer")
