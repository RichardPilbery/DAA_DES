import streamlit as st
st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

import platform
# Data processing imports
import pandas as pd
import numpy as np
from datetime import datetime
import re
# Plotting
import plotly.express as px
import plotly.graph_objects as go
from vidigi.animation import animate_activity_log, generate_animation
from vidigi.prep import reshape_for_animations, generate_animation_df
import _job_count_calculation
import _vehicle_calculation
import _utilisation_result_calculation
import _job_time_calcs
import _app_utils
from _app_utils import DAA_COLORSCHEME, iconMetricContainer, file_download_confirm, \
                        get_text, get_text_sheet, to_military_time

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

setup_state()

poppins_script = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
"""

quarto_string = ""

text_df = get_text_sheet("model")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("Run a Simulation")

with col2:
    st.image("app/assets/daa-logo.svg", width=200)

with st.sidebar:
    with stylable_container(css_styles="""
hr {
    border-color: #a6093d;
    background-color: #a6093d;
    color: #a6093d;
    height: 1px;
  }
""", key="hr"):
        st.divider()
    if 'number_of_runs_input' in st.session_state:
        with stylable_container(key="green_buttons",
            css_styles=f"""
                    button {{
                            background-color: {DAA_COLORSCHEME['teal']};
                            color: white;
                            border-color: white;
                        }}
                        """
            ):
            if st.button("Want to change some parameters? Click here.", type="primary", icon=":material/display_settings:"):
                st.switch_page("setup.py")
        st.subheader("Model Input Summary")
        quarto_string += "## Model Input Summary\n\n"

        num_helos_string = f"Number of Helicopters: {st.session_state.num_helicopters}"
        quarto_string += num_helos_string
        quarto_string += "\n\n"
        st.write(num_helos_string)

        # Avoid reading from utils due to odd issues it seems to be introducing
        # TODO: Explore why this is happening in more detail
        # u = Utils()
        # rota = u.HEMS_ROTA
        rota = pd.read_csv("actual_data/HEMS_ROTA.csv")

        for helicopter in rota[rota["vehicle_type"]=="helicopter"]["callsign"].unique():
            per_callsign_rota = rota[rota["callsign"]==helicopter]
            helicopter_rota_string = f"""
{helicopter} is an {per_callsign_rota["model"].values[0]} and runs
from {to_military_time(per_callsign_rota["summer_start"].values[0])}
to {to_military_time(per_callsign_rota["summer_end"].values[0])} in summer
and {to_military_time(per_callsign_rota["winter_start"].values[0])}
to {to_military_time(per_callsign_rota["winter_end"].values[0])} in winter.
"""
            # quarto_string += "🚁 "
            quarto_string += helicopter_rota_string
            st.caption(helicopter_rota_string)

        num_cars_string = f"Number of **Extra** (non-backup) Cars: {st.session_state.num_cars}"
        quarto_string += num_cars_string
        quarto_string += "\n\n"
        st.write(num_cars_string)
        callsign_group_counts = rota['callsign_group'].value_counts().reset_index()
        backup_cars_only = list(callsign_group_counts[callsign_group_counts['count']==1]['callsign_group'].values)


        for car in rota[rota["callsign_group"].isin(backup_cars_only)]["callsign"]:
            per_callsign_rota = rota[rota["callsign"]==car]
            car_rota_string = f"""
{car} is a {per_callsign_rota["model"].values[0]} and runs
from {to_military_time(per_callsign_rota["summer_start"].values[0])}
to {to_military_time(per_callsign_rota["summer_end"].values[0])} in summer
and {to_military_time(per_callsign_rota["winter_start"].values[0])}
to {to_military_time(per_callsign_rota["winter_end"].values[0])} in winter.
"""
            # quarto_string += "🚗 "
            quarto_string += car_rota_string
            st.caption(car_rota_string)


        if st.session_state.demand_adjust_type == "Overall Demand Adjustment":
            if st.session_state.overall_demand_mult == 100:
                demand_adjustment_string = f"Demand is based on historically observed demand with no adjustments."
            elif st.session_state.overall_demand_mult < 100:
                demand_adjustment_string = f"Modelled demand is {100-st.session_state.overall_demand_mult}% less than historically observed demand."
            elif st.session_state.overall_demand_mult > 100:
                demand_adjustment_string = f"Modelled demand is {st.session_state.overall_demand_mult-100}% more than historically observed demand."

            st.write(demand_adjustment_string)
            quarto_string += demand_adjustment_string
            quarto_string += "\n\n"

        # TODO: Add this in if we decide seasonal demand adjustment is a thing that's wanted
        elif st.session_state.demand_adjust_type == "Per Season Demand Adjustment":
            pass

        elif st.session_state.demand_adjust_type == "Per AMPDS Code Demand Adjustment":
            pass

        else:
            st.error("TELL A DEVELOPER: Check Conditional Code for demand modifier in model.py")


        with stylable_container(css_styles="""
hr {
    border-color: #a6093d;
    background-color: #a6093d;
    color: #a6093d;
    height: 1px;
  }
""", key="hr"):
            st.divider()

        replication_string = f"The model will run {st.session_state.number_of_runs_input} replications of {st.session_state.sim_duration_input} days, starting from {datetime.strptime(st.session_state.sim_start_date_input, '%Y-%m-%d').strftime('%A %d %B %Y')}."

        st.write(replication_string)
        quarto_string += replication_string.replace("will run", "ran")
        quarto_string += "\n\n"

        quarto_string += f"Activity durations are modified by a factor of {st.session_state.activity_duration_multiplier}\n\n"

        if st.session_state.create_animation_input:
            st.write("An animated output will be created.")
            st.info("Turn off this option if the model is running very slowly!")
        else:
            st.write("No animated output will be created.")

        if st.session_state.amb_data:
            st.write("SWAST Ambulance Activity will be modelled.")
        else:
            st.write("SWAST Ambulance Activity will not be modelled.")


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
    progress_text = "Simulation in progress. Please wait."
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
                        activity_duration_multiplier=float(st.session_state.activity_duration_multiplier)
                    )

                results.append(
                    run_results
                    )

                my_bar.progress((run+1)/st.session_state.number_of_runs_input, text=progress_text)

            # Turn into a single dataframe when all runs complete
            results_all_runs = pd.concat(results)

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
                        activity_duration_multiplier=float(st.session_state.activity_duration_multiplier)
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

        my_bar.empty()

        if st.session_state.create_animation_input:
            tab_names.append("Animation")
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                tab_names
            )
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                tab_names
            )

        def get_job_count_df():
            return _job_count_calculation.make_job_count_df(params_path="data/run_params_used.csv",
                                                            path="data/run_results.csv")

        def get_params_df():
            return pd.read_csv("data/run_params_used.csv")

        with tab1:
            quarto_string += "# Key Metrics\n\n"

            averaged_string = f"All Metrics are averaged across {st.session_state.number_of_runs_input} simulation runs"

            quarto_string += averaged_string
            quarto_string += "\n\n"

            st.info(averaged_string)

            report_message = st.empty()

            historical_utilisation_df_complete, historical_utilisation_df_summary = (
                _utilisation_result_calculation.make_RWC_utilisation_dataframe(
                    historical_df_path="historical_data/historical_monthly_resource_utilisation.csv",
                    rota_path="actual_data/HEMS_ROTA.csv",
                    service_path="data/service_dates.csv"
                    )
                )

            print(historical_utilisation_df_summary)

            t1_col1, t1_col2 = st.columns(2)

            with t1_col1:
                perc_unattended = _vehicle_calculation.get_perc_unattended_string(results_all_runs)

                quarto_string += "## Calls Not Attended\n\n"

                quarto_string += f"Across these runs of the simulation, on average a DAAT Resource was unable to attend {perc_unattended} calls\n\n"

                with iconMetricContainer(key="nonattend_metric", icon_unicode="e61f", family="outline"):
                    st.metric("Average Number of Calls DAAT Resource Couldn't Attend",
                            perc_unattended,
                            border=True)

                    st.warning("""
Data on the number of calls that were historically not attended due to resource unavailability is being added soon - a comparison with the model's figures
will be available at this point
                            """)

                    missed_calls_description = get_text("missed_calls_description", text_df)

                    st.caption(missed_calls_description)

                    quarto_string += missed_calls_description

            with t1_col2:
                quarto_string += "\n\n## Resource Utilisation"
                resource_use_wide, utilisation_df_overall, utilisation_df_per_run, utilisation_df_per_run_by_csg = _utilisation_result_calculation.make_utilisation_model_dataframe(
                    path="data/run_results.csv",
                    params_path="data/run_params_used.csv",
                    service_path="data/service_dates.csv",
                    rota_path="data/hems_rota_used.csv"
                )

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

            cars = ["C70", "C71", "C72"]

            car_metric_cols = st.columns(len(cars))

            historical_utilisation_df_summary.index = historical_utilisation_df_summary.index.str.replace("CC", "C")

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

#             with t1_col3:
#                 with iconMetricContainer(key="preferred_response_metric", icon_unicode="e838", family="outline"):
#                     st.metric("Preferred Resource Allocated",
#                             "907 of 1203 (75.4%)",
#                             border=True)
#                     st.caption("""
# This is the percentage of time where the 'preferred' resource was available at the time of the call
# for response.
# """)


        with tab2:
            tab_2_1, tab_2_2 = st.tabs(["Resource Utilisation", "'Missed' Calls"])
            with tab_2_1:
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

                with tab_2_2:

                    st.warning("""
Data on the number of calls that were historically not attended due to resource unavailability is being added soon - a comparison with the model's figures
will be available at this point
                            """)


        with tab3:

            # tab_3_1, tab_3_2, tab_3_3, tab_3_4, tab_3_5 = st.tabs([
            tab_3_1, tab_3_2, tab_3_3, tab_3_4 = st.tabs([
                "Jobs per Month",
                "Jobs per Hour",
                "Jobs per Day",
                "Job Durations - Overall",
                # "Job Durations - Split"
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
                            historical_monthly_job_data_path="historical_data/historical_jobs_per_month.csv"
                            )

                    _job_count_calculation.plot_monthly_calls(
                            call_df,
                            show_individual_runs=show_individual_runs,
                            use_poppins=False,
                            show_historical=show_real_data,
                            show_historical_individual_years=show_historical_individual_years,
                            historical_monthly_job_data_path="historical_data/historical_jobs_per_month.csv"
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

            with tab_3_4:
                historical_time_df = _job_time_calcs.get_historical_times(
                            'historical_data/historical_median_time_of_activities_by_month_and_resource_type.csv'
                            )

                fig_job_durations_historical =  _job_time_calcs.plot_historical_utilisation_vs_simulation_overall(
                        historical_time_df,
                        utilisation_model_df=_job_time_calcs.get_total_times_model(
                            get_summary=False,
                            path="data/run_results.csv",
                            params_path="data/run_params_used.csv",
                            rota_path="data/hems_rota_used.csv",
                            service_path="data/service_dates.csv"),
                        use_poppins=True
                        )

                _job_time_calcs.plot_historical_utilisation_vs_simulation_overall(
                        historical_time_df,
                        utilisation_model_df=_job_time_calcs.get_total_times_model(
                            get_summary=False,
                            path="data/run_results.csv",
                            params_path="data/run_params_used.csv",
                            rota_path="data/hems_rota_used.csv",
                            service_path="data/service_dates.csv"),
                        use_poppins=False
                        ).write_html("app/fig_outputs/fig_job_durations_historical.html",full_html=False, include_plotlyjs='cdn')#, post_script = poppins_script)

                st.plotly_chart(
                   fig_job_durations_historical
                    )

                st.caption("""
This plot looks at the total amount of time each resource was in use during the simulation.

All simulated points are represented in the box plots.

The blue bars give an indication of the historical averages

We would expect the median - the central horizontal line within the box portion of the box plots -
to fall within the blue box for each resource type, and likely to be fairly central within that
region.
""")

            # with tab_3_5:
            #     event_log_df = _job_time_calcs.create_simulation_event_duration_df(
            #                 event_log_path="data/run_results.csv"
            #                 )



            #     st.plotly_chart(
            #         _job_time_calcs.plot_activity_time_breakdowns(
            #             historical_activity_times=historical_time_df,
            #             event_log_df=event_log_df,
            #             title="Simulation Event Times Breakdown - Helicopter",
            #             vehicle_type="helicopter"
            #         )
            #     )

            #     st.plotly_chart(
            #         _job_time_calcs.plot_activity_time_breakdowns(
            #             historical_activity_times=historical_time_df,
            #             event_log_df=event_log_df,
            #             title="Simulation Event Times Breakdown - Car",
            #             vehicle_type="car"
            #         )
            #     )

        with tab4:

            st.caption("""
This tab contains visualisations to help model authors do additional checks into the underlying functioning of the model.

Most users will not need to look at the visualisations in this tab.
            """)

            tab_4_1, tab_4_2 = st.tabs(["Debug Resources", "Debug Events"])

            with tab_4_2:
                st.subheader("Event Overview")

                @st.fragment
                def event_overview_plot():
                    runs_to_display_eo = st.multiselect("Choose the runs to display", results_all_runs["run_number"].unique(), default=1)

                    events_over_time_df = results_all_runs[results_all_runs["run_number"].isin(runs_to_display_eo)]

                    events_over_time_df['time_type'] = events_over_time_df['time_type'].astype('str')

                    fig = px.scatter(
                            events_over_time_df,
                            x="timestamp_dt",
                            y="time_type",
                            # facet_row="run_number",
                            # showlegend=False,
                            color="time_type",
                            height=800,
                            title="Events Over Time - By Run"
                            )

                    fig.update_traces(marker=dict(size=3, opacity=0.5))

                    fig.update_layout(yaxis_title="", # Remove y-axis label
                                      yaxis_type='category',
                                      showlegend=False)
                    # Remove facet labels
                    fig.for_each_annotation(lambda x: x.update(text=""))

                    # fig.update_xaxes(rangeslider_visible=True,
                    # rangeselector=dict(
                    #     buttons=list([
                    #         dict(count=1, label="1m", step="month", stepmode="backward"),
                    #         dict(count=6, label="6m", step="month", stepmode="backward"),
                    #         dict(count=1, label="YTD", step="year", stepmode="todate"),
                    #         dict(count=1, label="1y", step="year", stepmode="backward"),
                    #         dict(step="all")
                    #     ]))
                    # )

                    st.plotly_chart(
                        fig,
                            use_container_width=True
                        )

                event_overview_plot()

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

                # st.plotly_chart(
                #         px.bar(
                #             event_counts_long[event_counts_long["time_type"].isin(["arrival", "AMB call start", "HEMS call start"])],
                #             x="run_number",
                #             y="value",
                #             facet_col="time_type",
                #             height=600
                #     )
                # )

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

                # amb_events = ["arrival", "AMB call start", "AMB clear"]

                # st.plotly_chart(
                #         px.funnel(
                #             event_counts_long[event_counts_long["time_type"].isin(amb_events)],
                #             facet_col="run_number",
                #             x="value",
                #             y="time_type",
                #             category_orders={"time_type": amb_events[::-1]},

                #     )
                # )

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

            with tab_4_1:
                st.subheader("Resource Use")

                resource_use_events_only = results_all_runs[results_all_runs["event_type"].str.contains("resource_use")]

                @st.fragment
                def resource_use_exploration_plots():

                    run_select_ruep = st.selectbox("Choose the run to show",
                                resource_use_events_only["run_number"].unique())


                    with st.expander("Click here to see the timings of resource use"):
                        st.dataframe(resource_use_events_only[resource_use_events_only["run_number"] == run_select_ruep])

                        st.dataframe(resource_use_events_only[resource_use_events_only["run_number"] == run_select_ruep]
                                        [["P_ID", "time_type", "timestamp_dt", "event_type"]]
                                        .melt(id_vars=["P_ID", "time_type", "event_type"],
                                        value_vars="timestamp_dt"))

                        resource_use_wide = (resource_use_events_only[resource_use_events_only["run_number"] == run_select_ruep]
                            [["P_ID", "time_type", "timestamp_dt", "event_type"]]
                            .pivot(columns="event_type", index=["P_ID","time_type"], values="timestamp_dt").reset_index())

                        # get the number of referrals and assign them a value
                        resources = resource_use_wide.time_type.unique()
                        resource_dict = {resource: index for index, resource in enumerate(resources)}
                        resource_use_wide["y_pos"] = resource_use_wide["time_type"].map(resource_dict)

                        # Convert time types to numerical values
                        # resource_use_wide["y_pos"] = resource_use_wide["time_type"].map(resource_dict)

                        # Compute duration for each resource use
                        resource_use_wide["resource_use_end"] = pd.to_datetime(resource_use_wide["resource_use_end"])
                        resource_use_wide["resource_use"] = pd.to_datetime(resource_use_wide["resource_use"])
                        # resource_use_wide["duration"] = resource_use_wide["resource_use_end"] - resource_use_wide["resource_use"]

                        # Convert time to numeric (e.g., seconds since the first event)
                        time_origin = resource_use_wide["resource_use"].min()  # Set the reference time
                        # resource_use_wide["resource_use_numeric"] = (resource_use_wide["resource_use"] - time_origin).dt.total_seconds()
                        # resource_use_wide["resource_use_end_numeric"] = (resource_use_wide["resource_use_end"] - time_origin).dt.total_seconds()

                        # Compute duration
                        # resource_use_wide["duration"] = resource_use_wide["resource_use_end_numeric"] - resource_use_wide["resource_use_numeric"]
                        resource_use_wide["duration"] = resource_use_wide["resource_use_end"] - resource_use_wide["resource_use"]

                        # Compute duration as seconds and multiply by 1000 (to account for how datetime axis
                        # is handled in plotly
                        resource_use_wide["duration_seconds"] = (resource_use_wide["resource_use_end"] - resource_use_wide["resource_use"]).dt.total_seconds()*1000
                        resource_use_wide["duration_minutes"] = resource_use_wide["duration_seconds"] / 1000 / 60

                        resource_use_wide["callsign_group"] = resource_use_wide["time_type"].str.extract("(\d+)")

                        resource_use_wide = resource_use_wide.sort_values(["callsign_group", "time_type"])

                        st.dataframe(resource_use_wide)

                        ######################################
                        # Load in the servicing schedule df
                        ######################################
                        service_schedule = pd.read_csv("data/service_dates.csv")
                        # Convert to appropriate datatypes
                        service_schedule["service_end_date"] = pd.to_datetime(service_schedule["service_end_date"])
                        service_schedule["service_start_date"] = pd.to_datetime(service_schedule["service_start_date"])
                        # Calculate duration for plotting and for hover text
                        service_schedule["duration_seconds"] = (service_schedule["service_end_date"] - service_schedule["service_start_date"]).dt.total_seconds()*1000
                        service_schedule["duration_days"] = (service_schedule["duration_seconds"] / 1000) / 60 / 60 / 24
                        # Match y position of the resource in the rest of the plot
                        service_schedule["y_pos"] = service_schedule["resource"].map(resource_dict)
                        # Limit the dataframe to only contain servicing
                        service_schedule = service_schedule[(service_schedule["service_start_date"] <= resource_use_wide.resource_use.max()) &
                                        (service_schedule["service_end_date"] >= resource_use_wide.resource_use.min())]

                        st.dataframe(service_schedule)

                    # Create figure
                    resource_use_fig = go.Figure()

                    # Add horizontal bars using actual datetime values
                    for idx, callsign in enumerate(resource_use_wide.time_type.unique()):
                        callsign_df = resource_use_wide[resource_use_wide["time_type"]==callsign]

                        service_schedule_df = service_schedule[service_schedule["resource"]==callsign]

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
                                customdata=service_schedule_df[['resource','duration_days','service_start_date', 'service_end_date']],
                                hovertemplate="Servicing %{customdata[0]} lasting %{customdata[1]} days (%{customdata[2]|%a %-e %b %Y} to %{customdata[3]|%a %-e %b %Y})<extra></extra>"
                            ))

                        # Add in boxes showing the duration of individual calls
                        resource_use_fig.add_trace(go.Bar(
                            x=callsign_df["duration_seconds"],  # Duration (Timedelta)
                            y=callsign_df["y_pos"],
                            base=callsign_df["resource_use"],  # Start time as actual datetime
                            orientation="h",
                            width=0.4,
                            marker=dict(color=list(DAA_COLORSCHEME.values())[idx],
                                        line=dict(color=list(DAA_COLORSCHEME.values())[idx], width=1)
                                        ),
                            name=callsign,
                            # customdata=np.stack((callsign_df['resource_use'], callsign_df['resource_use_end']), axis=-1),
                            customdata=callsign_df[['resource_use','resource_use_end','time_type', 'duration_minutes']],
                            hovertemplate="Response from %{customdata[2]} lasting %{customdata[3]} minutes (%{customdata[0]|%a %-e %b %Y %H:%M} to %{customdata[1]|%a %-e %b %Y %H:%M})<extra></extra>"
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
                    # rangeselector=dict(
                    #     buttons=list([
                    #         dict(count=1, label="1m", step="month", stepmode="backward"),
                    #         dict(count=6, label="6m", step="month", stepmode="backward"),
                    #         dict(count=1, label="YTD", step="year", stepmode="todate"),
                    #         dict(count=1, label="1y", step="year", stepmode="backward"),
                    #         dict(step="all")
                    #     ]))
                    )

                    resource_use_fig.write_html("app/fig_outputs/resource_use_fig.html",full_html=False, include_plotlyjs='cdn')#, post_script = poppins_script)

                    st.plotly_chart(
                        resource_use_fig
                    )

                resource_use_exploration_plots()

        with tab5:
            report_message = st.empty()

            report_message.info("Generating Report...")

            try:
                with open("app/fig_outputs/quarto_text.txt", "w") as text_file:
                    text_file.write(quarto_string)

                _app_utils.generate_quarto_report(run_quarto_check=False)

                report_message.success("Report Available for Download")

            except:
                ## error message
                report_message.error(f"Report cannot be generated - please speak to a devleoper")
