import streamlit as st
import platform

# Data processing imports
import pandas as pd
from datetime import datetime

# Memory management
import gc

from _app_utils import (
    get_text,
    get_text_sheet,
    summary_sidebar,
    generate_quarto_report,
)

from air_ambulance_des.utils import iconMetricContainer, COLORSCHEME, format_diff

# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Simulation imports
from air_ambulance_des.des_parallel_process import (
    runSim,
    parallelProcessJoblib,
    collateRunResults,
)
from _state_control import setup_state

from air_ambulance_des.utils import Utils

from streamlit_extras.stylable_container import stylable_container

from air_ambulance_des.class_simulation_trial_results import TrialResults
from air_ambulance_des.class_simulation_inputs import SimulationInputs
from air_ambulance_des.class_historic_results import HistoricResults

APP_DIR = Path(__file__).parent.resolve()

st.set_page_config(layout="wide")

with open(f"{APP_DIR}/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


setup_state()

# st.write("Current working directory:", os.getcwd())
# st.write("Contents:", os.listdir())

# Pull in required font
poppins_script = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
"""

quarto_string = ""

text_df = get_text_sheet("model")

# Avoid reading from utils due to odd issues it seems to be introducing
# u = Utils()
# rota = u.HEMS_ROTA
# rota = pd.read_csv("actual_data/HEMS_ROTA.csv")
# SERVICING_SCHEDULE = pd.read_csv('actual_data/service_schedules_by_model.csv')

# Set up the columns for the 'run simulation' button and DAA logo
col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("Run a Simulation")

with col2:
    st.image(f"{APP_DIR}/assets/daa-logo.svg", width=200)

# Add a sidebar with a toggle for debugging messages and a summary of the model input parameters
with st.sidebar:
    debug_messages = st.toggle(
        "Turn on debugging messages",
        value=st.session_state.debugging_messages_to_log,
        key="key_debugging_messages_to_log",
        on_change=lambda: setattr(
            st.session_state,
            "debugging_messages_to_log",
            st.session_state.key_debugging_messages_to_log,
        ),
        help="This will turn on display of messages in the developer terminal and write logging messages to the log.txt file",
    )

    summary_sidebar(quarto_string=quarto_string)

# Create the run button with a custom style
with stylable_container(
    key="run_buttons",
    css_styles=f"""
                    button {{
                            background-color: {COLORSCHEME["blue"]};
                            color: white;
                            border-color: white;
                        }}
                        """,
):
    button_run_pressed = st.button(
        "Click this button to run the simulation with the selected parameters",
        icon=":material/play_circle:",
    )

# If the user hasn't been to the setup page, warn them that they will be using the default
# parameters, and give them an easy way to navigate to the parameter setup page
if not st.session_state["visited_setup_page"]:
    if not button_run_pressed:
        with stylable_container(
            key="warning_buttons",
            css_styles=f"""
                    button {{
                            background-color: {COLORSCHEME["orange"]};
                            color: {COLORSCHEME["charcoal"]};
                            border-color: white;
                        }}
                        """,
        ):
            if st.button(
                "**Warning**\n\nYou haven't set up any parameters - default parameters will be used!\n\nClick this button to go to the parameter page, or click the blue button above\n\nif you are happy to use the default model parameters",
                icon=":material/warning:",
            ):
                st.switch_page("setup.py")

if button_run_pressed:
    # Clear cache when new run is started
    st.cache_data.clear()

    gc.collect()

    # MARK: Run Simulation
    progress_text = "Simulation in progress. Please wait."
    # This check is a way to guess whether it's running on
    # Streamlit community cloud
    if platform.processor() == "":
        my_bar = st.progress(0, text=progress_text)

    with st.spinner(
        f"Simulating {st.session_state.number_of_runs_input} replication(s) of {st.session_state.sim_duration_input} days. This may take several minutes...",
        show_time=True,
    ):
        # If running on community cloud, parallelisation will not work
        # so run instead using the runSim function sequentially
        if platform.processor() == "":
            print("Running sequentially")
            results = []

            for run in range(st.session_state.number_of_runs_input):
                run_results = runSim(
                    run=run,
                    total_runs=st.session_state.number_of_runs_input,
                    sim_duration=float(st.session_state.sim_duration_input * 24 * 60),
                    warm_up_time=float(st.session_state.warm_up_duration * 60),
                    sim_start_date=datetime.combine(
                        datetime.strptime(
                            st.session_state.sim_start_date_input, "%Y-%m-%d"
                        ).date(),
                        datetime.strptime(
                            st.session_state.sim_start_time_input, "%H:%M"
                        ).time(),
                    ),
                    amb_data=st.session_state.amb_data,
                    demand_increase_percent=float(st.session_state.overall_demand_mult)
                    / 100.0,
                    activity_duration_multiplier=float(
                        st.session_state.activity_duration_multiplier
                    ),
                    print_debug_messages=debug_messages,
                    random_seed=st.session_state.master_seed + run,
                )

                results.append(run_results)

                my_bar.progress(
                    (run + 1) / st.session_state.number_of_runs_input,
                    text=progress_text,
                )

            # Turn into a single dataframe when all runs complete
            pd.concat(results).to_csv(
                Utils.RUN_RESULTS_CSV, index=True, encoding="utf-8-sig"
            )

            my_bar.empty()

        # If running locally, use parallel processing function to speed up execution significantly
        else:
            print("Running in parallel")
            print(
                f"st.session_state.overall_demand_mult: {st.session_state.overall_demand_mult}"
            )
            print(
                f"st.session_state.sim_duration_input: {st.session_state.sim_duration_input}"
            )
            parallelProcessJoblib(
                total_runs=st.session_state.number_of_runs_input,
                sim_duration=float(st.session_state.sim_duration_input * 24 * 60),
                warm_up_time=float(st.session_state.warm_up_duration * 60),
                sim_start_date=datetime.combine(
                    datetime.strptime(
                        st.session_state.sim_start_date_input, "%Y-%m-%d"
                    ).date(),
                    datetime.strptime(
                        st.session_state.sim_start_time_input, "%H:%M"
                    ).time(),
                ),
                amb_data=st.session_state.amb_data,
                demand_increase_percent=float(st.session_state.overall_demand_mult)
                / 100.0,
                activity_duration_multiplier=float(
                    st.session_state.activity_duration_multiplier
                ),
                print_debug_messages=debug_messages,
                master_seed=st.session_state.master_seed,
            )
            collateRunResults()

        tab_names = [
            "Simulation Results Summary",
            "Key Visualisations",
            "Comparing Model with Historic Data",
            "Additional Outputs",
            "Download Output",
        ]

        # Set up the required number of tabs for results, which will vary depending on whether
        # the user has chosen to create an animation
        if st.session_state.create_animation_input:
            tab_names.append("Animation")
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)

        with tab5:
            report_message = st.empty()

        try:
            simulation_inputs = SimulationInputs(
                data_folder_path="data", actual_data_folder_path="actual_data"
            )
            print("simulation input object created successfully")

        except FileNotFoundError:
            print("data folder or actual_data folder not found")

        try:
            historical_data = HistoricResults(
                historical_data_folder_path="historical_data",
                historic_rota_df_path="tests/rotas_historic/HISTORIC_HEMS_ROTA.csv",
                historic_callsign_df_path="tests/rotas_historic/HISTORIC_callsign_registration_lookup.csv",
                historic_servicing_df_path="tests/rotas_historic/HISTORIC_service_dates.csv",
            )
            print("historical data object created successfully")

        except FileNotFoundError:
            print("historical data folder or historic rotas not found")

        try:
            trial_results = TrialResults(
                simulation_inputs=simulation_inputs,
                historical_data=historical_data,
                run_results=pd.read_csv("data/run_results.csv"),
                daily_availability_data=pd.read_csv("data/daily_availability.csv"),
            )
            print("trial results object created successfully")
        except FileNotFoundError:
            print("run results not found")

        # MARK: Result Summary
        with tab1:
            quarto_string += "# Key Metrics\n\n"

            averaged_string = f"All Metrics are averaged across {st.session_state.number_of_runs_input} simulation runs"

            quarto_string += "*"
            quarto_string += averaged_string
            quarto_string += "*\n\n"

            st.info(averaged_string)

            report_message = st.empty()

            st.subheader("Missed Jobs")

            t1_col1, t1_col2 = st.columns(2)

            with t1_col1:
                (
                    total_average_calls_received_per_year,
                    perc_unattended,
                    perc_unattended_normalised,
                ) = trial_results.get_perc_unattended_string_normalised()

                quarto_string += "## Calls Not Attended\n\n"

                quarto_string += f"Across these runs of the simulation, on average a DAAT Resource was unable to attend {perc_unattended} calls\n\n"

                with iconMetricContainer(
                    key="nonattend_metric", icon_unicode="e61f", family="outline"
                ):
                    st.metric(
                        "Average Number of Calls DAAT Resource Couldn't Attend",
                        perc_unattended,
                        border=True,
                    )
                    missed_calls_hist_string = (
                        historical_data.PLOT_historical_missed_jobs_data(
                            format="string"
                        )
                    )
                    st.caption(f"**{perc_unattended_normalised}**")
                    st.caption(
                        f"*This compares to an average of {missed_calls_hist_string:.1f}% of calls missed historically (approximately {total_average_calls_received_per_year * (float(missed_calls_hist_string) / 100):.0f} calls per year)*"
                    )

                    missed_calls_description = get_text(
                        "missed_calls_description", text_df
                    )

            with t1_col2:
                st.caption(missed_calls_description)

                quarto_string += missed_calls_description

                # with st.expander("View Breakdown"):
                #     outcome_df = _vehicle_calculation.resource_allocation_outcomes(results_all_runs)
                #     outcome_df["Count"] = (outcome_df["Count"]/st.session_state.number_of_runs_input).round(0)
                #     outcome_df.rename(columns={'Count':'Mean Calls per Simulation Run'}, inplace=True)
                #     st.dataframe(outcome_df)

            st.divider()

            st.markdown("## Critical Care, Enhanced Care and Helicopter Benefit")

            st.markdown("### Missed Jobs")

            col_ec_cc_sim, col_ec_cc_hist_sim = st.columns(2)

            with col_ec_cc_sim:
                st.markdown("#### Simulation Outputs")

                sim_missed_cc = trial_results.RETURN_missed_jobs_fig("CC")
                sim_missed_ec = trial_results.RETURN_missed_jobs_fig("EC")
                sim_missed_all_reg = trial_results.RETURN_missed_jobs_fig(
                    "REG"
                ) + trial_results.RETURN_missed_jobs_fig("REG - Helicopter Benefit")
                sim_missed_reg_heli_benefit = trial_results.RETURN_missed_jobs_fig(
                    "REG - Helicopter Benefit"
                )

                # ----------------------------------------------------------------------- #

                # ======= Calculate missed jobs under historical rotas/conditions ======= #
                hist_missed_cc = historical_data.RETURN_missed_jobs_fig("CC")
                hist_missed_ec = historical_data.RETURN_missed_jobs_fig("EC")
                hist_missed_all_reg = historical_data.RETURN_missed_jobs_fig(
                    "REG"
                ) + historical_data.RETURN_missed_jobs_fig("REG - Helicopter Benefit")
                hist_missed_reg_heli_benefit = historical_data.RETURN_missed_jobs_fig(
                    "REG - Helicopter Benefit"
                )
                # ======================================================================= #

                # '''''''''''' Calculate the difference '''''''''''' #
                diff_missed_cc = sim_missed_cc - hist_missed_cc
                diff_missed_ec = sim_missed_ec - hist_missed_ec
                diff_missed_all_reg = sim_missed_all_reg - hist_missed_all_reg
                diff_missed_reg_heli_benefit = (
                    sim_missed_reg_heli_benefit - hist_missed_reg_heli_benefit
                )

                """""" """""" """""" """""" """""" """""" """""" """""" ""  #

                missed_jobs_sim_string = f"""
    The simulation estimates that, with the proposed conditions, there would be - on average, per year - roughly

    - **{sim_missed_cc:.0f} critical care** jobs that would be missed due to no resource being available  (*{format_diff(diff_missed_cc)}*), with an estimated range of {historical_data.RETURN_missed_jobs_fig("CC", "min"):.0f} to {historical_data.RETURN_missed_jobs_fig("CC", "max"):.0f}

    - **{sim_missed_ec:.0f} enhanced care** jobs that would be missed due to no resource being available (*{format_diff(diff_missed_ec)}*) with an estimated range of {historical_data.RETURN_missed_jobs_fig("EC", "min"):.0f} to {historical_data.RETURN_missed_jobs_fig("EC", "max"):.0f}

    - **{sim_missed_all_reg:.0f} jobs with no predicted CC or EC intervention** that would be missed due to no resource being available (*{format_diff(diff_missed_all_reg)}*) with an estimated range of {historical_data.RETURN_missed_jobs_fig("REG", "min") + historical_data.RETURN_missed_jobs_fig("REG - Helicopter Benefit", "min"):.0f} to {historical_data.RETURN_missed_jobs_fig("REG", "max") + historical_data.RETURN_missed_jobs_fig("REG - Helicopter Benefit", "max"):.0f}

        - of these missed regular jobs, **{sim_missed_reg_heli_benefit:.0f}** may have benefitted from the attendance of a helicopter (*{format_diff(diff_missed_reg_heli_benefit)}*)
                            """

                st.write(missed_jobs_sim_string)

                quarto_string += "## Missed Jobs\n\n"
                quarto_string += missed_jobs_sim_string

            with col_ec_cc_hist_sim:
                missed_jobs_historical_comparison = f"""
    As CC, EC and helicopter benefit can only be determined for attended jobs, we cannot estimate the ratio for previously missed jobs.
    However, the simulation estimates that, with historical rotas and vehicles, there would be - on average, per year - roughly

    - {hist_missed_cc:.0f} critical care jobs that would be missed due to no resource being available *(estimated range of {historical_data.RETURN_missed_jobs_fig("CC", "min"):.0f} to {historical_data.RETURN_missed_jobs_fig("CC", "max"):.0f})*
    - {hist_missed_ec:.0f} enhanced care jobs that would be missed due to no resource being available *(estimated range of {historical_data.RETURN_missed_jobs_fig("EC", "min"):.0f} to {historical_data.RETURN_missed_jobs_fig("EC", "max"):.0f})*
    - {hist_missed_all_reg:.0f} jobs with no predicted CC or EC intervention that would be missed due to no resource being available *(estimated range of {historical_data.RETURN_missed_jobs_fig("REG", "min") + historical_data.RETURN_missed_jobs_fig("REG - Helicopter Benefit", "min"):.0f} to {historical_data.RETURN_missed_jobs_fig("REG", "max") + historical_data.RETURN_missed_jobs_fig("REG - Helicopter Benefit", "max"):.0f})*
        - of these missed regular jobs, {hist_missed_reg_heli_benefit:.0f} may have benefitted from the attendance of a helicopter *(estimated range of {historical_data.RETURN_missed_jobs_fig("REG - Helicopter Benefit", "min"):.0f} to {historical_data.RETURN_missed_jobs_fig("REG - Helicopter Benefit", "max"):.0f})*
                            """

                st.caption(missed_jobs_historical_comparison)

                quarto_string += "### Historical Missed Jobs\n\n"
                quarto_string += missed_jobs_historical_comparison

            st.markdown("### Suboptimal Resource Allocation")

            col_ec_cc_suboptimal_sim, col_ec_cc_suboptimal_hist_sim = st.columns(2)

            with col_ec_cc_suboptimal_sim:
                mean_cc_sent_ec, min_cc_sent_ec, max_cc_sent_ec = (
                    trial_results.RETURN_prediction_cc_patients_sent_ec_resource()
                )

                mean_heli_ben_sent_car, min_heli_ben_sent_car, max_heli_ben_sent_car = (
                    trial_results.RETURN_prediction_heli_benefit_patients_sent_car()
                )

                mean_cc_sent_ec_HIST, min_cc_sent_ec_HIST, max_cc_sent_ec_HIST = (
                    historical_data.RETURN_prediction_cc_patients_sent_ec_resource()
                )

                (
                    mean_heli_ben_sent_car_HIST,
                    min_heli_ben_sent_car_HIST,
                    max_heli_ben_sent_car_HIST,
                ) = historical_data.RETURN_prediction_heli_benefit_patients_sent_car()

                suboptimal_jobs_sim_string = f"""
                The simulation estimates that, with the proposed conditions, there would be - on average, per year - roughly

                - **{mean_cc_sent_ec:.0f} critical care (CC)** jobs that would be sent an enhanced care (EC) resource (*{format_diff(mean_cc_sent_ec - mean_cc_sent_ec_HIST)}*), with an estimated range of {min_cc_sent_ec:.0f} to {max_cc_sent_ec:.0f}

                - **{mean_heli_ben_sent_car:.0f} jobs that would benefit from a helicopter** that would be sent a car (*{format_diff(mean_heli_ben_sent_car - mean_heli_ben_sent_car_HIST)}*) with an estimated range of {min_heli_ben_sent_car:.0f} to {max_heli_ben_sent_car:.0f}
                """

                st.write(suboptimal_jobs_sim_string)

                quarto_string += "## Suboptimal Resource Allocation to Jobs\n\n"
                quarto_string += suboptimal_jobs_sim_string

            with col_ec_cc_suboptimal_hist_sim:
                suboptimal_jobs_hist_string = f"""
                As CC, EC and helicopter benefit can only be determined for attended jobs, we cannot estimate the ratio for previously missed jobs.
                However, the simulation estimates that, with historical rotas and vehicles, there would be - on average, per year - roughly

                - **{mean_cc_sent_ec_HIST:.0f} critical care (CC)** jobs that would be sent an enhanced care (EC) resource, with an estimated range of {min_cc_sent_ec_HIST:.0f} to {max_cc_sent_ec_HIST:.0f}

                - **{mean_heli_ben_sent_car_HIST:.0f} jobs that would benefit from a helicopter** that would be sent a car, with an estimated range of {min_heli_ben_sent_car_HIST:.0f} to {max_heli_ben_sent_car_HIST:.0f}
                """
                st.write(suboptimal_jobs_hist_string)

                quarto_string += "## Suboptimal Resource Allocation to Jobs - Historical Comparison\n\n"
                quarto_string += suboptimal_jobs_hist_string

            st.subheader("Resource Utilisation")

            quarto_string += "\n\n## Resource Utilisation"

            # Get unique callsigns for helicopters and cars from run_results
            if (
                "vehicle_type" in trial_results.run_results.columns
                and "callsign" in trial_results.run_results.columns
            ):
                all_helicopter_callsigns = sorted(
                    list(
                        trial_results.run_results[
                            trial_results.run_results["vehicle_type"] == "helicopter"
                        ]["callsign"]
                        .dropna()
                        .unique()
                    )
                )
                all_car_callsigns = sorted(
                    list(
                        trial_results.run_results[
                            trial_results.run_results["vehicle_type"] == "car"
                        ]["callsign"]
                        .dropna()
                        .unique()
                    )
                )
            else:
                st.error(
                    "The 'run_results' DataFrame is missing 'vehicle_type' or 'callsign' columns."
                )
                all_helicopter_callsigns = []
                all_car_callsigns = []

            # --- Display Helicopter Metrics ---
            st.markdown("### Helicopters")
            if all_helicopter_callsigns:
                helo_cols = st.columns(len(all_helicopter_callsigns))
                for idx, helo_callsign in enumerate(all_helicopter_callsigns):
                    quarto_string = trial_results.display_vehicle_utilisation_metric(
                        st_column=helo_cols[idx],
                        callsign_to_display=helo_callsign,
                        vehicle_type_label="Helicopter",
                        icon_unicode="f60c",
                        historical_data_class=historical_data,
                        current_quarto_string=quarto_string,
                    )
            else:
                st.info("No helicopter data found in the current run results.")

            st.caption(get_text("helicopter_utilisation_description", text_df))
            st.divider()

            # --- Display Car Metrics ---
            st.markdown("### Cars")
            if all_car_callsigns:
                car_metric_cols = st.columns(len(all_car_callsigns))
                for idx, car_callsign in enumerate(all_car_callsigns):
                    quarto_string = trial_results.display_vehicle_utilisation_metric(
                        historical_data_class=historical_data,
                        st_column=car_metric_cols[idx],
                        callsign_to_display=car_callsign,
                        vehicle_type_label="Car",
                        icon_unicode="eb3c",
                        current_quarto_string=quarto_string,
                    )
            else:
                st.info("No car data found in the current run results.")

            # Display a description for car utilisation
            # st.caption(get_text("car_utilisation_description", text_df))
            # st.divider() # If you add a caption above, a divider might be good here too.

            t1_col3, t1_col4 = st.columns(2)

        # MARK: Key Plots
        with tab2:
            tab_2_1, tab_2_2, tab_2_3, tab_2_4, tab_2_5 = st.tabs(
                [
                    "'Missed' Calls",
                    "Resource Utilisation",
                    "Split of Jobs by Callsign Group",
                    "CC and EC Benefit",
                    "Resource Tasking",
                ]
            )

            with tab_2_1:

                @st.fragment
                def missed_jobs():
                    show_proportions_per_hour = st.toggle(
                        "Show as proportion of jobs missed per hour", value=False
                    )

                    by_quarter = st.toggle("Stratify results by quarter", value=False)

                    st.plotly_chart(
                        trial_results.PLOT_missed_jobs(
                            show_proportions_per_hour=show_proportions_per_hour,
                            by_quarter=by_quarter,
                        )
                    )

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

            # NOTE!
            # The final plot in this tab (summary of missed calls over runs) is not created
            # # until tab_2_4, when a related plot is created.
            # It then gets put here.

            with tab_2_2:

                @st.fragment
                def create_utilisation_rwc_plot():
                    fig_utilisation = trial_results.PLOT_UTIL_rwc_plot()

                    fig_utilisation.write_html(
                        f"{APP_DIR}/fig_outputs/fig_utilisation.html",
                        full_html=False,
                        include_plotlyjs="cdn",
                    )  # , post_script = poppins_script)#,full_html=False, include_plotlyjs='cdn')

                    st.plotly_chart(fig_utilisation)

                create_utilisation_rwc_plot()

                historical_monthly_totals_df = pd.read_csv(
                    "historical_data/historical_monthly_totals_by_callsign.csv"
                )
                historical_monthly_totals_df["month"] = pd.to_datetime(
                    historical_monthly_totals_df["month"], format="%Y-%m-%d"
                )

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

                st.plotly_chart(
                    trial_results.PLOT_SIMULATION_utilisation_summary(
                        historical_results_obj=historical_data
                    )
                )

                st.plotly_chart(
                    trial_results.make_SIMULATION_utilisation_variation_plot(
                        historical_results_obj=historical_data
                    )
                )

                with tab_2_3:

                    @st.fragment
                    def plot_callsign_group_split():
                        x_is_callsign_group = st.toggle(
                            "Plot callsign group on the horizontal axis", value=False
                        )

                        st.plotly_chart(
                            trial_results.create_callsign_group_split_rwc_plot(
                                historical_data_obj=historical_data,
                                x_is_callsign_group=x_is_callsign_group,
                            )
                        )

                    plot_callsign_group_split()

                with tab_2_4:
                    st.caption("""
Historical data has been retrospectively audited to determine when jobs have included interventions that
could only be delivered by an EC or CC team.

This has then been used to inform the rate at which jobs with an EC or CC benefit are generated in the simulation.

While the numbers are low, it does not include a wide range of additional benefits that HEMS crews
bring to the scene. Work is now underway to improve the capture of these additional benefits, but
they are not reflected in the model.

For the model, it has been assumed that the split of CC calls, EC calls and calls where no CC or EC intervention
is delivered is consistent across the day. We do not have access to this data for time where jobs have not historically
been attended due to no resource being in service.

It is also assumed that the split of care categories is consistent across the year.

This data is affected by the fact that the historical actions will be affected by the crew that attended, and reflect
care delivered rather than ideal care. For example, if an EC crew attended a job that would benefit
from a CC intervention (due to no CC crew being on shift or the CC crew already being on another job),
only EC interventions would be delivered and only an EC benefit would have been recorded in the
dataset.
""")

                    st.subheader(
                        "Variation in projected missed calls across simulation runs"
                    )
                    st.write(
                        "This is compared with an estimate of the missed calls per year by category using historic rotas"
                    )

                    st.plotly_chart(
                        trial_results.plot_missed_calls_boxplot(
                            historical_results_obj=historical_data
                        )
                    )
                    tab_2_1.subheader(
                        "Variation in missed calls across simulation runs"
                    )

                    tab_2_1.plotly_chart(
                        trial_results.plot_missed_calls_boxplot(
                            historical_results_obj=historical_data,
                            what="summary",
                            historical_yearly_missed_calls_estimate=total_average_calls_received_per_year
                            * (float(missed_calls_hist_string) / 100),
                        )
                    )

                    st.subheader("Job Categories - Simulation vs Historical")

                    @st.fragment
                    def plot_cc_ec_split():
                        show_proportions_care_cat_plot = st.toggle(
                            "Show Proportions", True
                        )

                        st.plotly_chart(
                            trial_results.get_care_cat_counts_plot_sim(
                                show_proportions=show_proportions_care_cat_plot
                            )
                        )

                        st.caption("""
                        In this plot, we are predicting by the highest level of care provided.
                        (e.g. a job marked as 'CC' may also deliver an EC intervention, or an EC job may
                        also have a helicopter benefit)
                        """)

                        st.plotly_chart(
                            historical_data.PLOT_care_cat_counts_historic(
                                show_proportions=show_proportions_care_cat_plot
                            )
                        )

                        st.caption("""
                            We can also take a look at the proportion of jobs allocated to each category
                            at a high level to confirm the model is reflecting the historical trends.

                            Note that for historic data, we have excluded jobs that were not attended (and therefore
                            where the care category is not known) from the total number of jobs.
                            """)

                        st.dataframe(
                            trial_results.get_care_cat_proportion_table().drop(
                                columns=["Historic Job Counts", "Simulated Job Counts"]
                            )
                        )

                    plot_cc_ec_split()

            with tab_2_5:

                @st.fragment
                def job_count_heatmap():
                    normalise_heatmap_daily_jobs = st.toggle(
                        "Normalise by average daily jobs", False
                    )

                    fig_jobs_by_callsign_heatmap = trial_results.PLOT_job_count_heatmap(
                        normalise_per_day=normalise_heatmap_daily_jobs,
                        simulated_days=st.session_state.sim_duration_input,
                    )

                    st.plotly_chart(fig_jobs_by_callsign_heatmap)

                    fig_jobs_by_callsign_heatmap_monthly = (
                        trial_results.PLOT_job_count_heatmap_monthly(
                            normalise_per_day=normalise_heatmap_daily_jobs,
                            simulated_days=st.session_state.sim_duration_input,
                        )
                    )

                    st.plotly_chart(fig_jobs_by_callsign_heatmap_monthly)

                job_count_heatmap()

        # MARK: Historical Comparisons
        with tab3:
            tab_3_1, tab_3_2, tab_3_3, tab_3_4, tab_3_5, tab_3_6 = st.tabs(
                [
                    "Jobs per Month",
                    "Jobs by Hour of Day",
                    "Jobs by Day of Week",
                    "Jobs per Day - Distribution",
                    "Job Durations - Overall",
                    "Job Durations - Split",
                ]
            )

            with tab_3_1:

                @st.fragment
                def plot_monthly_jobs():
                    mj_1, mj_2 = st.columns(2)

                    show_real_data = mj_1.toggle(
                        "Compare with Real Data", value=True, disabled=False
                    )

                    show_individual_runs = mj_2.toggle(
                        "Show Individual Simulation Runs", value=False
                    )

                    if show_real_data:
                        historical_view_method = st.radio(
                            "Choose Historical Data Display Method",
                            ["Range", "Individual Lines"],
                            horizontal=True,
                        )
                        if historical_view_method == "Range":
                            show_historical_individual_years = False
                        else:
                            show_historical_individual_years = True
                    else:
                        show_historical_individual_years = False

                    fig_monthly_calls = trial_results.PLOT_monthly_calls(
                        show_individual_runs=show_individual_runs,
                        use_poppins=True,
                        show_historical=show_real_data,
                        show_historical_individual_years=show_historical_individual_years,
                        job_count_col="inc_date",
                    )

                    trial_results.PLOT_monthly_calls(
                        show_individual_runs=show_individual_runs,
                        use_poppins=False,
                        show_historical=show_real_data,
                        show_historical_individual_years=show_historical_individual_years,
                        job_count_col="inc_date",
                    ).write_html(
                        f"{APP_DIR}/fig_outputs/fig_monthly_calls.html",
                        full_html=False,
                        include_plotlyjs="cdn",
                    )  # , post_script = poppins_script)

                    return st.plotly_chart(fig_monthly_calls)

                plot_monthly_jobs()
                st.caption("""
Note that only full months in the simulation are included in this plot.
Partial months are excluded for ease of interpretation.
                           """)

            with tab_3_2:

                @st.fragment
                def plot_jobs_per_hour():
                    help_jph = get_text("help_jobs_per_hour", text_df)
                    jph_1, jph_2, jph_3, jph_4 = st.columns(4)

                    display_historic_jph = jph_1.toggle(
                        "Display Historic Data", value=True
                    )
                    average_per_month = jph_2.toggle(
                        "Display Average Calls Per Month", value=True, help=help_jph
                    )

                    display_advanced = jph_3.toggle(
                        "Display Advanced Plot", value=False
                    )

                    if not display_advanced:
                        display_error_bars_bar = jph_4.toggle("Display Variation")
                    else:
                        display_error_bars_bar = False

                    fig_hour_of_day = trial_results.PLOT_hourly_call_counts(
                        average_per_month=average_per_month,
                        box_plot=display_advanced,
                        show_error_bars_bar=display_error_bars_bar,
                        use_poppins=True,
                        show_historical=display_historic_jph,
                    )

                    trial_results.PLOT_hourly_call_counts(
                        average_per_month=average_per_month,
                        box_plot=display_advanced,
                        show_error_bars_bar=display_error_bars_bar,
                        use_poppins=False,
                        show_historical=display_historic_jph,
                    ).write_html(
                        f"{APP_DIR}/fig_outputs/fig_hour_of_day.html",
                        full_html=False,
                        include_plotlyjs="cdn",
                    )  # , post_script = poppins_script)

                    st.plotly_chart(fig_hour_of_day)

                plot_jobs_per_hour()

            with tab_3_3:

                @st.fragment
                def plot_jobs_per_day():
                    # help_jph = get_text("help_jobs_per_hour", text_df)
                    jpd_1, jpd_2, jpd_3, jpd_4 = st.columns(4)

                    display_historic_jph_pd = jpd_1.toggle(
                        "Display Historic Data", value=True, key="historic_pd"
                    )

                    average_per_month_pd = jpd_2.toggle(
                        "Display Average Calls Per Day",
                        value=True,
                        # help= help_jph,
                        key="average_pd",
                    )

                    display_advanced_pd = jpd_3.toggle(
                        "Display Advanced Plot", value=False, key="advanced_pd"
                    )

                    if not display_advanced_pd:
                        display_error_bars_bar_pd = jpd_4.toggle(
                            "Display Variation", key="variation_pd"
                        )
                    else:
                        display_error_bars_bar_pd = False

                    fig_day_of_week = trial_results.PLOT_daily_call_counts(
                        average_per_month=average_per_month_pd,
                        box_plot=display_advanced_pd,
                        show_error_bars_bar=display_error_bars_bar_pd,
                        use_poppins=True,
                        show_historical=display_historic_jph_pd,
                    )

                    trial_results.PLOT_daily_call_counts(
                        average_per_month=average_per_month_pd,
                        box_plot=display_advanced_pd,
                        show_error_bars_bar=display_error_bars_bar_pd,
                        use_poppins=False,
                        show_historical=display_historic_jph_pd,
                    ).write_html(
                        f"{APP_DIR}/fig_outputs/fig_day_of_week.html",
                        full_html=False,
                        include_plotlyjs="cdn",
                    )  # , post_script = poppins_script)

                    st.plotly_chart(fig_day_of_week)

                plot_jobs_per_day()

            #######################################
            # Histogram of calls received per day #
            #######################################

            with tab_3_4:

                @st.fragment()
                def plot_days_with_job_count_hist():
                    trial_results.PLOT_days_with_job_count_hist_ks()

                plot_days_with_job_count_hist()

            ##############################################
            # Historical Job Durations - Overall Summary #
            ##############################################

            with tab_3_5:

                @st.fragment
                def create_job_duration_plot():
                    plot_violin = st.toggle("Violin Plot?", value=False)

                    # Create plot for inclusion in streamlit
                    fig_job_durations_historical = trial_results.plot_historical_job_duration_vs_simulation_overall(
                        use_poppins=True,
                        write_to_html=True,
                        html_output_filepath=f"{APP_DIR}/fig_outputs/fig_job_durations_historical.html",
                        violin=plot_violin,
                    )

                    # Include job durations plot in streamlit app
                    st.plotly_chart(fig_job_durations_historical)

                    st.caption("""
    This plot looks at the total amount of time each resource was in use during the simulation.

    All simulated points are represented in the box plots.

    The blue bars give an indication of the historical averages. We would expect the median - the
    central horizontal line within the box portion of the box plots - to fall within the blue box for
    each resource type, and likely to be fairly central within that region.
    """)

                    historical_data.historical_time_df_cars_only = (
                        historical_data.historical_job_durations_breakdown[
                            historical_data.historical_job_durations_breakdown[
                                "vehicle_type"
                            ]
                            == "car"
                        ]
                    )
                    historical_data.historical_time_df_helos_only = (
                        historical_data.historical_job_durations_breakdown[
                            historical_data.historical_job_durations_breakdown[
                                "vehicle_type"
                            ]
                            == "helicopter"
                        ]
                    )

                    trial_results.simulated_job_time_df_cars_only = (
                        trial_results.resource_use_wide[
                            trial_results.resource_use_wide["vehicle_type"] == "car"
                        ]
                    )
                    trial_results.simulated_job_time_df_helos_only = (
                        trial_results.resource_use_wide[
                            trial_results.resource_use_wide["vehicle_type"]
                            == "helicopter"
                        ]
                    )

                    trial_results.calculate_ks_for_job_durations(
                        historical_data_series=historical_data.historical_time_df_helos_only[
                            historical_data.historical_time_df_helos_only["name"]
                            == "total_duration"
                        ]["value"],
                        simulated_data_series=trial_results.simulated_job_time_df_helos_only[
                            "resource_use_duration"
                        ],
                        what="helicopters",
                    )

                    trial_results.calculate_ks_for_job_durations(
                        historical_data_series=historical_data.historical_time_df_cars_only[
                            historical_data.historical_time_df_cars_only["name"]
                            == "total_duration"
                        ]["value"],
                        simulated_data_series=trial_results.simulated_job_time_df_cars_only[
                            "resource_use_duration"
                        ],
                        what="cars",
                    )

                create_job_duration_plot()

            ############################
            # Historical Job Durations - Breakdown #
            ############################

            with tab_3_6:
                st.plotly_chart(trial_results.PLOT_time_breakdown())

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

        # MARK: Debugging Plots
        with tab4:
            st.caption("""
This tab contains visualisations to help model authors do additional checks into the underlying functioning of the model.

Most users will not need to look at the visualisations in this tab.
            """)

            tab_4_1, tab_4_2, tab_4_3, tab_4_4 = st.tabs(
                [
                    "Debug Resources",
                    "Debug Events",
                    "Debug Outcomes",
                    "Debug Job Durations",
                ]
            )

            with tab_4_1:
                trial_results.display_resource_use_exploration()

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

                st.plotly_chart(trial_results.PLOT_jobs_per_callsign())

                st.subheader("Minutes per day on Shift")

                st.plotly_chart(trial_results.PLOT_daily_availability())

                st.subheader("Jobs Outcome by Category/Preference")

                @st.fragment
                def plot_preferred_outcome_by_hour():
                    show_proportions_job_outcomes_by_hour = st.toggle(
                        "Show Proportions",
                        False,
                        key="show_proportions_job_outcomes_by_hour",
                    )
                    st.plotly_chart(
                        trial_results.get_preferred_outcome_by_hour(
                            show_proportions=show_proportions_job_outcomes_by_hour
                        )
                    )

                plot_preferred_outcome_by_hour()

                st.plotly_chart(
                    trial_results.get_facet_plot_preferred_outcome_by_hour()
                )

            with tab_4_2:
                st.subheader("Event Overview")

                # @st.fragment
                # def event_overview_plot():
                #     runs_to_display_eo = st.multiselect(
                #         "Choose the runs to display",
                #         trial_results.run_results["run_number"].unique(),
                #         default=1,
                #     )

                #     st.plotly_chart(
                #         trial_results.PLOT_events_over_time(runs=runs_to_display_eo),
                #         use_container_width=True,
                #     )

                # event_overview_plot()

                st.plotly_chart(
                    trial_results.PLOT_cumulative_arrivals_per_run(),
                    use_container_width=True,
                )

                st.subheader("Event Counts")
                st.write(f"Period: {st.session_state.sim_duration_input} days")

                @st.fragment
                def event_funnel_plot():
                    hems_events_initial = [
                        "arrival",
                        "HEMS call start",
                        "HEMS allocated to call",
                        "HEMS mobile",
                        # "HEMS stood down en route",
                        "HEMS on scene",
                        # "HEMS patient treated (not conveyed)",
                        "HEMS leaving scene",
                        "HEMS arrived destination",
                        "HEMS clear",
                    ]

                    hems_events = st.multiselect(
                        "Choose the events to show",
                        trial_results.event_counts_long["time_type"].unique(),
                        hems_events_initial,
                    )

                    run_select = st.multiselect(
                        "Choose the runs to show",
                        trial_results.event_counts_long["run_number"].unique(),
                        1,
                    )

                    st.plotly_chart(
                        trial_results.PLOT_event_funnel_plot(hems_events, run_select)
                    )

                event_funnel_plot()

                @st.fragment
                def patient_viz():
                    st.subheader("Per-patient journey exploration")

                    patient_filter = st.selectbox(
                        "Select a patient", trial_results.run_results.P_ID.unique()
                    )

                    tab_list = st.tabs(
                        [
                            f"Run {i + 1}"
                            for i in range(st.session_state.number_of_runs_input)
                        ]
                    )

                    for idx, tab in enumerate(tab_list):
                        p_df = trial_results.run_results[
                            (trial_results.run_results.P_ID == patient_filter)
                            & (trial_results.run_results.run_number == idx + 1)
                        ].copy()

                        p_df["time_type"] = p_df["time_type"].astype("str")

                        tab.plotly_chart(
                            trial_results.PLOT_per_patient_events(p_df),
                            use_container_width=True,
                            key=f"p_viz_{patient_filter}_{idx}",
                        )

                patient_viz()

            with tab_4_3:

                @st.fragment
                def explore_outcomes():
                    plot_counts = st.toggle("Plot Counts", value=False)

                    st.caption(
                        "Note that these plots only cover patients for whom a resource was available to attend"
                    )

                    st.subheader("HEMS result by vehicle type")
                    try:
                        st.plotly_chart(
                            trial_results.plot_patient_outcomes(plot_counts=plot_counts)
                        )
                    except:
                        st.write("Error generating chart")

                    st.subheader("HEMS result by care category")
                    try:
                        st.plotly_chart(
                            trial_results.plot_patient_outcomes(
                                plot_counts=plot_counts,
                                group_cols="care_cat",
                            )
                        )
                    except:
                        st.write("Error generating chart")

                    st.subheader("HEMS Result by Outcome")
                    st.caption(
                        "Note this sums to 1 within each outcome, not within each hems result"
                    )
                    try:
                        st.plotly_chart(
                            trial_results.plot_patient_outcomes(
                                plot_counts=plot_counts,
                                group_cols="outcome",
                            )
                        )
                    except:
                        st.write("Error generating chart")

                    st.subheader("Outcome by Vehicle Type")
                    try:
                        st.plotly_chart(
                            trial_results.plot_patient_outcomes(
                                group_cols="vehicle_type",
                                outcome_col="outcome",
                                plot_counts=plot_counts,
                            )
                        )
                    except:
                        st.write("Error generating chart")

                    st.subheader("Vehicle Type by Care Cat")
                    st.caption(
                        "Note this sums to 1 within each cat, not within each vehicle type"
                    )
                    try:
                        st.plotly_chart(
                            trial_results.plot_patient_outcomes(
                                outcome_col="vehicle_type",
                                group_cols="care_cat",
                                plot_counts=plot_counts,
                            )
                        )
                    except:
                        st.write("Error generating chart")

                    st.header("Outcome variation across day")

                    try:
                        if plot_counts:
                            y_col_hourly_hems_result = "count"
                        else:
                            y_col_hourly_hems_result = "proportion"

                        st.plotly_chart(
                            trial_results.PLOT_outcome_variation_across_day(
                                y_col=y_col_hourly_hems_result
                            )
                        )
                    except:
                        st.write("Error generating chart")

                explore_outcomes()

            with tab_4_4:
                st.subheader("Duration by HEMS Outcome and Vehicle Type")
                st.caption(
                    "Outcomes are sorted by the average job duration (shortest first)"
                )
                st.plotly_chart(
                    trial_results.plot_total_times_by_hems_or_pt_outcome(
                        y="hems_result",
                        color="vehicle_type",
                        column_of_interest="hems_result",
                        show_group_averages=True,
                    )
                )

                st.markdown("#### Per-vehicle focus")
                st.plotly_chart(
                    trial_results.plot_total_times_by_hems_or_pt_outcome(
                        y="vehicle_type",
                        color="hems_result",
                        column_of_interest="hems_result",
                        show_group_averages=False,
                    )
                )

                st.divider()

                st.subheader("Duration by Patient Outcome and Vehicle Type")
                st.caption(
                    "Outcomes are sorted by the average job duration (shortest first)"
                )

                st.plotly_chart(
                    trial_results.plot_total_times_by_hems_or_pt_outcome(
                        y="outcome",
                        color="vehicle_type",
                        column_of_interest="outcome",
                        show_group_averages=True,
                    )
                )

                st.markdown("#### Per-vehicle focus")

                st.plotly_chart(
                    trial_results.plot_total_times_by_hems_or_pt_outcome(
                        y="vehicle_type",
                        color="outcome",
                        column_of_interest="outcome",
                        show_group_averages=False,
                    )
                )
        # MARK: Generate Report
        with tab5:

            @st.fragment()
            def generate_report_button():
                if st.button("Click here to generate the downloadable report"):
                    report_message.info("Generating Report...")
                    with st.spinner(
                        "Generating report. This may take a minute...", show_time=True
                    ):
                        try:
                            with open("fig_outputs/quarto_text.txt", "w") as text_file:
                                text_file.write(quarto_string)

                            msg = generate_quarto_report(run_quarto_check=False)

                            if msg == "success":
                                st.success("Report Available for Download")

                        except Exception as e:  # noqa
                            st.error(
                                "Report cannot be generated - please speak to a developer"
                            )

            generate_report_button()
