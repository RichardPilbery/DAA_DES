import streamlit as st
from _app_utils import iconMetricContainer, file_download_confirm
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("Compare Scenarios")

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

if not st.session_state.scenario_1_set:
    st.success("Scenario 1 has not been defined. Using Default (Current Operating Scenario) Parameters.")
if not st.session_state.scenario_2_set:
    st.error("Scenario 2 has not been defined. Please define a scenario to enable the run button on this page.")

    if st.button("Click here to go to the parameter page and define scenario 2",
                type="primary"):
        st.switch_page("setup.py")


col_button_1, col_button_2 = st.columns([0.4, 0.6])

if st.session_state.scenario_2_set:
    button_run_pressed = col_button_1.button("Run simulations", use_container_width=False)

    if button_run_pressed:
        @st.fragment
        def download_button_quarto():
            # st.download_button(
            with stylable_container(
                                        key="dl-button",
                                        css_styles="""
                    button {
                        float: right;
                        }
                        """
                                    ):
                st.button(
                    "Click here to download these results as a file",
                    on_click=file_download_confirm,
                    icon=":material/download:",
                    use_container_width=False
                    )


        with col_button_2:
            download_button_quarto()

        st.divider()

        col_x, col_y, col_z = st.columns([0.2, 0.6, 0.2])

        with col_y:
            with stylable_container(
            css_styles="""
                    {
                            background-color: #b6bfb9;
                            text-align: center;
                        }

                        """,
            key="info_average"
            ):

                st.info(f"All Metrics are averaged across {st.session_state.number_of_runs_input} simulation runs")



        col_scenario_1, col_scenario_blank, col_scenario_2 = st.columns([0.45, 0.1, 0.45])

        with col_scenario_1:
            st.header("Scenario 1")

            if st.session_state.scenario_1_set:
                st.info("User defined scenario")
            else:
                st.success("Default scenario")

            # TODO: Remove hardcoding and add logic
            st.metric("Metrics Better in This Scenario", value="1 of 5")

            st.divider()

            with iconMetricContainer(key="nonattend_metric", icon_unicode="e61f", family="outline"):
                    st.metric("Number of Calls DAAT Resource Couldn't Attend",
                            "184 of 1203 (15.3%)",
                            delta="+11.4% (137 fewer calls attended)",
                            delta_color="inverse",
                            border=True)
                    st.write(":red[WORSE: Fewer calls were attended in this scenario]")

                    st.caption("""
These are the 'missed' calls where no DAAT resource was available.
This could be due to
- no resource being on shift
- all resources being tasked to other jobs at the time of the call
""")

            st.divider()

            with iconMetricContainer(key="helo_util", icon_unicode="f60c", type="symbols"):
                    st.metric("Overall Helicopter Utilisation",
                            "78%",
                            delta="+17% (241 hours more)",
                            border=True)

                    st.caption("""
This is how much of the available time (where a helicopter is on shift and able to fly) the helicopter
was in use for.

Time where the helicopter was unable to fly due to weather conditions is not counted as available time here.
                """)

            st.divider()

            with iconMetricContainer(key="preferred_response_metric", icon_unicode="e838", family="outline"):
                    st.metric("Preferred Resource Allocated",
                            "802 of 1203 (66.7%)",
                            delta="-8.7% (105 fewer calls were allocated the preferred resource)",
                            delta_color="normal",
                            border=True)
                    st.write(":red[WORSE: Fewer calls received the ideal resource in this scenario]")

                    st.caption("""
This is the percentage of time where the 'preferred' resource was available at the time of the call
for response.
""")


        with col_scenario_2:
            st.header("Scenario 2")

            # TODO: Remove hardcoding and add logic
            st.info("User defined scenario: +1 helicopters")

            # TODO: Remove hardcoding and add logic
            st.metric("Metrics Better in This Scenario", value="4 of 5")

            st.divider()

            with iconMetricContainer(key="nonattend_metric", icon_unicode="e61f", family="outline"):
                    st.metric("Number of Calls DAAT Resource Couldn't Attend",
                            "47 of 1203 (3.9%)",
                            delta="-11.4% (137 more calls attended)",
                            delta_color="inverse",
                            border=True)
                    st.write(":green[BETTER: More calls were attended in this scenario]")
                    st.caption("""
These are the 'missed' calls where no DAAT resource was available.
This could be due to
- no resource being on shift
- all resources being tasked to other jobs at the time of the call
""")

            st.divider()

            with iconMetricContainer(key="helo_util", icon_unicode="f60c", type="symbols"):
                    st.metric("Overall Helicopter Utilisation",
                            "61%",
                            delta="-17% (241 hours less)",
                            border=True)

                    st.caption("""
This is how much of the available time (where a helicopter is on shift and able to fly) the helicopter
was in use for.

Time where the helicopter was unable to fly due to weather conditions is not counted as available time here.
                """)

            st.divider()

            with iconMetricContainer(key="preferred_response_metric", icon_unicode="e838", family="outline"):
                    st.metric("Preferred Resource Allocated",
                            "907 of 1203 (75.4%)",
                            delta="+8.7% (105 more calls were allocated the preferred resource)",
                            delta_color="normal",
                            border=True)
                    st.write(":green[BETTER: More calls received the ideal resource in this scenario]")

                    st.caption("""
This is the percentage of time where the 'preferred' resource was available at the time of the call
for response.
""")
