import streamlit as st
from _app_utils import iconMetricContainer, file_download_confirm

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("Compare Scenarios")

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

if not st.session_state.scenario_1_set:
    st.info("Scenario 1 has not been defined. Using Default (Current Operating Scenario) Parameters.")
if not st.session_state.scenario_2_set:
    st.error("Scenario 2 has not been defined. Please define a scenario to enable the run button on this page.")

    if st.button("Click here to go to the parameter page and define scenario 2",
                type="primary"):
        st.switch_page("setup.py")


if st.session_state.scenario_2_set:
    button_run_pressed = st.button("Run simulations")

    if button_run_pressed:
        @st.fragment
        def download_button_quarto():
            # st.download_button(
            st.button(
                "Click here to download these results as a file",
                on_click=file_download_confirm,
                icon=":material/download:"
                )

        download_button_quarto()

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

            with iconMetricContainer(key="nonattend_metric", icon_unicode="e0b0"):
                    st.metric("Number of Calls DAAT Resource Couldn't Attend",
                            "184 of 1203 (15.3%)",
                            border=True)
                    st.write(":red[BAD: Fewer calls were attended in this scenario]")

                    st.caption("""
These are the 'missed' calls where no DAAT resource was available.
This could be due to
- no resource being on shift
- all resources being tasked to other jobs at the time of the call
""")


        with col_scenario_2:
            st.header("Scenario 2")

            # TODO: Remove hardcoding and add logic
            st.info("User defined scenario: +1 helicopters")

            # TODO: Remove hardcoding and add logic
            st.metric("Metrics Better in This Scenario", value="4 of 5")

            st.divider()

            with iconMetricContainer(key="nonattend_metric", icon_unicode="e0b0"):
                    st.metric("Number of Calls DAAT Resource Couldn't Attend",
                            "47 of 1203 (3.9%)",
                            delta="-11.4% (137 more calls attended)",
                            delta_color="inverse",
                            border=True)
                    st.write(":green[GOOD: More calls were attended in this scenario]")
                    st.caption("""
These are the 'missed' calls where no DAAT resource was available.
This could be due to
- no resource being on shift
- all resources being tasked to other jobs at the time of the call
""")
