import streamlit as st
import pandas as pd
from datetime import time, datetime
# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from streamlit_extras.stylable_container import stylable_container

from utils import Utils

st.set_page_config(layout="wide")

setup_state()

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.session_state["visited_setup_page"] = True

help_helicopters = """
This parameter relates to the number of helicopters that will be present.

You can set the helicopter type, crew type and operational hours in the next step.
"""

help_cars = """
This parameter relates to the number of *additional* critical care cars that exist as an entity
separate from the helicopters.

You can set the operational hours and crew type of each car separately in the next step.

Do not include cars that are used by a helicopter crew in the case of helicopter unavailability
(due to servicing, inclement weather, etc.)
"""

st.title("Model Setup")

st.write("Welcome to the model setup page. Here, you can enter the ")

with st.expander("Want to set up the model parameters from a template? Click here."):
    col_template_1, col_template_2 = st.columns(2)

    with col_template_1:
        st.file_uploader("Click here to upload parameters from a template file", type="xlsx")


    with col_template_2:
        st.write("Not used the template before? Click here to download it.")
        st.download_button(data="parameter_template.xlsx",
                           label="Download Excel Template for Model Parameters",
                           file_name="daa_simulation_model_parameters_TEMPLATE.xlsx")

# TODO - make this operational
st.button("Return Model to Current DAA Operational Parameters", type="primary",
          on_click=reset_to_defaults)

st.divider()

st.header("Fleet Setup")


col_1_fleet_setup, col_2_fleet_setup, blank_col_fleet_setup = st.columns(3)

with col_1_fleet_setup:
    num_helicopters = st.number_input(
        "🚁 Set the number of helicopters",
        min_value=1,
        max_value=5,
        value=st.session_state.num_helicopters,
        help=help_helicopters,
        on_change= lambda: setattr(st.session_state, 'num_helicopters', st.session_state.key_num_helicopters),
        key="key_num_helicopters"
        )

with col_2_fleet_setup:
    num_cars = st.number_input(
        "🚗 Set the number of additional cars",
        min_value=0,
        max_value=5,
        value=st.session_state.num_cars,
        help=help_cars,
        on_change= lambda: setattr(st.session_state, 'num_cars', st.session_state.key_num_cars),
        key="key_num_cars"
        )

st.subheader("Fleet Makeup")

st.caption("☀️ Summer Rota runs from March to October")
st.caption("❄️ Winter Rota runs from November to February")

original_rota = Utils.HEMS_ROTA
original_rota["callsign_count"] = original_rota.groupby('callsign_group')['callsign_group'].transform('count')

default_helos = original_rota[original_rota["vehicle_type"]=="helicopter"]
default_cars = original_rota[(original_rota["vehicle_type"]=="car") &
                             (original_rota["callsign_count"] == 1)]

fleet_makeup_list = []

if st.session_state.num_helicopters == 1:
    fleet_makeup_list.append(default_helos.head(1))
    defaul_helos = default_helos.head(1)
elif st.session_state.num_helicopters >= 2:
    fleet_makeup_list.append(default_helos)

if st.session_state.num_cars >= 1:
    fleet_makeup_list.append(default_cars)

initial_fleet_df = pd.concat(fleet_makeup_list).drop(columns=["callsign_count"])

fleet_additional_car_list = []
fleet_additional_helo_list = []

# For any helicopters over and above the real helicopters that already exist,
# populate the dataframe with a default helicopter and a plausible callsign
if st.session_state.num_helicopters >2:
    for i in range(1, st.session_state.num_helicopters-1):
        fleet_additional_helo_list.append(
            {
        "callsign"             : f"H{initial_fleet_df['callsign_group'].astype('int').max()+i}",
        "category"             : "CC",
        "vehicle_type"         : "helicopter",
        "callsign_group"       : initial_fleet_df['callsign_group'].astype('int').max()+i,
        "summer_start"         : 7,
        "winter_start"         : 7,
        "summer_end"           : 19,
        "winter_end"           : 17,
        "model"                : "Airbus H145"
    }
        )

# For any cars over and above the real cars that already exist,
# populate the dataframe with a default car and a plausible callsign
# NOTE - this is only for cars that don't have an associated helicopter
# We will need to add car backups within the same callsign for any helicopter
# that is created
if st.session_state.num_cars >1:
    for i in range(1, st.session_state.num_cars):
        fleet_additional_car_list.append(
            {
        "callsign"             : f"CC{initial_fleet_df['callsign_group'].astype('int').max()+st.session_state.num_helicopters+i}",
        "category"             : "CC",
        "vehicle_type"         : "car",
        "callsign_group"       : initial_fleet_df['callsign_group'].astype('int').max()+st.session_state.num_helicopters+i,
        "summer_start"         : 8,
        "winter_start"         : 8,
        "summer_end"           : 18,
        "winter_end"           : 18,
        "model"                : "Volvo XC90"
    }
        )

if (st.session_state.num_helicopters > 2):

    final_helo_df = pd.concat(
        [default_helos,
        pd.DataFrame(fleet_additional_helo_list).set_index('callsign')]
        ).drop(columns=["callsign_count", "callsign_group"])
else:
    final_helo_df = default_helos.drop(columns=["callsign_count", "callsign_group"])

if (st.session_state.num_cars > 1):
    final_car_df = pd.concat(
        [default_cars,
        pd.DataFrame(fleet_additional_car_list).set_index('callsign')]
        ).drop(columns=["callsign_count", "callsign_group"])
else:
    final_car_df = default_cars.drop(columns=["callsign_count", "callsign_group"])

# st.write(final_fleet_df)

for time_col in ["summer_start", "summer_end", "winter_start", "winter_end"]:
    final_helo_df[time_col] = final_helo_df[time_col].apply(lambda x: time(x, 0))
    final_car_df[time_col] = final_car_df[time_col].apply(lambda x: time(x, 0))

final_helo_df["vehicle_type"] = final_helo_df["vehicle_type"].apply(lambda x: x.title())

# Create an editable dataframe for people to modify the parameters in
updated_helo_df = st.data_editor(
    final_helo_df.reset_index(),
    disabled=["vehicle_type"],
    hide_index=True,
    column_order=["vehicle_type", "callsign", "category", "model",
                  "summer_start", "summer_end", "winter_start", "winter_end"],
    column_config={
        "vehicle_type": "Vehicle Type",
        "callsign": "Callsign",
        "category": st.column_config.SelectboxColumn(
            "Care Type", options=["EC", "CC"]),
            "model": st.column_config.SelectboxColumn(
            "Model", options=["Airbus EC135", "Airbus H145"],
        ),
        "summer_start": st.column_config.TimeColumn(
            "Summer Start", format="HH:mm"
        ),
        "summer_end": st.column_config.TimeColumn(
            "Summer End", format="HH:mm"
        ),
        "winter_start": st.column_config.TimeColumn(
            "Winter Start", format="HH:mm"
        ),
        "winter_end": st.column_config.TimeColumn(
            "Winter End", format="HH:mm"
        )
        }
    )

final_car_df["vehicle_type"] = final_car_df["vehicle_type"].apply(lambda x: x.title())

updated_car_df = st.data_editor(final_car_df.reset_index(),
                                hide_index=True,
                                 disabled=["vehicle_type"],
                                 column_order=["vehicle_type", "callsign", "category", "model", "summer_start", "summer_end", "winter_start", "winter_end"],
                                 column_config={
        "vehicle_type": "Vehicle Type",
        "callsign": "Callsign",
        "category": st.column_config.SelectboxColumn(
            "Care Type", options=["EC", "CC"]
        ),
            "model": st.column_config.SelectboxColumn(
            "Model", options=["Volvo XC90"],
        ),
        "summer_start": st.column_config.TimeColumn(
            "Summer Start", format="HH:mm"
        ),
        "summer_end": st.column_config.TimeColumn(
            "Summer End", format="HH:mm"
        ),
        "winter_start": st.column_config.TimeColumn(
            "Winter Start", format="HH:mm"
        ),
        "winter_end": st.column_config.TimeColumn(
            "Winter End", format="HH:mm"
        )

        }
                                 )

st.divider()

st.header("Demand Parameters")

demand_adjust_type_high_level = st.radio("Adjust High-level Demand",
         ["Overall Demand Adjustment",
          "Per Season Demand Adjustment"],
          key="demand_adjust_type")

if demand_adjust_type_high_level == "Overall Demand Adjustment":
    overall_demand_mult = st.slider(
        "Overall Demand Adjustment",
        min_value=90,
        max_value=200,
        value=st.session_state.overall_demand_mult,
        format="%d%%",
        on_change= lambda: setattr(st.session_state, 'overall_demand_mult', st.session_state.key_overall_demand_mult),
        key="key_overall_demand_mult"
        )
elif demand_adjust_type_high_level == "Per Season Demand Adjustment":
    season_demand_col_1, season_demand_col_2, season_demand_col_3, season_demand_col_4 = st.columns(4)

    spring_demand_mult = season_demand_col_1.slider(
        "🌼 Spring Demand Adjustment",
        min_value=90,
        max_value=200,
        value=st.session_state.spring_demand_mult,
        format="%d%%",
        on_change= lambda: setattr(st.session_state, 'spring_demand_mult', st.session_state.key_spring_demand_mult),
        key="key_spring_demand_mult"
        )

    summer_demand_mult = season_demand_col_2.slider(
        "☀️ Summer Demand Adjustment",
        min_value=90,
        max_value=200,
        value=st.session_state.summer_demand_mult,
        format="%d%%",
        on_change= lambda: setattr(st.session_state, 'summer_demand_mult', st.session_state.key_summer_demand_mult),
        key="key_summer_demand_mult"
        )

    autumn_demand_mult = season_demand_col_3.slider(
        "🍂 Autumn Demand Adjustment",
        min_value=90,
        max_value=200,
        value=st.session_state.autumn_demand_mult,
        format="%d%%",
        on_change= lambda: setattr(st.session_state, 'autumn_demand_mult', st.session_state.key_autumn_demand_mult),
        key="key_autumn_demand_mult"
        )

    winter_demand_mult = season_demand_col_4.slider(
        "❄️ Winter Demand Adjustment",
        min_value=90,
        max_value=200,
        value=st.session_state.winter_demand_mult,
        format="%d%%",
        on_change= lambda: setattr(st.session_state, 'winter_demand_mult', st.session_state.key_winter_demand_mult),
        key="key_winter_demand_mult"
        )


with st.expander("Click here to set advanced model parameters"):
    amb_data = st.toggle(
        "Model ambulance service data",
        value=st.session_state.amb_data,
        on_change= lambda: setattr(st.session_state, 'amb_data', st.session_state.key_amb_data),
        key="key_amb_data"
        )

    sim_duration_input =  st.slider(
        "Simulation Duration (days)",
        min_value=1,
        max_value=365,
        value=st.session_state.sim_duration_input,
        on_change= lambda: setattr(st.session_state, 'sim_duration_input', st.session_state.key_sim_duration_input),
        key="key_sim_duration_input"
        )

    warm_up_duration =  st.slider(
        "Warm-up Duration (hours)",
        min_value=0,
        max_value=24*10,
        value=st.session_state.warm_up_duration,
        on_change= lambda: setattr(st.session_state, 'warm_up_duration', st.session_state.key_warm_up_duration),
        key="key_warm_up_duration"
        )

    start_date_input = st.date_input(
        "Select the starting day for the simulation",
        value=st.session_state.start_date_input,
        on_change=lambda: setattr(st.session_state, 'start_date_input', st.session_state.key_start_date_input),
        key="key_start_date_input"
        )

    start_time_input = st.time_input(
        "Select the starting time for the simulation",
        value=st.session_state.start_date_input,
        on_change=lambda: setattr(st.session_state, 'start_time_input', st.session_state.key_start_time_input),
        key="key_start_time_input"
        )

    st.markdown(f"The simulation will not start recording metrics until {(warm_up_duration / 24):.2f} days have elapsed")

    number_of_runs_input = st.slider(
        "Number of Runs",
        min_value=1,
        max_value=30,
        value=st.session_state.number_of_runs_input,
        on_change= lambda: setattr(st.session_state, 'number_of_runs_input', st.session_state.key_number_of_runs_input),
        key="key_number_of_runs_input"
        )

    create_animation_input = st.toggle(
        "Create Animation",
        value=st.session_state.create_animation_input,
        on_change= lambda: setattr(st.session_state, 'create_animation_input', st.session_state.key_create_animation_input),
        key="key_create_animation_input"
        )


# TODO - This just currently redownloads the blank template
# Will need to populate this
st.download_button(data="parameter_template.xlsx",
                    label="Save Your Parameters to a File",
                    type="primary",
                    file_name=f"daa_simulation_model_parameters_{datetime.now()}.xlsx"
)

# TODO: Make these do something more than just display a notification!
st.caption("""
If you want to compare multiple scenarios, set up each scenario using the sliders below
or by importing a completed parameter template file, then click on the relevant button.
""")
scenario_1_button_col, scenario_2_button_col = st.columns(2)

with scenario_1_button_col:
    with stylable_container(
        css_styles="""
                button {
                        background-color: #00205b;
                        color: white;
                    }
                    """,
        key="blue_buttons"
        ):
        st.button(
            "Set Scenario 1 to Current Parameters",
            on_click=set_scenario_1_params
            )

with scenario_2_button_col:
    with stylable_container(
        css_styles="""
                button {
                        background-color: #00205b;
                        color: white;
                    }
                    """,
        key="blue_buttons"
        ):
        st.button(
        "Set Scenario 2 to Current Parameters",
        type="primary",
        on_click=set_scenario_2_params
        )

st.divider()

with st.sidebar:
    with stylable_container(
        css_styles="""
                button {
                        background-color: green;
                        color: white;
                    }
                    """,
        key="green_buttons"
        ):
        if st.button("Finished Setting Up Parameters? Click here to go to the model page",
                     icon=":material/play_circle:"):
            st.switch_page("model.py")
        if st.button("Set up two scenarios you want to compare? Click here to go to the scenario comparison page",
                     icon=":material/compare:"):
            st.switch_page("compare_scenarios.py")
