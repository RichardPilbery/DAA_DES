import streamlit as st
import pandas as pd
# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import Utils

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

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
st.button("Return Model to Current DAA Operational Parameters", type="primary")

st.divider()

st.header("Fleet Setup")

col_1_fleet_setup, col_2_fleet_setup, blank_col_fleet_setup = st.columns(3)

with col_1_fleet_setup:
    num_helicopters = st.number_input("🚁 Set the number of helicopters",
                    1, 5, value=2,
                    help=help_helicopters, key="num_helicopters")

with col_2_fleet_setup:
    num_cars = st.number_input("🚗 Set the number of additional cars",
                    0, 5, value=1,
                    help=help_cars, key="num_cars")

st.subheader("Fleet Makeup")

st.caption("☀️ Summer Rota runs from March to October")
st.caption("❄️ Winter Rota runs from November to February")


original_rota = Utils.HEMS_ROTA
original_rota["callsign_count"] = original_rota.groupby('callsign_group')['callsign_group'].transform('count')

default_helos = original_rota[original_rota["vehicle_type"]=="helicopter"]
default_cars = original_rota[(original_rota["vehicle_type"]=="car") &
                             (original_rota["callsign_count"] == 1)]

fleet_makeup_list = []

if num_helicopters == 1:
    fleet_makeup_list.append(default_helos.head(1))
    defaul_helos = default_helos.head(1)
elif num_helicopters >= 2:
    fleet_makeup_list.append(default_helos)

if num_cars >= 1:
    fleet_makeup_list.append(default_cars)

initial_fleet_df = pd.concat(fleet_makeup_list).drop(columns=["callsign_count"])

fleet_additional_car_list = []
fleet_additional_helo_list = []

if num_helicopters >2:
    for i in range(1, num_helicopters-1):
        fleet_additional_helo_list.append(
            {
        "callsign"             : f"H{initial_fleet_df['callsign_group'].astype('int').max()+i}",
        "category"             : "CC",
        "vehicle_type"         : "helicopter",
        "callsign_group"       : initial_fleet_df['callsign_group'].astype('int').max()+i,
        "summer_start"         : 7,
        "winter_start"         : 7,
        "summer_end"           : 19,
        "winter_end"           : 17
    }
        )

if num_cars >1:
    for i in range(1, num_cars):
        fleet_additional_car_list.append(
            {
        "callsign"             : f"CC{initial_fleet_df['callsign_group'].astype('int').max()+num_helicopters+i}",
        "category"             : "CC",
        "vehicle_type"         : "car",
        "callsign_group"       : initial_fleet_df['callsign_group'].astype('int').max()+num_helicopters+i,
        "summer_start"         : 8,
        "winter_start"         : 8,
        "summer_end"           : 18,
        "winter_end"           : 18
    }
        )

if (num_helicopters > 2):

    final_helo_df = pd.concat(
        [default_helos,
        pd.DataFrame(fleet_additional_helo_list).set_index('callsign')]
        )
else:
    final_helo_df = default_helos

st.data_editor(final_helo_df)

if (num_cars > 1):
    final_car_df = pd.concat(
        [default_cars,
        pd.DataFrame(fleet_additional_car_list).set_index('callsign')]
        )
else:
    final_car_df = default_cars

# st.write(final_fleet_df)

st.data_editor(final_car_df)

st.divider()

st.header("Demand Parameters")

demand_adjust_type_high_level = st.radio("Adjust High-level Demand",
         ["Overall Demand Adjustment",
          "Per Season Demand Adjustment"],
          key="demand_adjust_type")

if demand_adjust_type_high_level == "Overall Demand Adjustment":
    overall_demand_mult = st.slider(
        "Overall Demand Multiplier",
        min_value=0.9,
        max_value=2.0,
        value=1.0,
        key="overall_demand_mult"
        )
elif demand_adjust_type_high_level == "Per Season Demand Adjustment":
    season_demand_col_1, season_demand_col_2, season_demand_col_3, season_demand_col_4 = st.columns(4)

    spring_demand_mult = season_demand_col_1.slider(
        "🌼 Spring Demand Multiplier",
        min_value=0.9,
        max_value=2.0,
        value=1.0,
        key="spring_demand_mult"
        )

    summer_demand_mult = season_demand_col_2.slider(
        "☀️ Summer Demand Multiplier",
        min_value=0.9,
        max_value=2.0,
        value=1.0,
        key="summer_demand_mult"
        )

    autumn_demand_mult = season_demand_col_3.slider(
        "🍂 Autumn Demand Multiplier",
        min_value=0.9,
        max_value=2.0,
        value=1.0,
        key="autumn_demand_mult"
        )

    winter_demand_mult = season_demand_col_4.slider(
        "❄️ Winter Demand Multiplier",
        min_value=0.9,
        max_value=2.0,
        value=1.0,
        key="winter_demand_mult"
        )


with st.expander("Click here to set advanced model parameters"):
    amb_data = st.toggle("Model ambulance service data", value=False, key="amb_data")

    sim_duration_input =  st.slider("Simulation Duration (days)", 1, 365, 7, key="sim_duration_input")

    warm_up_duration =  st.slider("Warm-up Duration (hours)", 0, 24*10, 0, key="warm_up_duration")
    st.markdown(f"The simulation will not start recording metrics until {(warm_up_duration / 24):.2f} days have elapsed")

    number_of_runs_input = st.slider("Number of Runs", 1, 30, 5, key="number_of_runs_input")

    create_animation_input = st.toggle("Create Animation", value=False, key="create_animation_input")
