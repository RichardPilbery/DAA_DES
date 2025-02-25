import streamlit as st
import pandas as pd
from datetime import time, datetime
# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from _state_control import setup_state, reset_to_defaults, \
                            set_scenario_1_params, set_scenario_2_params
from streamlit_extras.stylable_container import stylable_container

from utils import Utils

from _app_utils import get_text, get_text_sheet, DAA_COLORSCHEME

st.set_page_config(layout="wide")

setup_state()

text_df=get_text_sheet("setup")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.session_state["visited_setup_page"] = True

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title(get_text("page_title", text_df))

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

st.caption(get_text("page_description", text_df))

# with st.expander(get_text("expander_title_template", text_df)):
#     st.caption(get_text("expander_description_template", text_df))
#     col_template_1, col_template_2 = st.columns(2)

#     with col_template_1:
#         st.file_uploader(get_text("file_uploader_button", text_df), type="xlsx",
#                          disabled=True)


#     with col_template_2:
#         st.write(get_text("file_template_download_hint", text_df))
#         st.download_button(data="parameter_template.xlsx",
#                            label=get_text("file_template_download_button", text_df),
#                            file_name="daa_simulation_model_parameters_TEMPLATE.xlsx",
#                            disabled=True)

# st.divider()

st.button(get_text("reset_parameters_button", text_df),
          type="primary",
          on_click=reset_to_defaults,
          icon=":material/history:")
st.caption(get_text("reset_parameters_warning", text_df))

st.divider()

@st.fragment
def fleet_setup():

    st.header(get_text("header_fleet_setup", text_df))

    st.caption("""
*Note that while it is not currently possible to change the number of vehicles in
the fleet, this is planned for a future version of the model.*
    """)

    col_1_fleet_setup, col_2_fleet_setup, blank_col_fleet_setup = st.columns(3)

    with col_1_fleet_setup:
        num_helicopters = st.number_input(
            get_text("set_num_helicopters", text_df),
            min_value=1,
            max_value=5,
            disabled=True,
            value=st.session_state.num_helicopters,
            help=get_text("help_helicopters", text_df),
            on_change= lambda: setattr(st.session_state, 'num_helicopters', st.session_state.key_num_helicopters),
            key="key_num_helicopters"
            )

    with col_2_fleet_setup:
        num_cars = st.number_input(
            get_text("set_num_additional_cars", text_df),
            min_value=0,
            max_value=5,
            disabled=True,
            value=st.session_state.num_cars,
            help=get_text("help_cars", text_df),
            on_change= lambda: setattr(st.session_state, 'num_cars', st.session_state.key_num_cars),
            key="key_num_cars"
            )

    # st.subheader(get_text("header_fleet_makeup", text_df))

    original_rota = pd.read_csv("actual_data/HEMS_ROTA.csv")
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
    if st.session_state.num_cars > 1:
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

    return final_helo_df, final_car_df

final_helo_df, final_car_df = fleet_setup()

st.markdown("#### Set the Fleet Details")

col_summer, col_winter, col_summer_winter_spacing = st.columns(3)
with col_summer:
    st.caption(get_text("summer_rota_help", text_df))
with col_winter:
    st.caption(get_text("winter_rota_help", text_df))



@st.fragment
def fleet_editors(final_helo_df, final_car_df):
    # Create an editable dataframe for people to modify the parameters in
    st.markdown("##### Helicopters")
    st.caption("Columns with the :material/edit_note: symbol can be edited by double clicking the relevant table cell.")

    updated_helo_df = st.data_editor(
        final_helo_df.reset_index(),
        disabled=["vehicle_type"],
        hide_index=True,
        column_order=["vehicle_type", "callsign", "category", "model",
                    "summer_start", "summer_end", "winter_start", "winter_end"],
        column_config={
            "vehicle_type": "Vehicle Type",
            "callsign": st.column_config.TextColumn(
                "Callsign", disabled=True
                ),
            "category": st.column_config.SelectboxColumn(
                "Care Type",
                options=["EC", "CC"],
                disabled=True
                ),
            "model": st.column_config.SelectboxColumn(
                "Model",
                options=["Airbus EC135", "Airbus H145"],
            ),
            "summer_start": st.column_config.TimeColumn(
                "Summer Start", format="HH:mm",
                disabled=False
            ),
            "summer_end": st.column_config.TimeColumn(
                "Summer End", format="HH:mm",
                disabled=False
            ),
            "winter_start": st.column_config.TimeColumn(
                "Winter Start", format="HH:mm",
                disabled=False
            ),
            "winter_end": st.column_config.TimeColumn(
                "Winter End", format="HH:mm",
                disabled=False
            )
            }
        )

    st.caption("""
:red_car: **All helicopters in the model are automatically assumed to have a backup car assigned to them for use
when the helicopter is unavailable for any reason.**
""")


    st.markdown("##### Additional Cars")
    st.caption("""
In the table below you can also alter the parameters of the *additional* cars that have their own separate callsign
group and operate as a totally separate resource to the helicopters.""")
    st.caption("Columns with the :material/edit_note: symbol can be edited by double clicking the relevant table cell.")

    final_car_df["vehicle_type"] = final_car_df["vehicle_type"].apply(lambda x: x.title())

    updated_car_df = st.data_editor(final_car_df.reset_index(),
                                    hide_index=True,
                                    disabled=["vehicle_type"],
                                    column_order=["vehicle_type", "callsign", "category", "model", "summer_start", "summer_end", "winter_start", "winter_end"],
                                    column_config={
            "vehicle_type": "Vehicle Type",
            "callsign": st.column_config.TextColumn(
                "Callsign", disabled=True
                ),
            "category": st.column_config.SelectboxColumn(
                "Care Type", options=["EC", "CC"],
                disabled=True
            ),
                "model": st.column_config.SelectboxColumn(
                "Model", options=["Volvo XC90"],
                disabled=True
            ),
            "summer_start": st.column_config.TimeColumn(
                "Summer Start", format="HH:mm",
                disabled=False
            ),
            "summer_end": st.column_config.TimeColumn(
                "Summer End", format="HH:mm",
                disabled=False
            ),
            "winter_start": st.column_config.TimeColumn(
                "Winter Start", format="HH:mm",
                disabled=False
            ),
            "winter_end": st.column_config.TimeColumn(
                "Winter End", format="HH:mm",
                disabled=False
            )

            }
                                    )

    # Join the dataframes back together
    final_rota = pd.concat([updated_helo_df, updated_car_df]).drop(columns='index')

    # Convert vehicle type column back to expected capitalisation
    final_rota["vehicle_type"] = final_rota["vehicle_type"].str.lower()

    # Add callsign group column back in
    final_rota["callsign_group"] = final_rota["callsign"].str.extract("(\d+)")

    # Take a copy of the helicopter df to allow us to create the cars that go alongside it
    # We can assume operating hours and care category will be the same
    companion_car_df = updated_helo_df.copy()

    # TODO: For now, we have hardcoded companion cars to be Volvo XC90s
    companion_car_df["model"] = "Volvo XC90"
    # Register them as cars instead of helicopters
    companion_car_df["vehicle_type"] = "car"
    # Update callsign
    companion_car_df["callsign"] = companion_car_df["callsign"].str.replace("H", "CC")
    # Add callsign group column
    companion_car_df["callsign_group"] = companion_car_df["callsign"].str.extract("(\d+)")
    # Remove 'last_service' date
    companion_car_df = companion_car_df.drop(columns=["last_service"])

    # Join this onto the list of helicopters and separate cars, then sort
    final_rota = (pd.concat([final_rota, companion_car_df])
                  .sort_values(["callsign_group", "vehicle_type"], ascending=[True, False])
                  .drop(columns='index')
                  )

    # Remove the servicing columns as they will reflect the originally set models in the
    # default rota
    final_rota = final_rota.drop(columns=["service_schedule_months","service_duration_weeks"])

    # Merge with service schedule df to get actual servicing intervals for chosen model
    # of helicopter
    final_rota = final_rota.merge(Utils.SERVICING_SCHEDULES_BY_MODEL, on="model", how="left")

    # Convert the time columns back to something the model can understand
    for col in ["summer_start", "winter_start", "summer_end", "winter_end"]:
        final_rota[col] = final_rota[col].apply(lambda x: x.hour)

    # Sort the columns into the order of the original rota
    column_order = Utils.HEMS_ROTA_DEFAULT.columns
    final_rota = final_rota[column_order]

    # print(final_rota)

    # Write the rota back to a csv
    final_rota.to_csv('actual_data/HEMS_ROTA.csv', index=False)

fleet_editors(final_helo_df, final_car_df)

st.divider()

st.header("Demand Parameters")

st.caption("""
At present it is only possible to apply an overall demand adjustment, which increases the number
of calls that will be received per day in the model. You can use the slider below to carry out
this adjustment.

In future, the model will allow more granular control of additional demand.
""")

demand_adjust_type = "Overall Demand Adjustment"

# TODO: Add to session state
# demand_adjust_type = st.radio("Adjust High-level Demand",
#          ["Overall Demand Adjustment",
#           "Per Season Demand Adjustment",
#           "Per AMPDS Code Demand Adjustment"],
#           key="demand_adjust_type",
#           horizontal=True,
#           disabled=True
#           )

if demand_adjust_type == "Overall Demand Adjustment":
    overall_demand_mult = st.slider(
        "Overall Demand Adjustment",
        min_value=90,
        max_value=200,
        value=st.session_state.overall_demand_mult,
        format="%d%%",
        on_change= lambda: setattr(st.session_state, 'overall_demand_mult', st.session_state.key_overall_demand_mult),
        key="key_overall_demand_mult"
        )
# elif demand_adjust_type == "Per Season Demand Adjustment":
#     season_demand_col_1, season_demand_col_2, season_demand_col_3, season_demand_col_4 = st.columns(4)

#     spring_demand_mult = season_demand_col_1.slider(
#         "🌼 Spring Demand Adjustment",
#         min_value=90,
#         max_value=200,
#         value=st.session_state.spring_demand_mult,
#         format="%d%%",
#         on_change= lambda: setattr(st.session_state, 'spring_demand_mult', st.session_state.key_spring_demand_mult),
#         key="key_spring_demand_mult"
#         )

#     summer_demand_mult = season_demand_col_2.slider(
#         "☀️ Summer Demand Adjustment",
#         min_value=90,
#         max_value=200,
#         value=st.session_state.summer_demand_mult,
#         format="%d%%",
#         on_change= lambda: setattr(st.session_state, 'summer_demand_mult', st.session_state.key_summer_demand_mult),
#         key="key_summer_demand_mult"
#         )

#     autumn_demand_mult = season_demand_col_3.slider(
#         "🍂 Autumn Demand Adjustment",
#         min_value=90,
#         max_value=200,
#         value=st.session_state.autumn_demand_mult,
#         format="%d%%",
#         on_change= lambda: setattr(st.session_state, 'autumn_demand_mult', st.session_state.key_autumn_demand_mult),
#         key="key_autumn_demand_mult"
#         )

#     winter_demand_mult = season_demand_col_4.slider(
#         "❄️ Winter Demand Adjustment",
#         min_value=90,
#         max_value=200,
#         value=st.session_state.winter_demand_mult,
#         format="%d%%",
#         on_change= lambda: setattr(st.session_state, 'winter_demand_mult', st.session_state.key_winter_demand_mult),
#         key="key_winter_demand_mult"
#         )

# elif demand_adjust_type == "Per AMPDS Code Demand Adjustment":
#     st.write("Coming Soon!")

# else:
#     st.error("TELL A DEVELOPER: Check Conditional Code for demand modifier in setup.py")

st.divider()

# st.subheader(get_text("header_scenario", text_df))

# # TODO: Make these do something more than just display a notification!
# st.caption(get_text("scenario_comparison_help", text_df))
# scenario_1_button_col, scenario_2_button_col = st.columns(2)

# with scenario_1_button_col:
#     with stylable_container(
#         css_styles="""
#                 button {
#                         background-color: #00205b;
#                         color: white;
#                     }
#                     """,
#         key="blue_buttons"
#         ):
#         st.button(
#             "Set Scenario 1 to Current Parameters",
#             on_click=set_scenario_1_params,
#             disabled=True
#             )

# with scenario_2_button_col:
#     with stylable_container(
#         css_styles="""
#                 button {
#                         background-color: #00205b;
#                         color: white;
#                     }
#                     """,
#         key="blue_buttons"
#         ):
#         st.button(
#         "Set Scenario 2 to Current Parameters",
#         type="primary",
#         disabled=True,
#         on_click=set_scenario_2_params
#         )

# st.divider()

st.header(get_text("additional_params_header", text_df))

st.caption(get_text("additional_params_help", text_df))

with st.expander(get_text("additional_params_expander_title", text_df)):
    number_of_runs_input = st.slider(
        "Number of Runs",
        min_value=1,
        max_value=30,
        value=st.session_state.number_of_runs_input,
        on_change= lambda: setattr(st.session_state, 'number_of_runs_input', st.session_state.key_number_of_runs_input),
        key="key_number_of_runs_input"
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

    st.caption(f"The simulation will not start recording metrics until {(warm_up_duration / 24):.2f} days have elapsed")

    sim_start_date_input = st.date_input(
        "Select the starting day for the simulation",
        value=st.session_state.sim_start_date_input,
        on_change=lambda: setattr(st.session_state, 'sim_start_date_input', st.session_state.key_sim_start_date_input),
        key="key_sim_start_date_input"
        )

    sim_start_time_input = st.time_input(
        "Select the starting time for the simulation",
        value=st.session_state.sim_start_time_input,
        on_change=lambda: setattr(st.session_state, 'sim_start_time_input', st.session_state.key_sim_start_time_input),
        key="key_sim_start_time_input"
        )

    create_animation_input = st.toggle(
        "Create Animation",
        value=st.session_state.create_animation_input,
        on_change= lambda: setattr(st.session_state, 'create_animation_input', st.session_state.key_create_animation_input),
        key="key_create_animation_input",
        disabled=True
        )

    amb_data = st.toggle(
        "Model ambulance service data",
        value=st.session_state.amb_data,
        on_change= lambda: setattr(st.session_state, 'amb_data', st.session_state.key_amb_data),
        key="key_amb_data",
        disabled=True
        )

st.divider()

# st.subheader(get_text("header_params_reuse", text_df))

# st.caption(get_text("params_reuse_description", text_df))

# # TODO - This just currently redownloads the blank template
# # Will need to populate this
# st.download_button(data="parameter_template.xlsx",
#                     label="Click to Download the Parameter File",
#                     type="primary",
#                     disabled=True,
#                     file_name=f"daa_simulation_model_parameters_{datetime.now()}.xlsx"
# )

# st.divider()

with st.sidebar:
    with stylable_container(
        css_styles=f"""
                    button {{
                            background-color: {DAA_COLORSCHEME['teal']};
                            color: white;
                            border-color: white;
                        }}
                        """, key="teal_buttons"
            ):
        if st.button("Finished setting up parameters?\n\nClick here to go to the model page",
                     icon=":material/play_circle:"):
            st.switch_page("model.py")
        # if st.button("Set up two scenarios you want to compare? Click here to go to the scenario comparison page",
        #              icon=":material/compare:"):
        #     st.switch_page("compare_scenarios.py")
