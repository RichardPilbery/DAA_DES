import streamlit as st
import pandas as pd
from pandas.api.types import CategoricalDtype
from datetime import time, datetime
import calendar

# Workaround to deal with relative import issues
# https://discuss.streamlit.io/t/importing-modules-in-pages/26853/2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _state_control import setup_state, reset_to_defaults, DEFAULT_INPUTS

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

setup_state()

from streamlit_extras.stylable_container import stylable_container
from air_ambulance_des.utils import (
    Utils,
    COLORSCHEME,
    MONTH_MAPPING,
    REVERSE_MONTH_MAPPING,
)
from _app_utils import (
    get_text,
    get_text_sheet,
    get_rota_month_strings,
)

SETUP_EXCEL = "Set parameters from Excel Template (recommended)"
SETUP_SCENARIO = "Choose a predefined scenario"
SETUP_MANUAL = "Set parameters manually"

u = Utils()

text_df = get_text_sheet("setup")

st.session_state["visited_setup_page"] = True

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title(get_text("page_title", text_df))

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

st.caption(get_text("page_description", text_df))

uploaded_file = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "radio_key" not in st.session_state:
    st.session_state.radio_key = 0


def reset_all_to_defaults():
    reset_to_defaults()
    st.session_state.uploader_key += 1
    st.session_state.radio_key += 1


st.button(
    get_text("reset_parameters_button", text_df),
    type="primary",
    on_click=reset_all_to_defaults,
    icon=":material/history:",
)

st.caption(get_text("reset_parameters_warning", text_df))


st.info(
    "Several ways of defining the model parameters - such as helicopter and car rotas, care level available at different times, and more - are provided."
)

setup_type = st.radio(
    "Choose how to set model parameters:",
    [
        SETUP_EXCEL,
        SETUP_SCENARIO,
        SETUP_MANUAL,
    ],
)

st.divider()

# MARK: Excel Template
if setup_type == SETUP_EXCEL:
    col_download, col_upload = st.columns([0.3, 0.7])

    @st.fragment
    def download_template():
        template_path = Path(__file__).parent / "inputs" / "parameter_template.xlsx"
        st.download_button(
            "Download file template",
            data=template_path.read_bytes(),
            file_name="parameter_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col_download:
        st.write("Click the button to download an Excel template you can fill in.")
        download_template()

    with col_upload:
        uploaded_file = st.file_uploader(
            label="Upload a completed template",
            accept_multiple_files=False,
            type=".xlsx",
            key=f"uploader_{st.session_state.uploader_key}",
        )

# MARK: Select Scenario
elif setup_type == SETUP_SCENARIO:
    scenario_files = {
        f.name: str(f.relative_to("scenarios"))
        for f in Path("scenarios").rglob("*.xlsx")
    }

    selected_scenario = st.radio(
        "Select a scenario",
        options=["None"] + list(scenario_files.keys()),
        format_func=lambda x: x.replace(".xlsx", ""),
        key=f"radio_{st.session_state.radio_key}",
    )

    if selected_scenario != "None":
        uploaded_file = f"scenarios/{scenario_files[selected_scenario]}"

if setup_type in [
    SETUP_EXCEL,
    SETUP_SCENARIO,
]:
    # If user has uploaded a file (the variable will return 'None' if the file uploader widget
    # has not yet been interacted with)
    if uploaded_file is not None:
        # Go through each sheet of the template, reading them one by onee and assigning them
        # to variables (which reflect how the related csvs are named in the data folder)
        scenario_details = pd.read_excel(uploaded_file, sheet_name="scenario_details")

        callsign_registration_lookup = pd.read_excel(
            uploaded_file, sheet_name="callsign_registration_lookup"
        )

        hems_rota = pd.read_excel(uploaded_file, sheet_name="HEMS_ROTA")

        service_schedules_by_model = pd.read_excel(
            uploaded_file, sheet_name="service_schedules_by_model"
        )
        rota_start_end_months = pd.read_excel(
            uploaded_file, sheet_name="rota_start_end_months"
        )

        service_history = pd.read_excel(
            uploaded_file, sheet_name="service_history", dtype="str", na_values=0
        )

        def try_date_format(x):
            try:
                return pd.to_datetime(x).strftime("%Y-%m-%d")
            except:
                return x

        service_history["last_service"] = service_history["last_service"].apply(
            lambda x: try_date_format(x)
        )

        upper_allowable_time_bounds = pd.read_excel(
            uploaded_file, sheet_name="upper_allowable_time_bounds"
        )

        school_holidays = pd.read_excel(
            uploaded_file, sheet_name="school_holidays", dtype="str"
        )

        school_holidays["start_date"] = school_holidays["start_date"].apply(
            lambda x: try_date_format(x)
        )

        school_holidays["end_date"] = school_holidays["end_date"].apply(
            lambda x: try_date_format(x)
        )

        additional_parameters = pd.read_excel(
            uploaded_file, sheet_name="additional_parameters"
        )

        # For each sheet from the template, overwrite the related csvs in the actual_data folder
        # TODO: Consider whether the school holidays file needs to be added in to this
        input_data_files = {
            # This first file has a slightly different structure to allow for
            "callsign_registration_lookup": callsign_registration_lookup.drop(
                columns=["vehicle_type", "callsign_group"]
            ).copy(),
            "HEMS_ROTA": hems_rota,
            "service_schedules_by_model": service_schedules_by_model,
            "rota_start_end_months": rota_start_end_months,
            "service_history": service_history,
            "upper_allowable_time_bounds": upper_allowable_time_bounds,
            "school_holidays": school_holidays,
        }

        for name, df in input_data_files.items():
            df.to_csv(f"actual_data/{name}.csv", index=False)

        st.info(
            "Here are your uploaded parameters. Please check these look correct, and update and reupload your parameter file if they are not.\n\n"
            "When you are happy, click on the 'Run Simulation' button in the left-hand bar to move to the model running page."
        )

        st.subheader(
            f"**Scenario:** {scenario_details[scenario_details['what'] == 'scenario_name']['value'].values[0]}"
        )

        st.write(
            f"**Description:** {scenario_details[scenario_details['what'] == 'scenario_description']['value'].values[0]}"
        )

        st.caption(
            f"**Notes:** {scenario_details[scenario_details['what'] == 'scenario_notes']['value'].values[0]}"
        )

        st.subheader("Available Resources")

        st.dataframe(callsign_registration_lookup, hide_index=True)

        st.subheader("Rota")

        st.dataframe(hems_rota, hide_index=True)

        st.subheader("Service Schedules per model")

        st.dataframe(service_schedules_by_model, hide_index=True)

        st.subheader("Summer and Winter Rota Start and End Months")

        st.dataframe(rota_start_end_months, hide_index=True)

        st.subheader("Servicing History")

        st.dataframe(
            service_history,
            hide_index=True,
            column_config={
                "last_service": st.column_config.DateColumn(
                    format="DD MMMM YYYY",
                ),
            },
        )

        st.subheader("Minimum and Maximum Durations for Activities")

        st.dataframe(upper_allowable_time_bounds, hide_index=True)

        st.subheader("School Holidays")

        st.dataframe(
            school_holidays,
            hide_index=True,
            column_config={
                "start_date": st.column_config.DateColumn(
                    format="DD MMMM YYYY",
                ),
                "end_date": st.column_config.DateColumn(
                    format="DD MMMM YYYY",
                ),
            },
        )

        st.subheader("Additional Parameters")

        # To avoid warnings about mixed data types within a single columns,here we transpose the
        # additional parameters dataframe to display it (so parameter names become columns, not rows)
        st.dataframe(additional_parameters.set_index("parameter").T, hide_index=True)

        st.session_state.number_of_runs_input = additional_parameters[
            additional_parameters["parameter"] == "number_of_runs"
        ]["value"].values[0]

        st.session_state.sim_duration_input = additional_parameters[
            additional_parameters["parameter"] == "simulation_duration_days"
        ]["value"].values[0]

        st.session_state.warm_up_duration = additional_parameters[
            additional_parameters["parameter"] == "simulation_warm_up_duration_hours"
        ]["value"].values[0]

        st.session_state.sim_start_date_input = (
            additional_parameters[
                additional_parameters["parameter"] == "simulation_start_date"
            ]["value"]
            .values[0]
            .strftime("%Y-%m-%d")
        )

        sim_start_time_input = additional_parameters[
            additional_parameters["parameter"] == "simulation_start_time"
        ]["value"].values[0]

        st.session_state.sim_start_time_input = pd.to_datetime(
            f"{st.session_state.sim_start_date_input} {sim_start_time_input}"
        ).strftime("%H:%M")

        st.session_state.master_seed = additional_parameters[
            additional_parameters["parameter"] == "master_random_seed"
        ]["value"].values[0]

        st.session_state.activity_duration_multiplier = float(
            additional_parameters[
                additional_parameters["parameter"] == "activity_duration_multiplier"
            ]["value"].values[0]
        )

        # Add a button to move to the model running page if parameter setup to the user's liking
        with stylable_container(
            css_styles=f"""
                            button {{
                                    background-color: {COLORSCHEME["teal"]};
                                    color: white;
                                    border-color: white;
                                }}
                                """,
            key="teal_buttons",
        ):
            if st.button(
                "Happy with your parameters?\n\nClick here to go to the model page.",
                icon=":material/play_circle:",
            ):
                st.switch_page("model.py")

# MARK: Build manually
else:
    st.header("HEMS Rota Builder")

    # MARK: m: Rota Dates
    @st.fragment
    def rota_start_end_dates():
        st.markdown("### Summer and Winter Setup")

        col_summer_start, col_summer_end, col_summer_spacing = st.columns(3)

        with col_summer_start:
            start_month = st.selectbox(
                "Select **start** month (inclusive) for Summer Rota",
                list(MONTH_MAPPING.keys()),
                index=st.session_state.summer_start_month_index,
                key="key_summer_start_month_index",
                on_change=lambda: setattr(
                    st.session_state,
                    "summer_start_month_index",
                    # note index is 1 less than actual month due to zero indexing in python
                    MONTH_MAPPING[st.session_state.key_summer_start_month_index] - 1,
                ),
            )
        with col_summer_end:
            end_month = st.selectbox(
                "Select **end** month (inclusive) for Summer Rota",
                list(MONTH_MAPPING.keys()),
                index=st.session_state.summer_end_month_index,
                key="key_summer_end_month_index",
                on_change=lambda: setattr(
                    st.session_state,
                    "summer_end_month_index",
                    # note index is 1 less than actual month due to zero indexing in python
                    MONTH_MAPPING[st.session_state.key_summer_end_month_index] - 1,
                ),
            )

            (
                start_month_num,
                end_month_num,
                summer_start_date,
                summer_end_date,
                summer_end_day,
                winter_start_date,
                winter_end_date,
                winter_end_day,
            ) = get_rota_month_strings(start_month, end_month)

        if start_month_num <= end_month_num:
            # Output
            st.write(
                f"☀️ Summer rota runs from {summer_start_date} to {summer_end_date} (inclusive)"
            )
            st.write(
                f"❄️ Winter rota runs from {winter_start_date} to {winter_end_date} (inclusive)"
            )

            pd.DataFrame(
                [
                    {"what": "summer_start_month", "month": start_month_num},
                    {"what": "summer_end_month", "month": end_month_num},
                    {"what": "summer_start_month_string", "month": start_month},
                    {"what": "summer_end_month_string", "month": end_month},
                ]
            ).to_csv("actual_data/rota_start_end_months.csv", index=False)
        else:
            default_start_month = DEFAULT_INPUTS["summer_start_month_index"] + 1
            default_end_month = DEFAULT_INPUTS["summer_end_month_index"] + 1
            default_start_month_name = REVERSE_MONTH_MAPPING[default_start_month]
            default_end_month_name = REVERSE_MONTH_MAPPING[default_end_month]

            default_summer_end_day = calendar.monthrange(2024, end_month_num)[
                1
            ]  # Assume leap year for Feb
            default_summer_end_date = (
                f"{default_summer_end_day}th {default_end_month_name}"
            )
            st.error(
                f"""End month must be later than start month. Using default summer start of 1st {default_start_month_name} and summer end of {default_summer_end_date}."""
            )
            pd.DataFrame(
                [
                    {"what": "summer_start_month", "month": default_start_month},
                    {"what": "summer_end_month", "month": default_end_month},
                ]
            ).to_csv("actual_data/rota_start_end_months.csv", index=False)

    rota_start_end_dates()

    # MARK: m: fleet setup
    @st.fragment
    def fleet_setup():
        st.markdown(f"""### {get_text("header_fleet_setup", text_df)}""")

        st.caption("""At present, the fleet cannot be expanded beyond what is currently
                available, though resources can be removed. """)

        col_1_fleet_setup, col_2_fleet_setup, blank_col_fleet_setup = st.columns(3)

        with col_1_fleet_setup:
            num_helicopters = st.number_input(
                get_text("set_num_helicopters", text_df),
                min_value=1,
                max_value=2,
                disabled=False,
                value=st.session_state.num_helicopters,
                help=get_text("help_helicopters", text_df),
                on_change=lambda: setattr(
                    st.session_state,
                    "num_helicopters",
                    st.session_state.key_num_helicopters,
                ),
                key="key_num_helicopters",
            )

        with col_2_fleet_setup:
            num_cars = st.number_input(
                get_text("set_num_additional_cars", text_df),
                min_value=0,
                max_value=1,
                disabled=False,
                value=st.session_state.num_cars,
                help=get_text("help_cars", text_df),
                on_change=lambda: setattr(
                    st.session_state, "num_cars", st.session_state.key_num_cars
                ),
                key="key_num_cars",
            )

        # Pull in the callsign and model lookup
        # For each, we will pull through the edited dataframe and the default dataframe
        # This allows us to ensure we are able to add in information about any default resources
        # even if we have removed it from the saved non-default lookups
        callsign_lookup = pd.read_csv("actual_data/callsign_registration_lookup.csv")
        callsign_lookup_default = pd.read_csv(
            "actual_data/callsign_registration_lookup_DEFAULT.csv"
        )
        callsign_lookup_columns = callsign_lookup.columns

        models = pd.read_csv("actual_data/service_schedules_by_model.csv")
        models_default = pd.read_csv(
            "actual_data/service_schedules_by_model_DEFAULT.csv"
        )
        models_columns = models.columns

        hems_rota_default = pd.read_csv("actual_data/HEMS_ROTA_DEFAULT.csv")

        callsign_lookup = pd.concat(
            [callsign_lookup, callsign_lookup_default]
        ).reset_index(drop=True)
        callsign_lookup = callsign_lookup.drop_duplicates(keep="first")
        print(callsign_lookup)

        models = pd.concat([models, models_default]).reset_index(drop=True)
        models = models.drop_duplicates(keep="first", subset=["model", "vehicle_type"])
        model_options = list(
            models[models["vehicle_type"] == "helicopter"]["model"].unique()
        )
        print(models)

        potential_fleet = callsign_lookup.merge(models, how="left", on=["model"])

        potential_fleet["callsign_group"] = potential_fleet["callsign"].str.extract(
            r"(\d+)"
        )

        potential_fleet["callsign_count"] = potential_fleet.groupby("callsign_group")[
            "callsign_group"
        ].transform("count")

        default_helos = potential_fleet[potential_fleet["vehicle_type"] == "helicopter"]

        default_cars = potential_fleet[
            (potential_fleet["vehicle_type"] == "car")
            & (potential_fleet["callsign_count"] == 1)
        ]

        default_helos = default_helos.head(num_helicopters)
        default_helos["has_car"] = True

        default_cars = default_cars.head(num_cars)

        st.markdown("#### Define the Helicopters")
        st.caption(
            "Columns with the :material/edit_note: symbol can be edited by double clicking the relevant table cell."
        )

        updated_helos_df = st.data_editor(
            default_helos,
            hide_index=True,
            key="helicopter_data_editor",
            column_order=[
                "callsign",
                #   "callsign_group",
                "registration",
                "has_car",
                "model",  # "service_schedule_months", "service_duration_weeks"
            ],
            column_config={
                "registration": st.column_config.TextColumn(
                    label="Registration", required=True
                ),
                "callsign": st.column_config.TextColumn(
                    label="Callsign", required=True
                ),
                "callsign_group": st.column_config.TextColumn(
                    label="Callsign", disabled=True
                ),
                "has_car": st.column_config.CheckboxColumn(label="Has a Backup Car"),
                "model": st.column_config.SelectboxColumn(
                    label="Model", options=model_options, required=True
                ),
                # "service_schedule_months": st.column_config.NumberColumn(label="Servicing Interval (Months)", disabled=True),
                # "service_duration_weeks": st.column_config.NumberColumn(label="Servicing Interval (Weeks)", disabled=True)
            },
        )

        st.markdown("#### Define the Backup Cars")

        backup_cars = potential_fleet[
            (potential_fleet["vehicle_type"] == "car")
            & (potential_fleet["callsign_count"] > 1)
            & (potential_fleet["callsign_group"]).isin(
                updated_helos_df[updated_helos_df["has_car"] == True][
                    "callsign_group"
                ].unique()
            )
        ]

        updated_backup_cars_df = st.data_editor(
            backup_cars,
            hide_index=True,
            key="backup_car_data_editor",
            column_order=[
                "callsign",
                #   "callsign_group",
                "registration",
                "model",  # "service_schedule_months", "service_duration_weeks"
            ],
            column_config={
                "registration": st.column_config.TextColumn(
                    label="Registration", required=True
                ),
                "callsign": st.column_config.TextColumn(
                    label="Callsign", required=True
                ),
                "callsign_group": st.column_config.TextColumn(
                    label="Callsign", disabled=True
                ),
                "model": st.column_config.SelectboxColumn(
                    label="Model",
                    options=potential_fleet[potential_fleet["vehicle_type"] == "car"][
                        "model"
                    ].unique(),
                    required=True,
                ),
                # "service_schedule_months": st.column_config.NumberColumn(label="Servicing Interval (Months)", disabled=True),
                # "service_duration_weeks": st.column_config.NumberColumn(label="Servicing Interval (Weeks)", disabled=True)
            },
        )

        st.markdown("#### Define the Standalone Cars")
        st.caption(
            "Columns with the :material/edit_note: symbol can be edited by double clicking the relevant table cell."
        )

        updated_cars_df = st.data_editor(
            default_cars,
            hide_index=True,
            key="standalone_car_data_editor",
            column_order=[
                "callsign",
                #   "callsign_group",
                "registration",
                "model",
                #   "service_schedule_months", "service_duration_weeks"
            ],
            column_config={
                "registration": st.column_config.TextColumn(
                    label="Registration", required=True
                ),
                "callsign": st.column_config.TextColumn(
                    label="Callsign", required=True
                ),
                # "callsign_group": st.column_config.TextColumn(label="Callsign", disabled=True),
                "model": st.column_config.SelectboxColumn(
                    label="Model",
                    options=potential_fleet[potential_fleet["vehicle_type"] == "car"][
                        "model"
                    ].unique(),
                    required=True,
                ),
                # "service_schedule_months": st.column_config.NumberColumn(label="Servicing Interval (Months)", disabled=True),
                # "service_duration_weeks": st.column_config.NumberColumn(label="Servicing Interval (Weeks)", disabled=True)
            },
        )

        final_df = pd.concat(
            [updated_helos_df, updated_backup_cars_df, updated_cars_df]
        )

        final_df[callsign_lookup_columns].drop_duplicates().to_csv(
            "actual_data/callsign_registration_lookup.csv", index=False
        )

        # final_df[models_columns].drop_duplicates().to_csv("actual_data/service_schedules_by_model.csv", index=False)

        # hems_rota_default['callsign_group'] = hems_rota_default['callsign_group'].astype('str')
        hems_rota = hems_rota_default[
            hems_rota_default["callsign"].isin(final_df.callsign.unique())
        ]
        hems_rota.to_csv("actual_data/HEMS_ROTA.csv", index=False)

        with st.expander("Click here to view the final fleet dataframes"):
            st.markdown("### Callsign Lookup")
            st.dataframe(
                final_df[callsign_lookup_columns].drop_duplicates(), hide_index=True
            )
            st.markdown("### Vehicle Model Details")
            st.dataframe(
                pd.read_csv("actual_data/service_schedules_by_model.csv"),
                hide_index=True,
            )

        st.markdown("### Individual Rota Setup")

        # Load callsign registration (this should reflect the output of fleet_setup)
        try:
            callsign_registration_lookup_df = pd.read_csv(
                "actual_data/callsign_registration_lookup.csv"
            )
        except FileNotFoundError:
            st.error(
                "actual_data/callsign_registration_lookup.csv not found. Please run Fleet Setup first."
            )
            st.stop()

        # Derive callsign_group and vehicle_type if not already perfect from the CSV
        # (The fleet_setup already does this, so it might be redundant if CSV is always up-to-date)
        if "callsign_group" not in callsign_registration_lookup_df.columns:
            callsign_registration_lookup_df["callsign_group"] = (
                callsign_registration_lookup_df["callsign"].str.extract(r"(\d+)")
            )
        if (
            "vehicle_type" not in callsign_registration_lookup_df.columns
            and "model" in callsign_registration_lookup_df.columns
        ):
            # This logic might need to align with how vehicle_type is determined in fleet_setup
            # For simplicity, let's assume it's present or use a simplified model-based inference
            callsign_registration_lookup_df["vehicle_type"] = (
                callsign_registration_lookup_df["model"].apply(
                    lambda x: (
                        "helicopter"
                        if isinstance(x, str) and "Airbus" in x or "H1" in x
                        else "car"
                    )  # Simplified
                )
            )

        # Load default rota (this is the master template for shifts)
        try:
            df_default_rota = pd.read_csv("actual_data/HEMS_ROTA_DEFAULT.csv")
        except FileNotFoundError:
            st.error("HEMS_ROTA_DEFAULT.csv not found!")
            st.stop()

        # Sort: group, then helicopter first
        # Ensure 'vehicle_type' exists before trying to use it for sorting
        if "vehicle_type" in callsign_registration_lookup_df.columns:
            vehicle_order = CategoricalDtype(
                categories=["helicopter", "car"], ordered=True
            )
            callsign_registration_lookup_df["vehicle_type"] = (
                callsign_registration_lookup_df["vehicle_type"].astype(vehicle_order)
            )
            sorted_lookup_df = callsign_registration_lookup_df.sort_values(
                by=["callsign_group", "vehicle_type"]
            )
        else:
            st.warning(
                "'vehicle_type' column missing in callsign lookup. Rota setup might be incomplete."
            )
            sorted_lookup_df = (
                callsign_registration_lookup_df.copy()
            )  # Proceed without vehicle type sorting if missing

        rota_data = {}
        helicopter_rotas_by_group = {}

        # Define columns for the rota editor UI and their configuration
        editable_rota_columns = [
            "category",
            "summer_start",
            "summer_end",
            "winter_start",
            "winter_end",
        ]
        rota_editor_column_config = {
            "category": st.column_config.SelectboxColumn(
                label="Category", options=["CC", "EC"], required=True
            ),
            "summer_start": st.column_config.NumberColumn(
                label="Summer Start Hour",
                min_value=0,
                max_value=23,
                step=1,
                required=True,
            ),
            "summer_end": st.column_config.NumberColumn(
                label="Summer End Hour",
                min_value=0,
                max_value=23,
                step=1,
                required=True,
            ),
            "winter_start": st.column_config.NumberColumn(
                label="Winter Start Hour",
                min_value=0,
                max_value=23,
                step=1,
                required=True,
            ),
            "winter_end": st.column_config.NumberColumn(
                label="Winter End Hour",
                min_value=0,
                max_value=23,
                step=1,
                required=True,
            ),
        }
        # Column order for the editor will be just the editable_rota_columns
        rota_editor_column_order = editable_rota_columns[:]

        for idx, row_lookup in sorted_lookup_df.iterrows():
            callsign = row_lookup["callsign"]
            model = row_lookup.get(
                "model", "N/A"
            )  # Use .get for safety if 'model' might be missing
            # Ensure vehicle_type and callsign_group are present in row_lookup
            vehicle_type = row_lookup.get("vehicle_type", "unknown")
            group = row_lookup.get("callsign_group", "unknown_group")

            if vehicle_type == "unknown" or group == "unknown_group":
                st.warning(
                    f"Skipping {callsign} due to missing vehicle_type or callsign_group in lookup data."
                )
                continue

            st.markdown(f"#### Set up rota for {callsign} ({model})")

            # Prepare existing_rota_for_resource: contains all columns (identifiers + schedule)
            existing_rota_for_resource = df_default_rota[
                df_default_rota["callsign"] == callsign
            ].copy()

            if existing_rota_for_resource.empty:
                num_rows_input = st.number_input(
                    f"Number of shifts for {callsign}",
                    min_value=1,
                    max_value=5,
                    value=2,
                    key=f"{callsign}_num_rows",
                )
                default_category_val = "EC" if vehicle_type == "helicopter" else "CC"
                existing_rota_for_resource = pd.DataFrame(
                    {
                        "callsign": [callsign] * num_rows_input,
                        "category": [default_category_val] * num_rows_input,
                        "vehicle_type": [vehicle_type] * num_rows_input,
                        "callsign_group": [group] * num_rows_input,
                        "summer_start": [7] * num_rows_input,
                        "winter_start": [7] * num_rows_input,
                        "summer_end": [19] * num_rows_input,
                        "winter_end": [19] * num_rows_input,
                    }
                )

            # This DataFrame will be passed to st.data_editor, containing only schedule columns
            data_for_rota_editor = existing_rota_for_resource[
                editable_rota_columns
            ].copy()

            current_edited_df = None  # This will hold the final DataFrame for this resource (with all columns)

            if vehicle_type == "car" and group in helicopter_rotas_by_group:
                toggle_key = f"{callsign}_same_as_heli"
                # Initialize session state for the toggle if not already present
                if toggle_key not in st.session_state:
                    st.session_state[toggle_key] = True  # Default to using heli rota

                use_heli_rota = st.toggle(
                    f"Use same rota as helicopter for group {group}?", key=toggle_key
                )

                if use_heli_rota:
                    # helicopter_rotas_by_group stores the full DataFrame (identifiers + schedule)
                    full_heli_rota_template = helicopter_rotas_by_group[group].copy()

                    # Update identifiers for the car
                    full_heli_rota_template["callsign"] = callsign
                    full_heli_rota_template["vehicle_type"] = "car"
                    full_heli_rota_template["callsign_group"] = (
                        group  # Should be the same group
                    )

                    current_edited_df = full_heli_rota_template
                    st.info(f"Using helicopter rota for {callsign} ({model}).")
                else:
                    # Custom rota for the car - show editor with only schedule columns
                    edited_schedule_df = st.data_editor(
                        data_for_rota_editor,  # Contains only editable_rota_columns
                        column_order=rota_editor_column_order,
                        column_config=rota_editor_column_config,
                        hide_index=True,
                        num_rows="dynamic",
                        key=f"{callsign}_custom_car_rota_editor",
                    )
                    # Reconstruct the full DataFrame
                    current_edited_df = edited_schedule_df.copy()
                    current_edited_df["callsign"] = callsign
                    current_edited_df["vehicle_type"] = "car"  # Explicitly car
                    current_edited_df["callsign_group"] = group
            else:
                # Default editor for helicopters, or cars not having the sync toggle
                edited_schedule_df = st.data_editor(
                    data_for_rota_editor,  # Contains only editable_rota_columns
                    column_order=rota_editor_column_order,
                    column_config=rota_editor_column_config,
                    hide_index=True,
                    num_rows="dynamic",
                    key=f"{callsign}_default_rota_editor",
                )
                # Reconstruct the full DataFrame
                current_edited_df = edited_schedule_df.copy()
                current_edited_df["callsign"] = callsign
                current_edited_df["vehicle_type"] = (
                    vehicle_type  # This vehicle's actual type
                )
                current_edited_df["callsign_group"] = group

            rota_data[callsign] = current_edited_df

            if vehicle_type == "helicopter":
                # current_edited_df is the helicopter's fully reconstructed rota
                helicopter_rotas_by_group[group] = current_edited_df.copy()

        st.markdown("## Full Rota Preview")
        if rota_data:
            full_rota_df = pd.concat(rota_data.values(), ignore_index=True)

            # Ensure all core columns from df_default_rota are present and in the correct order
            # The columns 'callsign', 'vehicle_type', 'callsign_group' should now be correctly
            # populated from the loop logic.

            # Make sure all columns from the default rota schema are present
            for col in df_default_rota.columns:
                if col not in full_rota_df.columns:
                    # Add missing columns with appropriate default (e.g., NA or specific values)
                    # This handles cases where df_default_rota might have more columns than generated.
                    full_rota_df[col] = pd.NA
                    if (
                        col == "category" and "category" not in full_rota_df.columns
                    ):  # Example default
                        full_rota_df[col] = (
                            "EC"  # Or some other logic for default category
                        )

            # Select and order columns according to the df_default_rota schema
            # Filter full_rota_df.columns to only those that are also in df_default_rota.columns
            # to prevent errors if full_rota_df accidentally gains extra columns not in the schema.
            final_columns_ordered = [
                col for col in df_default_rota.columns if col in full_rota_df.columns
            ]
            full_rota_df = full_rota_df[final_columns_ordered]

            st.dataframe(full_rota_df, hide_index=True)
            # Save to HEMS_ROTA.csv (the working file), not HEMS_ROTA_DEFAULT.csv
            full_rota_df.to_csv("actual_data/HEMS_ROTA.csv", index=False)
            st.success("Final rota automatically saved to HEMS_ROTA.csv!")
        else:
            st.warning("No rota data generated to preview or save.")

    fleet_setup()

    # MARK: m: Demand Adjust
    # st.header("Demand Parameters")

    # st.caption("""
    # At present it is only possible to apply an overall demand adjustment, which increases the number
    # of calls that will be received per day in the model. You can use the slider below to carry out
    # this adjustment.

    # In future, the model will allow more granular control of additional demand.
    # """)

    # demand_adjust_type = "Overall Demand Adjustment"

    # # TODO: Add to session state
    # # demand_adjust_type = st.radio("Adjust High-level Demand",
    # #          ["Overall Demand Adjustment",
    # #           "Per Season Demand Adjustment",
    # #           "Per AMPDS Code Demand Adjustment"],
    # #           key="demand_adjust_type",
    # #           horizontal=True,
    # #           disabled=True
    # #           )

    # if demand_adjust_type == "Overall Demand Adjustment":
    #     overall_demand_mult = st.slider(
    #         "Overall Demand Adjustment",
    #         min_value=90,
    #         max_value=200,
    #         value=st.session_state.overall_demand_mult,
    #         format="%d%%",
    #         on_change= lambda: setattr(st.session_state, 'overall_demand_mult', st.session_state.key_overall_demand_mult),
    #         key="key_overall_demand_mult"
    #         )

    overall_demand_mult = 100

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

    # MARK: m: Extra Params

    st.divider()

    st.header(get_text("additional_params_header", text_df))

    st.caption(get_text("additional_params_help", text_df))

    @st.fragment
    def additional_params_expander():
        st.markdown("### Replications")

        # Note that to work correctly with session state (without odd bugs where you have to use
        # the sliders twice in a row to get the input to 'stick') this pattern is used throughout.
        # The variable the output is assigned to is inconsequential.
        # The manual setting of the key is necessary. Generally the name of the key is set to the
        # same as th session state variable with 'key_' at the start, but this naming convention
        # isn't necessary to its functioning.
        # The session state variable itself - as referenced in the on_change parameter - must first
        # be initialised in the file app/_state_control.py

        number_of_runs_input = st.slider(
            "Number of Runs",
            min_value=1,
            max_value=100,
            value=st.session_state.number_of_runs_input,
            on_change=lambda: setattr(
                st.session_state,
                "number_of_runs_input",
                st.session_state.key_number_of_runs_input,
            ),
            key="key_number_of_runs_input",
            help="""
    This controls how many times the simulation will repeat. On each repeat, while the core parameters
    will stay the same, slight randomness will occur in the patterns of arrivals and in choices like which
    resource responds, what the outcome of each job is, and more.
    \n\n
    Think of it like watching 10 different versions of the same day in an emergency control room.
    In one run, a call comes in five minutes earlier. In another, a different crew is dispatched.
    A patient might recover quickly in one case, or need extra help in another.
    \n\n
    By running the simulation multiple times, we get a better sense of the range of things that *could*
    happen — not just a single outcome, but the full picture of what’s likely and what’s possible.
    For example, by running 10 replications of 1 year, we can get a better sense of how well the model
    will cope in busier and quieter times, helping to understand the likely range of performance that
    might be observed in the real world.
    """,
        )

        st.markdown("### Simulation and Warm-up Duration")

        col_button_1, col_button_2, col_button_3, col_button_4 = st.columns(4)

        col_button_1.button(
            "Set sim duration to 4 weeks",
            on_click=lambda: setattr(st.session_state, "sim_duration_input", 7 * 4),
        )

        col_button_2.button(
            "Set sim duration to 1 year",
            on_click=lambda: setattr(st.session_state, "sim_duration_input", 365),
        )

        col_button_3.button(
            "Set sim duration to 2 years",
            on_click=lambda: setattr(st.session_state, "sim_duration_input", 365 * 2),
        )

        col_button_4.button(
            "Set sim duration to 3 years",
            on_click=lambda: setattr(st.session_state, "sim_duration_input", 365 * 3),
        )

        sim_duration_input = st.slider(
            "Simulation Duration (days)",
            min_value=1,
            max_value=365 * 3,
            value=st.session_state.sim_duration_input,
            on_change=lambda: setattr(
                st.session_state,
                "sim_duration_input",
                st.session_state.key_sim_duration_input,
            ),
            key="key_sim_duration_input",
        )

        warm_up_duration = st.slider(
            "Warm-up Duration (hours)",
            min_value=0,
            max_value=24 * 10,
            value=st.session_state.warm_up_duration,
            on_change=lambda: setattr(
                st.session_state,
                "warm_up_duration",
                st.session_state.key_warm_up_duration,
            ),
            key="key_warm_up_duration",
        )

        st.caption(
            f"The simulation will not start recording metrics until {(warm_up_duration / 24):.2f} days have elapsed"
        )

        st.markdown("### Dates")

        st.caption("""
    This affects when the simulation will start. The simulation time affects the rota that is used,
    as well as the demand (average number of jobs received).
    """)

        sim_start_date_input = st.date_input(
            "Select the starting day for the simulation",
            value=st.session_state.sim_start_date_input,
            on_change=lambda: setattr(
                st.session_state,
                "sim_start_date_input",
                st.session_state.key_sim_start_date_input.strftime("%Y-%m-%d"),
            ),
            key="key_sim_start_date_input",
            min_value="2023-01-01",
            max_value="2027-01-01",
        ).strftime("%Y-%m-%d")

        sim_start_time_input = st.time_input(
            "Select the starting time for the simulation",
            value=st.session_state.sim_start_time_input,
            on_change=lambda: setattr(
                st.session_state,
                "sim_start_time_input",
                st.session_state.key_sim_start_time_input.strftime("%H:%M"),
            ),
            key="key_sim_start_time_input",
        ).strftime("%H:%M")

        st.markdown("### Reproducibility and Variation")

        master_seed = st.number_input(
            "Set the master random seed",
            1,
            100000,
            42,
            key="key_master_seed",
            on_change=lambda: setattr(
                st.session_state, "master_seed", st.session_state.key_master_seed
            ),
        )

        st.markdown("### Other Modifiers")

        activity_duration_multiplier = st.slider(
            "Apply a multiplier to activity times",
            value=st.session_state.activity_duration_multiplier,
            max_value=2.0,
            min_value=0.7,
            on_change=lambda: setattr(
                st.session_state,
                "activity_duration_multiplier",
                st.session_state.key_activity_duration_multiplier,
            ),
            key="key_activity_duration_multiplier",
            help="""
    This lengthens or shortens all generated activity times by the selected multiplier.
    \n\n
    For example, if a journey time of 10 minutes was generated, this would be shortened to 8 minutes if
    this multiplier is set to 0.8, or lengthened to 12 minutes if the multiplier was set to 1.2.
    \n\n
    This can be useful for experimenting with the impact of longer or shorter activity times, or for
    temporarily adjusting the model if activity times are not accurately reflecting reality.
    """,
        )

        create_animation_input = st.toggle(
            "Create Animation",
            value=st.session_state.create_animation_input,
            on_change=lambda: setattr(
                st.session_state,
                "create_animation_input",
                st.session_state.key_create_animation_input,
            ),
            key="key_create_animation_input",
            disabled=True,
            help="Coming soon!",
        )

        amb_data = st.toggle(
            "Model ambulance service data",
            value=st.session_state.amb_data,
            on_change=lambda: setattr(
                st.session_state, "amb_data", st.session_state.key_amb_data
            ),
            key="key_amb_data",
            disabled=True,
            help="Coming soon!",
        )

    with st.expander(get_text("additional_params_expander_title", text_df)):
        additional_params_expander()

    st.divider()

    with st.sidebar:
        with stylable_container(
            css_styles=f"""
                        button {{
                                background-color: {COLORSCHEME["teal"]};
                                color: white;
                                border-color: white;
                            }}
                            """,
            key="teal_buttons",
        ):
            if st.button(
                "Finished setting up parameters?\n\nClick here to go to the model page.",
                icon=":material/play_circle:",
            ):
                st.switch_page("model.py")
