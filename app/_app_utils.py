import streamlit as st
import pandas as pd
import os
import subprocess
import platform
from datetime import datetime
import calendar
from air_ambulance_des.utils import (
    COLORSCHEME,
    to_military_time,
    get_rota_month_strings,
)
from streamlit_extras.stylable_container import stylable_container


def file_download_confirm():
    st.toast("File Downloaded", icon=":material/download:")


@st.cache_data
def get_text_sheet(sheet):
    return pd.read_excel("app/assets/text.xlsx", sheet_name=sheet, engine="calamine")


@st.cache_data
def get_text(reference, text_df):
    return text_df[text_df["reference"] == reference]["text"].values[0]


@st.cache_data
def get_quarto(repo_name, quarto_version="1.5.57"):
    print(f"Output of platform.processor(): {platform.processor()}")
    print(f"type:  {type(platform.processor())}")
    print("Attempting to download Quarto")
    # Download Quarto
    os.system(
        f"wget https://github.com/quarto-dev/quarto-cli/releases/download/v{quarto_version}/quarto-{quarto_version}-linux-amd64.tar.gz"
    )

    # Create directory and extract Quarto
    os.system(f"tar -xvzf quarto-{quarto_version}-linux-amd64.tar.gz")
    # Check the contents of the folder we are in
    os.system("pwd")

    # # Ensure PATH is updated in the current Python process
    # Check current path
    os.system("echo $PATH")
    # Create a folder and symlink quarto to that location
    os.system(f"mkdir -p /mount/src/{repo_name}/local/bin")
    os.system(
        f"ln -s /mount/src/{repo_name}/quarto-{quarto_version}/bin/quarto /mount/src/{repo_name}/local/bin"
    )
    # Update path
    os.system(f"echo 'export PATH=$PATH:/mount/src/{repo_name}/local/bin' >> ~/.bashrc")
    os.system("source /etc/bash.bashrc")
    # alternative method for good measure
    os.environ["PATH"] = f"/mount/src/{repo_name}/local/bin:{os.environ['PATH']}"

    # ensure path updates have propagated through
    print(os.environ["PATH"])
    # Install jupyter even if not in requirements
    os.system("python3 -m pip install jupyter")
    # Install second copy of requirements (so accessible by Quarto - can't access packages
    # that are installed as part of community cloud instance setup process)
    os.system(f"python3 -m pip install -r /mount/src/{repo_name}/requirements.txt")

    print("Trying to run 'quarto check' command")
    try:
        os.system("quarto check")
        result = subprocess.run(
            ["quarto", "check"], capture_output=True, text=True, shell=True
        )
        print(result.stdout)
        print(result.stderr)
        print("Quarto check run")
    except PermissionError:
        print("Permission error encountered when running 'quarto check'")
    except:  # noqa
        print("Other unspecified error when running quarto check")


@st.fragment
def generate_quarto_report(run_quarto_check=False):
    """
    Passed an empty placeholder, put in a download button or a disabled download
    button in the event of failure
    """
    print("Trying to generate a downloadable quarto report")
    output_dir = os.path.join(os.getcwd(), "app/outputs")
    qmd_filename = "app/air_ambulance_simulation_output.qmd"
    qmd_path = os.path.join(os.getcwd(), qmd_filename)
    print(f"Trying to find quarto template in {qmd_path}")
    html_filename = os.path.basename(qmd_filename).replace(".qmd", ".html")
    # html_filename = f"simulation_output_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    # print(html_filename)
    # dest_html_path = os.path.join(output_dir,f"simulation_output_{datetime.now().strftime('%H-%m-%d_%H%M')}.html")
    dest_html_path = os.path.join(output_dir, html_filename)

    # print(dest_html_path)

    try:
        if run_quarto_check:
            print("Trying to run 'quarto check' command")
            subprocess.run(["quarto", "check"])

        print("Running Quarto Render Command")

        ## forces result to be html
        result = subprocess.run(
            [
                "quarto",
                "render",
                qmd_path,
                "--to",
                "html",
                "--output-dir",
                output_dir,
                # , "--output-file"
                # , html_filename
            ],
            capture_output=True,
            text=True,
        )

        print("Quarto Render Command run succesfully")
        print(f"Destination Path: {dest_html_path}")
    except:  # noqa
        ## error message
        print("Report cannot be generated")

    if os.path.exists(dest_html_path):
        print(
            f"Destination file {dest_html_path} found in filesystem - obtaining for download"
        )
        with open(dest_html_path, "r") as f:
            html_data = f.read()

        with stylable_container(
            key="report_dl_buttons",
            css_styles=f"""
                    button {{
                            background-color: {COLORSCHEME["green"]};
                            color: white;
                            border-color: white;
                        }}
                        """,
        ):
            st.download_button(
                label="Download Report",
                data=html_data,
                file_name=html_filename,
                mime="text/html",
            )

            return "success"
    else:
        ## error message
        print("Generated file found not in filesystem")
        try:
            print(f"Report failed to generate\n\n_{result}_")
        except UnboundLocalError:
            print("Report failed to generate")

        st.button(label="Error Generating Downloadable Report", disabled=True)

        st.warning("""It has not been possible to generate a downloadable copy of the simulation outputs.
                Please speak to a developer""")

        return "failure"


def summary_sidebar(quarto_string):
    with stylable_container(
        css_styles="""
hr {
    border-color: #a6093d;
    background-color: #a6093d;
    color: #a6093d;
    height: 1px;
  }
""",
        key="hr",
    ):
        st.divider()
    if "number_of_runs_input" in st.session_state:
        with stylable_container(
            key="green_buttons",
            css_styles=f"""
                    button {{
                            background-color: {COLORSCHEME["teal"]};
                            color: white;
                            border-color: white;
                        }}
                        """,
        ):
            if st.button(
                "Want to change some parameters? Click here.",
                type="primary",
                icon=":material/display_settings:",
            ):
                st.switch_page("setup.py")
        st.subheader("Model Input Summary")
        quarto_string += "## Model Input Summary\n\n"

        rota_start_end_months = pd.read_csv("actual_data/rota_start_end_months.csv")

        (
            start_month_num,
            end_month_num,
            summer_start_date,
            summer_end_date,
            summer_end_day,
            winter_start_date,
            winter_end_date,
            winter_end_day,
        ) = get_rota_month_strings(
            start_month=rota_start_end_months[
                rota_start_end_months["what"] == "summer_start_month_string"
            ]["month"].values[0],
            end_month=rota_start_end_months[
                rota_start_end_months["what"] == "summer_end_month_string"
            ]["month"].values[0],
        )

        summer_string = f"☀️ Summer rota runs from {summer_start_date} to {summer_end_date} (inclusive)"
        winter_string = f"❄️ Winter rota runs from {winter_start_date} to {winter_end_date} (inclusive)"

        st.write(summer_string)
        st.write(winter_string)

        quarto_string += "\n\n"
        quarto_string += summer_string
        quarto_string += "\n"
        quarto_string += winter_string
        quarto_string += "\n\n"

        num_helos_string = f"Number of Helicopters: {st.session_state.num_helicopters}"
        quarto_string += "### "
        quarto_string += num_helos_string
        quarto_string += "\n\n"
        st.write(f"### {num_helos_string}")

        rota = (
            pd.read_csv("actual_data/HEMS_ROTA.csv")
            .merge(
                pd.read_csv("actual_data/callsign_registration_lookup.csv"),
                on="callsign",
                how="left",
            )
            .merge(
                pd.read_csv("actual_data/service_schedules_by_model.csv"),
                on=["model", "vehicle_type"],
                how="left",
            )
        )

        # Group by helicopter resources
        helicopters = rota[rota["vehicle_type"] == "helicopter"]
        grouped_heli = helicopters.groupby("callsign")

        quarto_string = ""

        for callsign, group in grouped_heli:
            model = group.iloc[0]["model"]
            header = f"#### {callsign} is an {model} and\n"
            body = ""
            for _, row in group.iterrows():
                summer = f"\n- runs a {row['category']} service from {to_military_time(row['summer_start'])} to {to_military_time(row['summer_end'])} in summer"
                winter = f"\n- runs a {row['category']} service from {to_military_time(row['winter_start'])} to {to_military_time(row['winter_end'])} in winter"
                body += f"    {summer}\n    {winter}\n\n"
            result = header + body + "\n"
            quarto_string += result
            st.caption(result)

        # Grouping callsign groups
        callsign_group_counts = rota["callsign_group"].value_counts().reset_index()
        callsign_group_counts.columns = ["callsign_group", "count"]

        extra_cars_only = list(
            callsign_group_counts[callsign_group_counts["count"] == 1][
                "callsign_group"
            ].values
        )
        backup_cars_only = list(
            callsign_group_counts[callsign_group_counts["count"] > 1][
                "callsign_group"
            ].values
        )

        # Backup cars
        backup_cars = rota[
            rota["callsign_group"].isin(backup_cars_only)
            & (rota["vehicle_type"] != "helicopter")
        ]
        grouped_backup = backup_cars.groupby("callsign")

        quarto_string += "\n\n### Backup Cars\n\n"

        for callsign, car_group in grouped_backup:
            model = car_group.iloc[0]["model"]
            group_id = car_group.iloc[0]["callsign_group"]

            heli_group = helicopters[helicopters["callsign_group"] == group_id]

            # Compare car and heli rotas for group
            identical = (
                car_group[
                    [
                        "summer_start",
                        "summer_end",
                        "winter_start",
                        "winter_end",
                        "category",
                    ]
                ]
                .reset_index(drop=True)
                .equals(
                    heli_group[
                        [
                            "summer_start",
                            "summer_end",
                            "winter_start",
                            "winter_end",
                            "category",
                        ]
                    ].reset_index(drop=True)
                )
            )

            if identical:
                message = f"The backup car {callsign} in group {group_id} has the same rota as the helicopter.\n\n"
                quarto_string += message
                st.caption(message)
            else:
                message = f"The backup car {callsign} in group {group_id} has a different rota to the helicopter.\n\n"
                quarto_string += message
                st.caption(message)
                header = f"#### {callsign} is a {model} and\n"
                body = ""
                for _, row in car_group.iterrows():
                    summer = f"\n- runs a {row['category']} service from {to_military_time(row['summer_start'])} to {to_military_time(row['summer_end'])} in summer"
                    winter = f"\n- runs a {row['category']} service from {to_military_time(row['winter_start'])} to {to_military_time(row['winter_end'])} in winter"
                    body += f"    {summer}\n    {winter}\n\n"
                result = header + body + "\n"
                quarto_string += result
                st.caption(result)

        # Extra (non-backup) cars
        extra_cars = rota[
            rota["callsign_group"].isin(extra_cars_only)
            & (rota["vehicle_type"] != "helicopter")
        ]
        grouped_extra = extra_cars.groupby("callsign")

        num_cars_string = (
            f"Number of **Extra** (non-backup) Cars: {st.session_state.num_cars}"
        )
        quarto_string += "\n\n### " + num_cars_string + "\n\n"
        st.write(f"### {num_cars_string}")

        for callsign, group in grouped_extra:
            model = group.iloc[0]["model"]
            header = f"#### {callsign} is a {model} and\n"
            body = ""
            for _, row in group.iterrows():
                summer = f"\n- runs a {row['category']} service from {to_military_time(row['summer_start'])} to {to_military_time(row['summer_end'])} in summer"
                winter = f"\n- runs a {row['category']} service from {to_military_time(row['winter_start'])} to {to_military_time(row['winter_end'])} in winter"
                body += f"    {summer}\n    {winter}\n\n"
            result = header + body + "\n"
            quarto_string += result
            st.caption(result)

        if st.session_state.demand_adjust_type == "Overall Demand Adjustment":
            if st.session_state.overall_demand_mult == 100:
                demand_adjustment_string = "Demand is based on historically observed demand with no adjustments."
            elif st.session_state.overall_demand_mult < 100:
                demand_adjustment_string = f"Modelled demand is {100 - st.session_state.overall_demand_mult}% less than historically observed demand."
            elif st.session_state.overall_demand_mult > 100:
                demand_adjustment_string = f"Modelled demand is {st.session_state.overall_demand_mult - 100}% more than historically observed demand."

            st.write(demand_adjustment_string)

            quarto_string += "\n\n### Simulation Parameters\n\n"
            quarto_string += demand_adjustment_string
            quarto_string += "\n\n"

        # TODO: Add this in if we decide seasonal demand adjustment is a thing that's wanted
        elif st.session_state.demand_adjust_type == "Per Season Demand Adjustment":
            pass

        elif st.session_state.demand_adjust_type == "Per AMPDS Code Demand Adjustment":
            pass

        else:
            st.error(
                "TELL A DEVELOPER: Check Conditional Code for demand modifier in model.py"
            )

        with stylable_container(
            css_styles="""
hr {
    border-color: #a6093d;
    background-color: #a6093d;
    color: #a6093d;
    height: 1px;
  }
""",
            key="hr",
        ):
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
