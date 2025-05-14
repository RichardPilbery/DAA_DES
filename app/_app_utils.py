from streamlit_extras.stylable_container import stylable_container
import streamlit as st
import pandas as pd
import os
import subprocess
import platform
from datetime import datetime

def format_sigfigs(x, sigfigs=4):
    from math import log10, floor

    try:
        if x == 0:
            return "0.0"
        elif x < 1e-10:
            return "<1e-10"
        else:
            digits = sigfigs - 1 - floor(log10(abs(x)))
            return f"{x:.{digits}f}"
    except (ValueError, TypeError):
        return str(x)

def iconMetricContainer(key,icon_unicode,css_style=None,icon_color='grey', family="filled", type="icons"):
    """Function that returns a CSS styled container for adding a Material Icon to a Streamlit st.metric value

    CREDIT for starter version of this code: https://discuss.streamlit.io/t/adding-an-icon-to-a-st-metric-easily/59140?u=sammi1

    Args:
        key (str): Unique key for the component
        iconUnicode (str): Code point for a Material Icon, you can find them here https://fonts.google.com/icons. Sample \e8b6
        css_style(str, optional): Additional CSS to apply
        icon_color (str, optional): HTML Hex color value for the icon. Defaults to 'grey'.
        family(str, optional): "filled" or "outline". Only works with type = "icons"
        type(str, optional): "icons" or "symbols"

    Returns:
        DeltaGenerator: A container object. Elements can be added to this container using either the 'with'
        notation or by calling methods directly on the returned object.
    """

    if (family == "filled") and (type=="icons"):
        font_family = "Material Icons"
    elif (family == "outline") and (type=="icons"):
        font_family = "Material Icons Outlined"
    # elif (family == "filled") and (type=="symbols"):
    #     font_family = "Material Symbols"
    elif type=="symbols":
        font_family = "Material Symbols Outlined"
    else:
        print("ERROR - Check Params for iconMetricContainer")
        font_family = "Material Icons"

    css_style_icon=f'''
                    div[data-testid="stMetricValue"]>div::before
                    {{
                        font-family: {font_family};
                        content: "\{icon_unicode}";
                        vertical-align: -20%;
                        color: {icon_color};
                    }}
                    '''

    if css_style is not None:
        css_style_icon += """

        """

        css_style_icon += css_style

    iconMetric=stylable_container(
                key=key,
                css_styles=css_style_icon
            )
    return iconMetric

def file_download_confirm():
    st.toast("File Downloaded", icon=":material/download:")

@st.cache_data
def get_text_sheet(sheet):
    return pd.read_excel("app/assets/text.xlsx", sheet_name=sheet, engine="calamine")

@st.cache_data
def get_text(reference, text_df):
    return text_df[text_df["reference"] == reference]['text'].values[0]

DAA_COLORSCHEME = {
    "red": "#D50032",
    "navy": "#00205B",
    "blue": "#1D428A",
    "teal": "#00B0B9",
    "lightblue": "#C0F0F2",
    "green": "#56E39F",
    "orange": "#FFA400",
    "yellow": "#F8C630",
    "darkgreen": "#264027",
    "verylightblue": "#D5F5F6",
    "lightgrey": "#CCCCCC",
    "darkgrey": "#4D4D4D",
    "charcoal": "#1F1F1F",
}

# 99th Percentile
def q99(x):
    return x.quantile(0.99)

# 95th Percentile
def q95(x):
    return x.quantile(0.95)

# 90th Percentile
def q90(x):
    return x.quantile(0.9)

# 10th Percentile
def q10(x):
    return x.quantile(0.1)

# 75th Percentile
def q75(x):
    return x.quantile(0.75)

# 25th Percentile
def q25(x):
    return x.quantile(0.25)

def to_military_time(hour: int) -> str:
    return f"{hour:02d}00"

@st.cache_data
def get_quarto(repo_name, quarto_version="1.5.57"):
    print(f"Output of platform.processor(): {platform.processor()}")
    print(f"type:  {type(platform.processor())}")
    print("Attempting to download Quarto")
    # Download Quarto
    os.system(f"wget https://github.com/quarto-dev/quarto-cli/releases/download/v{quarto_version}/quarto-{quarto_version}-linux-amd64.tar.gz")

    # Create directory and extract Quarto
    os.system(f"tar -xvzf quarto-{quarto_version}-linux-amd64.tar.gz")
    # Check the contents of the folder we are in
    os.system("pwd")

    # # Ensure PATH is updated in the current Python process
    # Check current path
    os.system("echo $PATH")
    # Create a folder and symlink quarto to that location
    os.system(f"mkdir -p /mount/src/{repo_name}/local/bin")
    os.system(f"ln -s /mount/src/{repo_name}/quarto-{quarto_version}/bin/quarto /mount/src/{repo_name}/local/bin")
    # Update path
    os.system(f"echo 'export PATH=$PATH:/mount/src/{repo_name}/local/bin' >> ~/.bashrc")
    os.system('source /etc/bash.bashrc')
    # alternative method for good measure
    os.environ['PATH'] = f"/mount/src/{repo_name}/local/bin:{os.environ['PATH']}"

    # ensure path updates have propagated through
    print(os.environ['PATH'])
    # Install jupyter even if not in requirements
    os.system("python3 -m pip install jupyter")
    # Install second copy of requirements (so accessible by Quarto - can't access packages
    # that are installed as part of community cloud instance setup process)
    os.system(f"python3 -m pip install -r /mount/src/{repo_name}/requirements.txt")

    print("Trying to run 'quarto check' command")
    try:
        os.system("quarto check")
        result = subprocess.run(['quarto', 'check'], capture_output=True, text=True, shell=True)
        print(result.stdout)
        print(result.stderr)
        print("Quarto check run")
    except PermissionError:
        print("Permission error encountered when running 'quarto check'")
    except: # noqa
        print("Other unspecified error when running quarto check")

@st.fragment
def generate_quarto_report(run_quarto_check=False):
    """
    Passed an empty placeholder, put in a download button or a disabled download
    button in the event of failure
    """
    print("Trying to generate a downloadable quarto report")
    output_dir = os.path.join(os.getcwd(),'app/outputs')
    qmd_filename = 'app/air_ambulance_simulation_output.qmd'
    qmd_path = os.path.join(os.getcwd(),qmd_filename)
    print(f"Trying to find quarto template in {qmd_path}")
    html_filename = os.path.basename(qmd_filename).replace('.qmd', '.html')
    # html_filename = f"simulation_output_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    # print(html_filename)
    # dest_html_path = os.path.join(output_dir,f"simulation_output_{datetime.now().strftime('%H-%m-%d_%H%M')}.html")
    dest_html_path = os.path.join(output_dir,html_filename)

    # print(dest_html_path)

    try:
        if run_quarto_check:
            print("Trying to run 'quarto check' command")
            subprocess.run(["quarto"
                        , "check"])

        print("Running Quarto Render Command")

        ## forces result to be html
        result = subprocess.run(["quarto"
                                , "render"
                                , qmd_path
                                , "--to"
                                , "html"
                                , "--output-dir"
                                , output_dir
                                # , "--output-file"
                                # , html_filename
                                ]
                                , capture_output=True
                                , text=True)

        print("Quarto Render Command run succesfully")
        print(f"Destination Path: {dest_html_path}")
    except: #noqa
        ## error message
        print("Report cannot be generated")

    if os.path.exists(dest_html_path):
        print(f"Destination file {dest_html_path} found in filesystem - obtaining for download")
        with open(dest_html_path, "r") as f:
            html_data = f.read()

        with stylable_container(key="report_dl_buttons",
            css_styles=f"""
                    button {{
                            background-color: {DAA_COLORSCHEME['green']};
                            color: white;
                            border-color: white;
                        }}
                        """
            ):
            st.download_button(
                    label="Download Report",
                    data=html_data,
                    file_name=html_filename,
                    mime="text/html"
                )

            return "success"
    else:
        ## error message
        print("Generated file found not in filesystem")
        try:
            print(f"Report failed to generate\n\n_{result}_")
        except UnboundLocalError:
            print("Report failed to generate")

        st.button(
                label="Error Generating Downloadable Report",
                disabled=True
            )

        st.warning("""It has not been possible to generate a downloadable copy of the simulation outputs.
                Please speak to a developer""")

        return "failure"


def summary_sidebar(quarto_string):
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
        quarto_string += "### "
        quarto_string += num_helos_string
        quarto_string += "\n\n"
        st.write(num_helos_string)


        rota = (
            pd.read_csv("actual_data/HEMS_ROTA.csv")
            .merge(
                        pd.read_csv("actual_data/callsign_registration_lookup.csv"),
                        on="callsign",
                        how="left"
                    )
            .merge(
               pd.read_csv("actual_data/service_schedules_by_model.csv"),
                on=["model","vehicle_type"],
                how="left"
            )
        )

        for idx, row in rota[rota["vehicle_type"]=="helicopter"].iterrows():
            helicopter_rota_string = f"""
{row["callsign"]} is an {row["model"]} and runs a {row["category"]} service
from {to_military_time(row["summer_start"])}
to {to_military_time(row["summer_end"])} in summer
and {to_military_time(row["winter_start"])}
to {to_military_time(row["winter_end"])} in winter.
"""
            # quarto_string += "🚁 "
            quarto_string += helicopter_rota_string
            st.caption(helicopter_rota_string)

        num_cars_string = f"Number of **Extra** (non-backup) Cars: {st.session_state.num_cars}"
        quarto_string += "\n\n### "
        quarto_string += num_cars_string
        quarto_string += "\n\n"
        st.write(num_cars_string)
        callsign_group_counts = rota['callsign_group'].value_counts().reset_index()
        backup_cars_only = list(callsign_group_counts[callsign_group_counts['count']==1]['callsign_group'].values)


        for idx, row in rota[rota["callsign_group"].isin(backup_cars_only)].iterrows():
            car_rota_string = f"""
{row["callsign"]} is a {row["model"]} and runs
from {to_military_time(row["summer_start"])}
to {to_military_time(row["summer_end"])} in summer
and {to_military_time(row["winter_start"])}
to {to_military_time(row["winter_end"])} in winter.
"""
            # quarto_string += "🚗 "
            quarto_string += car_rota_string
            st.caption(car_rota_string)


        if st.session_state.demand_adjust_type == "Overall Demand Adjustment":
            if st.session_state.overall_demand_mult == 100:
                demand_adjustment_string = "Demand is based on historically observed demand with no adjustments."
            elif st.session_state.overall_demand_mult < 100:
                demand_adjustment_string = f"Modelled demand is {100-st.session_state.overall_demand_mult}% less than historically observed demand."
            elif st.session_state.overall_demand_mult > 100:
                demand_adjustment_string = f"Modelled demand is {st.session_state.overall_demand_mult-100}% more than historically observed demand."

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
