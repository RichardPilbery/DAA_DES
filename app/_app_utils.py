from streamlit_extras.stylable_container import stylable_container
import streamlit as st
import pandas as pd
import os
import subprocess
import platform
from datetime import datetime


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
    except:
        print("Other unspecified error when running quarto check")

@st.fragment
def generate_quarto_report(run_quarto_check=False):
    """
    Passed an empty placeholder, put in a download button or a disabled download
    button in the event of failure
    """
    output_dir = os.path.join(os.getcwd(),'app/outputs')
    qmd_filename = 'app/air_ambulance_simulation_output.qmd'
    qmd_path = os.path.join(os.getcwd(),qmd_filename)
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
    except:
        ## error message
        print(f"Report cannot be generated")

    if os.path.exists(dest_html_path):
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
        print(f"Report failed to generate\n\n_{result}_")

        st.button(
                label="Error Generating Downloadable Report",
                disabled=True
            )

        st.warning("""It has not been possible to generate a downloadable copy of the simulation outputs.
                Please speak to a developer""")

        return "failure"
