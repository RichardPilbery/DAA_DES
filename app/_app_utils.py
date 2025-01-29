from streamlit_extras.stylable_container import stylable_container
import streamlit as st
## logic_diagram.py
import schemdraw
from schemdraw import flow
import streamlit as st

# Set default flowchart box fill colors
flow.Box.defaults['fill'] = '#eeffff'
flow.Start.defaults['fill'] = '#ffeeee'
flow.Decision.defaults['fill'] = '#ffffee'
flow.Circle.defaults['fill'] = '#eeeeee'


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

def create_logic_diagram(number_labels = False, session_data = None):
    """
    Credit to Dom Rowney

    https://github.com/DomRowney/Project_Toy_MECC/blob/main/streamlit_app/logic_diagram.py
    """
    ## create a drawing class
    with schemdraw.Drawing() as d:

        label_call = "Call Arrives \n at SWAST"

        call = flow.Circle(r=d.unit/2).label(label_call).drop("S")
        flow.Arrow().at(call.S).down(d.unit/2)

        ## Save the drawing to a temporary file
        img_path = "logic_diagram.png"
        d.save(img_path)
        return img_path
