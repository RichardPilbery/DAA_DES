import streamlit as st
from _app_utils import get_text, get_text_sheet, DAA_COLORSCHEME
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">',
    unsafe_allow_html=True,
)

st.markdown(
    '<script src="https://kit.fontawesome.com/178c4e86b3.js" crossorigin="anonymous"></script>',
    unsafe_allow_html=True,
)

text_df=get_text_sheet("welcome")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title(get_text("page_title", text_df))

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

st.header("Primary Project Team")

col_h, col_r, col_s = st.columns(3)

with col_h:
    st.subheader("Hannah Trebilcock")
    st.markdown("*Devon Air Ambulance*")

    st.image("app/assets/ht.png", width=200)

    st.markdown("""
#### Project Areas

- Project management
- Input data analysis and wrangling
""")

with col_r:
    st.subheader("Richard Pilbery")
    st.markdown("*Yorkshire Ambulance Service NHS Trust*")

    st.image("app/assets/rp.jpg", width=200)


    st.markdown("""
#### Project Areas

- Simulation model coding
""")

    # st.image("assets/rp.png")
    st.markdown(
"""
Richard is a paramedic research fellow with over 20 years experience in the ambulance service.
"""

    )

    rp_col_1, rp_col_2, rp_col_3 = st.columns(3)

    with rp_col_1:
        with stylable_container(
        key="container_orcid",
        css_styles=r"""
            .stLinkButton p:before {
                font-family: 'Font Awesome 6 Brands';
                content: '\f8d2';
                display: inline-block;
                padding-right: 3px;
                vertical-align: middle;
                font-weight: 200;
            }
            """,
    ):
            st.link_button("ORCID", url="https://orcid.org/0000-0002-5797-9788")

    with rp_col_2:
        with stylable_container(
    key="container_github",
    css_styles=r"""
        .stLinkButton p:before {
            font-family: 'Font Awesome 6 Brands';
            content: '\f09b';
            display: inline-block;
            padding-right: 3px;
            vertical-align: middle;
            font-weight: 200;
        }
        """,
):
            st.link_button("Github", url="https://github.com/RichardPilbery")



with col_s:
    st.subheader("Sammi Rosser")
    st.markdown("*University of Exeter - PenCHORD Team*")

    st.image("app/assets/sr.jpg", width=200)

    st.markdown("""
#### Project Areas

- Web application development
- Data visualisation
- Additional model coding

""")

    st.markdown(
"""
Sammi works as part of the [Peninsula Collaborative for health Operational Research and Data Science (PenCHORD)](https://medicine.exeter.ac.uk/health-community/research/penchord/),
specialising in computer simulation and web application development. Sammi has previously worked in the NHS and now teaches advanced data analysis and modelling techniques
on the [Health Service Modelling Associates Programme](https://www.hsma.co.uk).
"""

    )

    sr_col_1, sr_col_2, sr_col_3 = st.columns(3)

    with sr_col_1:
        with stylable_container(
        key="container_orcid",
        css_styles=r"""
            .stLinkButton p:before {
                font-family: 'Font Awesome 6 Brands';
                content: '\f8d2';
                display: inline-block;
                padding-right: 3px;
                vertical-align: middle;
                font-weight: 200;
            }
            """,
    ):
            st.link_button("ORCID", url="https://orcid.org/0000-0002-9552-8988")

    with sr_col_2:
        with stylable_container(
        key="container_github",
        css_styles=r"""
            .stLinkButton p:before {
                font-family: 'Font Awesome 6 Brands';
                content: '\f09b';
                display: inline-block;
                padding-right: 3px;
                vertical-align: middle;
                font-weight: 200;
            }
            """,
    ):
            st.link_button("Github", url="https://github.com/Bergam0t")




# st.write(get_text("page_description", text_df))
