import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

from _app_utils import get_text, get_text_sheet

text_df = get_text_sheet("glossary")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title(get_text("page_title", text_df))

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

st.caption(get_text("page_description", text_df))

glossary = pd.read_csv("app/assets/glossary.csv", sep=",")

# glossary["Definition"] = glossary["Definition"].str.wrap(90)


glossary["Term"] = glossary["Term"].apply(lambda x: f"<b>{x}</b>")

glossary["Term"] = glossary["Term"].str.wrap(30)

# glossary = glossary.set_index("Term")

glossary = glossary.applymap(lambda x: x.replace("\n", "<br>"))

# Show as a static table
st.markdown(
    glossary.to_html(escape=False, index=False, justify="left"), unsafe_allow_html=True
)

# print(glossary)

# st.table(
#    glossary,
#     # use_container_width=True,
#     # hide_index=True,
# )
