import streamlit as st
from consts import *
from graphs import *
from handle_inputs import *

# streamlit run C:/technion/semester7/Demo/main.py
st.set_page_config(layout="wide", page_title="SeerCuts", page_icon=":pencil2:")
st.write("### SeerCuts: Find the best binning for your dataset")

st.sidebar.write("## Inputs :page_with_curl:")

df = handle_file_upload()
if df is not None:
    process_inputs(df, binning_df)

