from consts import *
from handle_inputs import *
from fonts import *

# streamlit run C:/technion/semester7/Demo/main.py
st.set_page_config(layout="wide", page_title="SeerCuts", page_icon=":pencil2:")
write_to_screen("SeerCuts: Find the best binning for your dataset", 40)

enlarge_sidebar_text()

st.sidebar.markdown('<div style="font-size: 30px; font-weight: bold;">Inputs <span style="font-size: 30px;">&#128221;</span></div>', unsafe_allow_html=True)
st.session_state.clicked_point = False

df = handle_file_upload()
if df is not None:
    process_inputs(df, binning_df)

