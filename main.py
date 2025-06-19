from consts import *
from handle_inputs import *
from fonts import *

# streamlit run C:/technion/semester7/Demo/main.py
# streamlit run main.py --server.enableXsrfProtection false
st.set_page_config(layout="wide", page_title="SeerCuts", page_icon=":pencil2:")
#write_to_screen("SeerCuts: Find the best binning for your dataset", 20)
st.write("### SeerCuts: Discretize your data in a meaningful way")

#enlarge_sidebar_text()

st.sidebar.markdown('<div style="font-size: 18px; font-weight: bold;">Inputs <span style="font-size: 18px;">&#128221;</span></div>', unsafe_allow_html=True)
if 'clicked_point' not in st.session_state:
    st.session_state.clicked_point = False
if 'show_apply' not in st.session_state:
    st.session_state.show_apply = True
if 'show_binned_table' not in st.session_state:
    st.session_state.show_binned_table = False
df, dataset_json = handle_file_upload()
if df is not None:
    process_inputs(df, dataset_json)

