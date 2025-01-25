import streamlit as st


def write_to_screen(text, size):
    # Define a dynamic CSS style with the font size
    st.markdown(f"""
        <style>
            .dynamic-text {{
                font-size: {size}px !important;
                font-weight: bold;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Use the CSS class 'dynamic-text' in st.write to display the text
    st.write(f'<p class="dynamic-text">{text}</p>', unsafe_allow_html=True)


def write_sidebar_to_screen(text, size):
    # Define a dynamic CSS style for sidebar with the specified font size
    st.sidebar.markdown(f"""
        <style>
            .sidebar-text {{
                font-size: {size}px !important;
                font-weight: bold;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Use the dynamic CSS class 'sidebar-text' in st.sidebar.write to display the text
    st.sidebar.write(f'<p class="sidebar-text">{text}</p>', unsafe_allow_html=True)


def enlarge_file_uploader_label():
    st.sidebar.markdown("""
        <style>
            /* Increase font size for the file uploader label */
            .stFileUploader label {
                font-size: 22px !important;
                font-weight: bold !important;
            }

            /* Increase font size for the 'Drag and drop' and 'Browse files' button texts */
            .stFileUploader .css-1v3fvcr {
                font-size: 18px !important;
                font-weight: bold !important;
            }

            /* Target the 'browse files' button in the file uploader */
            .stFileUploader button {
                font-size: 18px !important;
                font-weight: bold !important;
            }
        </style>
    """, unsafe_allow_html=True)


def enlarge_sidebar_text():
    st.sidebar.markdown("""
        <style>
            /* Increase font size for the sidebar header */
            .sidebar .sidebar-content {
                font-size: 30px !important;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)


def enlarge_selectbox():
    st.markdown(
        """
        <style>
        div[data-testid="stSelectbox"] > div {
            font-size: 20px; 
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def enlarge_sidebar_widgets():
    st.markdown("""
        <style>
            /* Target the label of multiselect and selectbox in the sidebar */
            .css-1d391kg, .css-1yqipk0, .css-1v3nx6b, .stSelectbox, .stMultiselect, .stFileUploader {
                font-size: 22px !important;
                font-weight: bold !important;
            }

            /* Target the "Choose the attribute features" label */
            div.stSidebar > div > div > div.stMultiSelect > label {
                font-size: 22px !important; /* Increase font size for multiselect label */
                font-weight: bold !important; /* Make font bold */
            }

            /* Target the "Choose the outcome feature" label */
            div.stSidebar > div > div > div.stSelectbox > label {
                font-size: 22px !important;
                font-weight: bold !important;
            }

            /* Target file uploader in the sidebar */
            div.stSidebar > div > div > div.stFileUploader > label {
                font-size: 22px !important;
                font-weight: bold !important;
            }
        </style>
    """, unsafe_allow_html=True)
