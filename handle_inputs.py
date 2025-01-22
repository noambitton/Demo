import streamlit as st
import pandas as pd
from graphs import *


def select_features_from_csv(df):
    st.sidebar.write("### Select Features")
    attribute_features = st.sidebar.multiselect("Choose the attribute features:", df.columns.tolist())
    outcome_feature = st.sidebar.selectbox("Choose the outcome feature:", ["Select Outcome"] + df.columns.tolist())
    return attribute_features, outcome_feature


def select_task_option():
    # Dropdown for Task selection
    task_option = st.sidebar.selectbox("Choose a Task",
                                       ["Select Task", "Visualizations", "Prediction", "Data Imputation"])
    return task_option


def handle_file_upload():
    uploaded = st.sidebar.file_uploader("#### Upload a dataset", type=["csv"])
    if uploaded:
        return pd.read_csv(uploaded)
    else:
        st.code("# Upload your dataset to start!", language='python')
        return None


def process_inputs(df, binning_df):
    # Using session_state to retain graph and sorting method choices
    if 'selected_graph' not in st.session_state:
        st.session_state.selected_graph = ""
    if 'selected_sorting' not in st.session_state:
        st.session_state.selected_sorting = ""

    attribute_features, outcome_feature = select_features_from_csv(df)
    task_option = select_task_option()
    col = st.columns([2, 1])
    col1, col2 = col[0], col[1]
    best_df = binning_df.iloc[:2]  # Sample data for testing
    new_graph_method_flag = False

    # Check if attributes, outcome, and task are selected
    if attribute_features and outcome_feature != "Select Outcome" and task_option != "Select Task":
        with col1:
            st.write("### Graphs")
            graph_method = st.selectbox("Select Graph Method", ["", "Naive", "SeerCuts"], index=["", "Naive", "SeerCuts"].index(st.session_state.selected_graph))

            # Update the session state with the selected graph
            if graph_method != st.session_state.selected_graph:
                st.session_state.selected_graph = graph_method
                new_graph_method_flag = True
                # display_graph(st.session_state.selected_graph, best_df, col)

        with col2:
            sorting_method = st.selectbox("Select Sorting Method", ["", "Utility", "Semantic"], index=["", "Utility", "Semantic"].index(st.session_state.selected_sorting))  # Keep previous choice

            # Update the session state with the selected sorting method
            if sorting_method != st.session_state.selected_sorting:
                st.session_state.selected_sorting = sorting_method
                # display_table(st.session_state.selected_sorting, best_df, col)
        if st.session_state.selected_graph:
            display_graph(st.session_state.selected_graph, best_df, col, new_graph_method_flag)

        if st.session_state.selected_sorting:
            display_table(st.session_state.selected_sorting, best_df, col)

    else:
        st.warning("Please select all required options (Attributes, Outcome, Task) to proceed.")

