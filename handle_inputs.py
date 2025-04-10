import streamlit as st
import pandas as pd
from graphs import *
from consts import *


def get_color_mapping(df):
    unique_options = df[COLOR_ON_COLUMN].astype(str).unique()
    colors = px.colors.qualitative.Plotly
    return {'1': '#11c739', '0': '#a1c0dc'}
    return {option: colors[i % len(colors)] for i, option in enumerate(unique_options)}


def select_features_from_csv(df):
    #enlarge_sidebar_widgets()
    write_sidebar_to_screen("Select Features", 18)
    #enlarge_selectbox()
    attribute_features = st.sidebar.multiselect("Choose the attribute features:", df.columns.tolist())

    outcome_feature = st.sidebar.selectbox("Choose the outcome feature:", ["Select Outcome"] + df.columns.tolist())
    return attribute_features, outcome_feature


def select_task_option():
   # enlarge_selectbox()
    task_option = st.sidebar.selectbox("Choose a Task",
                                       ["Select Task", "Visualizations", "Prediction", "Data Imputation"])
    return task_option


def handle_file_upload():
    #enlarge_file_uploader_label()
    uploaded = st.sidebar.file_uploader("#### Upload a dataset", type=["csv"])
    if uploaded:
        return pd.read_csv(uploaded)
    else:
        st.code("# Upload your dataset to start!", language='python')
        return None


def process_inputs(df):
    # Using session_state to retain graph and sorting method choices
    if 'selected_graph' not in st.session_state:
        st.session_state.selected_graph = ""
    if 'selected_sorting' not in st.session_state:
        st.session_state.selected_sorting = ""

    attribute_features, outcome_feature = select_features_from_csv(df)
    task_option = select_task_option()

    if task_option=="Visualizations":
        best_binning_df_naive = vis_naive_df
        best_binning_df_seercuts = vis_seercuts_df
    if task_option=="Prediction":
        best_binning_df_naive = pred_binning_df
        best_binning_df_seercuts = pred_binning_df
    col = st.columns([2, 1])
    col1, col2 = col[0], col[1]

    # Update the check to ensure both attributes, outcome, task, graph method, and sorting method are selected before displaying the content
    if attribute_features and outcome_feature != "Select Outcome" and task_option != "Select Task":
        best_binning_df_naive['ID'] = best_binning_df_naive['ID'].astype(str)
        best_binning_df_seercuts['ID'] = best_binning_df_seercuts['ID'].astype(str)
        new_graph_method_flag = False

        color_mapping_naive = get_color_mapping(best_binning_df_naive)
        color_mapping_seercuts = get_color_mapping(best_binning_df_seercuts)
        with col1:
            graph_method = st.selectbox("Select Graph Method", ["", "Exhaustive", "SeerCuts"],
                                        index=2 if not st.session_state.selected_graph else ["", "Exhaustive",
                                                                                             "SeerCuts"].index(
                                            st.session_state.selected_graph))

            # Update the session state with the selected graph
            if graph_method != st.session_state.selected_graph:
                st.session_state.selected_graph = graph_method
                new_graph_method_flag = True
                # display_graph(st.session_state.selected_graph, best_df, col)

        with col2:
           # enlarge_selectbox()
            sorting_method = st.selectbox("Select Sorting Method", ["", "Utility", "Semantic"],
                                          index=1 if not st.session_state.selected_sorting else ["", "Utility",
                                                                                                 "Semantic"].index(
                                              st.session_state.selected_sorting))  # Utility is default

            # Update the session state with the selected sorting method
            if sorting_method != st.session_state.selected_sorting:
                st.session_state.selected_sorting = sorting_method
                # display_table(st.session_state.selected_sorting, best_df, col)

        # Now only show the graph and table if both are selected
        if graph_method and sorting_method:
            if st.session_state.selected_graph:
                display_graph(st.session_state.selected_graph, best_binning_df_naive, best_binning_df_seercuts, df, col, new_graph_method_flag, color_mapping_naive, color_mapping_seercuts, attribute_features)

            if st.session_state.selected_sorting:
                display_table(st.session_state.selected_sorting, graph_method, best_binning_df_naive, best_binning_df_seercuts, col, color_mapping_naive, color_mapping_seercuts)

    else:
        st.warning("Please select all required options (Attributes, Outcome, Task, Graph, Sorting) to proceed.")
