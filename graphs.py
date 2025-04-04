# and enlarge the title to visually look as big as the “Utility” label of the scatter plot
# For the graphed histograms, please make the x and y-axis tick font bigger.
# Ideally, the same size as the x and y-axis tick font of the scatter plot


import time
import plotly.express as px
import pandas as pd
from streamlit_plotly_events import plotly_events
from fonts import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

COLOR_ON_COLUMN = "Estimated" #'ID'  # Column to map the color on


def show_histogram(matched_row, df, attribute_features, green_flag):
    # Create columns for displaying histograms
    cols = st.columns(len(attribute_features))

    for i, attribute in enumerate(attribute_features):
        if attribute not in df.columns:
            st.write(f"Attribute '{attribute}' not found in dataset.")
            continue

        if not pd.api.types.is_numeric_dtype(df[attribute]):
            continue

        partition_edges = matched_row.iloc[0]["Partition"][i]
        bins = np.array(partition_edges)
        attribute_data = df[attribute].dropna()
        bin_counts, bin_edges = np.histogram(attribute_data, bins=bins)
        bin_labels = [f"[{bin_edges[i]}, {bin_edges[i + 1]})" for i in range(len(bin_edges) - 1)]

        # Plot histogram
        plt.figure(figsize=(5, 3))
        # Plot histogram with matching colors
        if green_flag:
            hist_color = "#11c739"
        else:
            hist_color = "#cce8f7"
        sns.barplot(x=bin_labels, y=bin_counts, color=hist_color)
        plt.ylabel("Count")
        plt.title(f"{attribute}", fontsize=12)
        plt.xticks(rotation=45)
        plt.tick_params(axis='both', which='major', labelsize=16)

        # Show the histogram in the respective column
        with cols[i]:
            st.pyplot(plt)


def plot_graph(binning_df, df, title, delay, new_method_flag, color_mapping, attribute_features):
    start_time = time.time()

    if new_method_flag:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("We are searching the best strategy for you")
        time_past = 0
        for i in range(1, 101):
            status_text.text(f"{time_past} seconds")
            progress_bar.progress(i)
            time.sleep(delay / 100)
            time_past += delay / 100
        progress_bar.empty()
        status_text.text("")

    fig = px.scatter(
        binning_df,
        x="Utility",
        y="Semantic",
        color=COLOR_ON_COLUMN,
        color_discrete_map=color_mapping,
        hover_data={"ID": True, "Utility": True, "Semantic": True},
        labels={"Estimated": "Candidate Partition"},
    )

    # Update marker size and layout
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(
        title=title,
        title_font=dict(size=18),
        xaxis_title_font=dict(size=12),
        yaxis_title_font=dict(size=12),
        xaxis_title="Utility",
        yaxis_title="Semantic",
        font=dict(size=12),
        showlegend=True,
        xaxis=dict(
            range=[0, 1.1],  # Set the range from 0 to 100
            dtick=0.2  # Set the interval between ticks to 10
        )
    )

    legend_newnames = {'1':'Pareto', '0': 'Explored'}
    fig.for_each_trace(lambda t: t.update(name = legend_newnames[t.name],
                                      legendgroup = legend_newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, legend_newnames[t.name])
                                     )
                  )
    clicked_point = plotly_events(fig, click_event=True, select_event=False)

    if clicked_point:
        st.session_state.clicked_point = True
        point = clicked_point[0]  # First clicked point
        utility = point["x"]
        semantic = point["y"]

        # Find the corresponding ID in the dataframe
        matched_row = binning_df[(binning_df["Utility"] == utility) & (binning_df["Semantic"] == semantic)]

        if not matched_row.empty:
            green_flag = color_mapping.get(str(matched_row.iloc[0][COLOR_ON_COLUMN]),
                                           '') == "#11c739"  # Green color hex

            show_histogram(matched_row, df, attribute_features, green_flag)

    elapsed_time = time.time() - start_time
    #write_to_screen(f"We explored 100 candidates for finding the best strategy in {elapsed_time:.2f} seconds", 22)


def display_graph(selected_method, best_binning_df_naive, best_binning_df_seercuts, df, col, new_method_flag, color_mapping_naive, color_mapping_seercuts, attribute_features):
    col1, col2 = col[0], col[1]
    with col1:
        if not st.session_state.show_binned_table:
            if selected_method == "Exhaustive":
                plot_graph(best_binning_df_naive, df, "Exhaustive Method: Utility vs Semantic", 5, new_method_flag, color_mapping_naive, attribute_features)
            elif selected_method == "SeerCuts":
                plot_graph(best_binning_df_seercuts, df, "SeerCuts: Utility vs Semantic", 0.5, new_method_flag, color_mapping_seercuts, attribute_features)
        else:
            df=pd.read_csv("data/Inspection_table/diabetes_binned.csv")
            st.dataframe(df)


# Define callback functions
def on_apply_click():
    st.session_state.show_apply = False
    st.session_state.show_binned_table = True


def on_return_click():
    st.session_state.show_apply = True
    st.session_state.show_binned_table = False


def display_table(sort_order, selected_method, best_binning_df_naive, best_binning_df_seercuts, col, color_mapping_naive, color_mapping_seercuts):
    col1, col2 = col[0], col[1]
    with col2:
        if selected_method == "Exhaustive":
            best_df = best_binning_df_naive
            color_mapping = color_mapping_naive
        elif selected_method == "SeerCuts":
            best_df = best_binning_df_seercuts  # Fixed incorrect assignment
            color_mapping = color_mapping_seercuts

        # Sort based on selected criteria
        if sort_order == "Utility":
            sorted_binning_df = best_df.sort_values(by="Utility", ascending=False)
        elif sort_order == "Semantic":
            sorted_binning_df = best_df.sort_values(by="Semantic", ascending=False)

        # Add color column based on the color mapping
        sorted_binning_df['color'] = sorted_binning_df[COLOR_ON_COLUMN].map(color_mapping)
        #print(color_mapping)
        table_color_mapping = {'1': '#11c739', '0': '#cce8f7'}

        # Generate the HTML table with scrolling
        table_html = '<div style="max-height: 400px; overflow-y: auto;">'  # Scrolling container
        table_html += '<table style="width:100%; border-collapse: collapse;">'
        table_html += "<thead><tr><th>ID</th><th>Semantic</th><th>Utility</th></tr></thead><tbody>"

        for idx, row in sorted_binning_df.iterrows():
            color = table_color_mapping.get(row[COLOR_ON_COLUMN], 'white')  # Default to 'gray' if not found
            #color = 'gray'
            table_html += f'<tr style="background-color:{color};">'
            table_html += f'<td>{row["ID"]}</td><td>{row["Semantic"]:.2f}</td><td>{row["Utility"]:.2f}</td></tr>'

        table_html += "</tbody></table></div>"  # Close the scrollable container

        # Display table with scrolling
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds two line breaks

        if selected_method == "Exhaustive":
            #write_to_screen(f"We explored 72 out of 22,192 candidates and found the best partitions in 7.41 seconds", 18)
            write_to_screen(f"We explored 146 out of 146 candidates and found the best partitions in 11.45 seconds", 18)
        else:
            write_to_screen(f"We explored 72 out of 22,192 candidates and found the best partitions in 7.41 seconds", 18)
            #write_to_screen(f"We explored 28 out of 146 candidates and found the best partitions in 2.32 seconds", 18)
        
        if st.session_state.clicked_point:
            # Step 1: Show the Apply button only if it hasn't been clicked yet
            if st.session_state.show_apply:
                st.button("Apply & Inspect", key="apply_button", on_click=on_apply_click)  # Add unique key to prevent multiple clicks

            # Step 2: If Apply is clicked, hide it and show other buttons
            else:
                col3, col4 = st.columns(2)
                with col3:
                    st.button("Return", key="return_button", on_click=on_return_click)









