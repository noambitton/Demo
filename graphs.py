import time
import plotly.express as px
import pandas as pd
import streamlit as st
from streamlit_plotly_events import plotly_events
from fonts import *

if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = None


def show_histogram(df, clicked_data):
    # TODO: add here the real histograms
    utility_values = df['Utility']
    fig_hist = px.histogram(utility_values, nbins=5, title="Histogram")
    st.plotly_chart(fig_hist)


def plot_graph(df, title, delay, new_method_flag, color_mapping):
    start_time = time.time()

    if new_method_flag:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("We are searching the best strategy for you")
        for i in range(1, 101):
            progress_bar.progress(i)
            time.sleep(delay / 100)
        progress_bar.empty()
        status_text.text("")

    fig = px.scatter(
        df,
        x="Utility",
        y="Semantic",
        color="ID",
        color_discrete_map=color_mapping,
        hover_data={"ID": False, "Utility": True, "Semantic": True}
    )

    # Update marker size and layout
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(
        title=title,
        xaxis_title="Utility",
        yaxis_title="Semantic",
        font=dict(size=16),
        showlegend=False
    )
    clicked_point = plotly_events(fig, click_event=True, select_event=False)

    # Display clicked data if available
    if clicked_point:
        point = clicked_point[0]  # Access the first clicked point
        write_to_screen(f"You clicked on: Utility = {point['x']}, Semantic = {point['y']}", 30)
        show_histogram(df, clicked_point)
    # Display elapsed time
    elapsed_time = time.time() - start_time
    write_to_screen(f"We explored 100 candidates for finding the best strategy in {elapsed_time:.2f} seconds", 22)


def display_graph(selected_method, best_df, col, new_method_flag, color_mapping):
    col1, col2 = col[0], col[1]
    with col1:
        if selected_method == "Naive":
            plot_graph(best_df, "Naive Method: Utility vs Semantic", 5, new_method_flag, color_mapping)
        elif selected_method == "SeerCuts":
            plot_graph(best_df, "SeerCuts: Utility vs Semantic", 0.5, new_method_flag, color_mapping)


def display_table(sort_order, best_df, col, color_mapping):
    col1, col2 = col[0], col[1]
    with col2:
        if sort_order == "Utility":
            sorted_binning_df = best_df.sort_values(by="Utility", ascending=False)
        elif sort_order == "Semantic":
            sorted_binning_df = best_df.sort_values(by="Semantic", ascending=False)

        sorted_binning_df['color'] = sorted_binning_df['ID'].map(color_mapping)

        # Build the HTML for the table
        table_html = '<table style="width:100%; border-collapse: collapse;">'
        table_html += "<thead><tr><th>ID</th><th>Semantic</th><th>Utility</th></tr></thead><tbody>"

        for _, row in sorted_binning_df.iterrows():
            # Get the color for the current ID
            color = color_mapping.get(row['ID'], 'gray')  # Default to 'gray' if color not found
            # Apply the color to the row
            table_html += f'<tr style="background-color:{color};"><td>{row["ID"]}</td><td>{row["Semantic"]:.2f}</td><td>{row["Utility"]:.2f}</td></tr>'

        table_html += "</tbody></table>"

        # Display the table using st.markdown() with raw HTML
        st.markdown(table_html, unsafe_allow_html=True)
