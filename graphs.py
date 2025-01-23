import time
import plotly.express as px
import pandas as pd
import streamlit as st


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
    fig = px.scatter(
        df,
        x="utility",
        y="semantic",
        color="binning_option",
        color_discrete_map=color_mapping,
        hover_data={"binning_option": True, "utility": True, "semantic": True}
    )

    # Update marker size and layout
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(
        title=title,
        xaxis_title="Utility",
        yaxis_title="Semantic",
        font=dict(size=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    if new_method_flag:
        status_text.text("")

    # Display elapsed time
    elapsed_time = time.time() - start_time
    st.write(f"We explored 100 candidates for finding the best strategy in {elapsed_time:.2f} seconds")


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
            sorted_binning_df = best_df.sort_values(by="utility", ascending=False)
        elif sort_order == "Semantic":
            sorted_binning_df = best_df.sort_values(by="semantic", ascending=False)

        sorted_binning_df['color'] = sorted_binning_df['binning_option'].map(color_mapping)

        # Build the HTML for the table
        table_html = '<table style="width:100%; border-collapse: collapse;">'
        table_html += "<thead><tr><th>Binning Option</th></tr></thead><tbody>"

        for _, row in sorted_binning_df.iterrows():
            # Get the color for the current binning_option
            color = color_mapping.get(row['binning_option'], 'gray')  # Default to 'gray' if color not found
            # Apply the color to the row
            table_html += f'<tr style="background-color:{color};"><td>{row["binning_option"]}</td></tr>'

        table_html += "</tbody></table>"

        # Display the table using st.markdown() with raw HTML
        st.markdown(table_html, unsafe_allow_html=True)

        # TODO: check how to remove the index
        # sorted_binning_df = sorted_binning_df[['binning_option']].reset_index(drop=True)
        # st.table(sorted_binning_df)
