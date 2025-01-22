import time
import plotly.express as px
import pandas as pd
import streamlit as st


def plot_graph(df, title, delay, new_method_flag):
    start_time = time.time()

    progress_bar = st.progress(0)
    status_text = st.empty()
    if new_method_flag:
        status_text.text("We are searching the best strategy for you")
        for i in range(1, 101):
            progress_bar.progress(i)
            time.sleep(delay / 100)

    fig = px.scatter(
        df,
        x="utility",
        y="semantic",
        hover_data={"binning_option": True, "utility": True, "semantic": True}
    )

    fig.update_traces(marker=dict(color='blue', size=12))
    fig.update_layout(title=title, xaxis_title="Utility", yaxis_title="Semantic", font=dict(size=18))
    st.plotly_chart(fig)

    status_text.text("")
    # Display elapsed time
    elapsed_time = time.time() - start_time
    st.write(f"We explored 100 candidates for finding the best strategy in {elapsed_time:.2f} seconds")


def display_graph(selected_method, best_df, col, new_method_flag):
    col1, col2 = col[0], col[1]
    with col1:
        if selected_method == "Naive":
            plot_graph(best_df, "Naive Method: Utility vs Semantic", 5, new_method_flag)
        elif selected_method == "SeerCuts":
            plot_graph(best_df, "SeerCuts: Utility vs Semantic", 2, new_method_flag)


def display_table(sort_order, best_df, col):
    col1, col2 = col[0], col[1]
    with col2:
        if sort_order == "Utility":
            sorted_binning_df = best_df.sort_values(by="utility", ascending=False)
        elif sort_order == "Semantic":
            sorted_binning_df = best_df.sort_values(by="semantic", ascending=False)

        # TODO: check how to remove the index
        sorted_binning_df = sorted_binning_df[['binning_option']].reset_index(drop=True)
        st.table(sorted_binning_df)
