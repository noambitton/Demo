import time
import plotly.express as px
import pandas as pd
from streamlit_plotly_events import plotly_events
from fonts import *



def show_histogram(df, clicked_data):
    st.session_state.clicked_point=True
    # Display the clicked data
    if clicked_data[0]['x']>0.5:
    # Load your images (replace with your actual image paths or PIL images)
        image_1 = "datasets/age_binning1.png"
        image_2 = "datasets/BMI_binning1.png"
        image_3 = "datasets/glucose_binning1.png"
    else:
        image_1 = "datasets/age_binning2.png"
        image_2 = "datasets/BMI_binning2.png"
        image_3 = "datasets/glucose_binning2.png"


    # Create three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image_1, use_container_width=True, width=400)

    with col2:
        st.image(image_2, use_container_width=True, width=400)

    with col3:
        st.image(image_3, use_container_width=True, width=400)





def plot_graph(df, title, delay, new_method_flag, color_mapping):
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
        # write_to_screen(f"You clicked on: Utility = {point['x']}, Semantic = {point['y']}", 30)
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


        table_html = '<table style="width:100%; border-collapse: collapse;">'
        table_html += "<thead><tr><th>ID</th><th>Semantic</th><th>Utility</th></tr></thead><tbody>"

        for idx, row in sorted_binning_df.iterrows():
            # Get the color for the current ID
            color = color_mapping.get(row['ID'], 'gray')  # Default to 'gray' if color not found
            # Apply the color to the row
            table_html += f'<tr style="background-color:{color};">'
            table_html += f'<td>{row["ID"]}</td><td>{row["Semantic"]:.2f}</td><td>{row["Utility"]:.2f}</td></tr>'

        table_html += "</tbody></table>"

        # Display the table using st.markdown() with raw HTML
        st.markdown(table_html, unsafe_allow_html=True)

        # Add download button

        if st.session_state.clicked_point:

            download_df = pd.DataFrame()

            # Convert to CSV and provide download
            csv = download_df.to_csv(index=False)
            st.download_button(
                label="Apply & Download",
                data=csv,
                file_name=f"applied_binning.csv",
                mime="text/csv"
            )







