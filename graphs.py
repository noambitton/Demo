import time
import plotly.express as px
import pandas as pd
from streamlit_plotly_events import plotly_events
from fonts import *



def show_histogram(df, clicked_data):
    # Display the clicked data
    if clicked_data[0]['x']>0.8:
    # Load your images (replace with your actual image paths or PIL images)
        image_1 = "data/age_binning12.png"
        image_2 = "data/BMI_binning12.png"
        image_3 = "data/glucose_binning12.png"
    else:
        image_1 = "data/age_binning22.png"
        image_2 = "data/BMI_binning22.png"
        image_3 = "data/glucose_binning22.png"


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
        hover_data={"ID": True, "Utility": True, "Semantic": True}
    )

    # Update marker size and layout
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(
        title=title,
        xaxis_title="Utility",
        yaxis_title="Semantic",
        font=dict(size=16),
        showlegend=False,
        xaxis=dict(
            range=[0, 1.1],  # Set the range from 0 to 100
            dtick=0.2  # Set the interval between ticks to 10
        )
    )
    clicked_point = plotly_events(fig, click_event=True, select_event=False)

    # Display clicked data if available
    if clicked_point:
        st.session_state.clicked_point = True

        point = clicked_point[0]  # Access the first clicked point
        # write_to_screen(f"You clicked on: Utility = {point['x']}, Semantic = {point['y']}", 30)
        show_histogram(df, clicked_point)
    # Display elapsed time
    elapsed_time = time.time() - start_time
    write_to_screen(f"We explored 100 candidates for finding the best strategy in {elapsed_time:.2f} seconds", 22)


def display_graph(selected_method, best_binning_df_naive, best_binning_df_seercuts, col, new_method_flag, color_mapping_naive, color_mapping_seercuts):
    col1, col2 = col[0], col[1]
    with col1:
        if not st.session_state.show_binned_table:
            if selected_method == "Exhaustive":
                plot_graph(best_binning_df_naive, "Exhaustive Method: Utility vs Semantic", 5, new_method_flag, color_mapping_naive)
            elif selected_method == "SeerCuts":
                plot_graph(best_binning_df_seercuts, "SeerCuts: Utility vs Semantic", 0.5, new_method_flag, color_mapping_seercuts)
        else:
            df=pd.read_csv("data/diabetes_binned.csv")
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

        sorted_binning_df['color'] = sorted_binning_df['ID'].map(color_mapping)

        # Generate the HTML table with scrolling
        table_html = '<div style="max-height: 400px; overflow-y: auto;">'  # Scrolling container
        table_html += '<table style="width:100%; border-collapse: collapse;">'
        table_html += "<thead><tr><th>ID</th><th>Semantic</th><th>Utility</th></tr></thead><tbody>"

        for idx, row in sorted_binning_df.iterrows():
            color = color_mapping.get(row['ID'], 'gray')  # Default to 'gray' if not found
            table_html += f'<tr style="background-color:{color};">'
            table_html += f'<td>{row["ID"]}</td><td>{row["Semantic"]:.2f}</td><td>{row["Utility"]:.2f}</td></tr>'

        table_html += "</tbody></table></div>"  # Close the scrollable container

        # Display table with scrolling
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds two line breaks

        if st.session_state.clicked_point:
            # Step 1: Show the Apply button only if it hasn't been clicked yet
            if st.session_state.show_apply:
                st.button("Apply & Inspect", key="apply_button", on_click=on_apply_click)  # Add unique key to prevent multiple clicks

            # Step 2: If Apply is clicked, hide it and show other buttons
            else:
                col3, col4 = st.columns(2)
                with col3:
                    st.button("Return", key="return_button", on_click=on_return_click)









