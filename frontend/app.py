import streamlit as st
import requests
import plotly.graph_objects as go
import json

FASTAPI_URL = "http://127.0.0.1:8000/upload/"


st.title("A/B Testing Analysis Service")

# Load A/B test data
ab_data = st.file_uploader("Upload A/B Test Data (CSV format)", type=["csv"])

# Check if a file is uploaded
if ab_data:
    st.write("A/B Test Data Uploaded Successfully!")

    # Choose analysis method
    analysis_method = st.selectbox("Select Analysis Method", ["classic_bootstrap", "cuped_bootstrap", "cupac"])

    # Load pre-experiment data for CUPED if chosen
    pre_experiment_data = None
    if analysis_method in ["cuped_bootstrap", "cupac"]:
        pre_experiment_data = st.file_uploader("Upload Pre-Experiment Data (CSV format)", type=["csv"])

        if pre_experiment_data:
            st.write("Pre-Experiment Data Uploaded Successfully!")

    # Display the "Analyze" button based on the chosen method and the data uploaded
    if analysis_method == "classic_bootstrap" or (analysis_method in ["cuped_bootstrap", "cupac"] and pre_experiment_data):

        if st.button('Analyze'):
            files = {"file": ab_data.getvalue()}

            if pre_experiment_data:
                files["pre_experiment_file"] = pre_experiment_data.getvalue()

            data = {"method": analysis_method}
            response = requests.post(FASTAPI_URL, data=data, files=files)

            if response.status_code == 200:
                results = response.json()

                # Convert string representations back to dictionaries
                initial_data_distribution = json.loads(results["initial_data_distribution"])
                bootstrapped_differences = json.loads(results["bootstrapped_differences"])

                # Display the Results
                st.subheader("Analysis Results")
                st.write(f"P-value: {results['p_value']:.4f}")
                st.write(f"Confidence Interval: {results['confidence_interval']}")
                st.write(f"Metric Value (Control): {results['metric_value_control']:.4f}")
                st.write(f"Metric Value (Variant): {results['metric_value_variant']:.4f}")

                # Display the Plotly plots
                st.subheader("Initial Data Distribution")
                fig_1 = go.Figure(data=initial_data_distribution["data"], layout=initial_data_distribution["layout"])
                st.plotly_chart(fig_1)

                st.subheader("Bootstrapped Differences")
                fig_2 = go.Figure(data=bootstrapped_differences["data"], layout=bootstrapped_differences["layout"])
                st.plotly_chart(fig_2)


            else:
                st.write(f"Error in analysis: {response.text}")
