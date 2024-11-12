import pandas as pd
import streamlit as st
from my_module import (
    auto_map_columns,
    preprocess_and_analyze,
    display_dataset_preview,
    display_statistical_analysis,
    plot_transaction_distribution,
    plot_time_series,
    plot_correlation_heatmap,
    plot_anomalies,
    display_anomaly_table,
    download_pdf_report,
)

def main():
    st.title("Financial Activity Dashboard")
    st.sidebar.title("Upload and Analyze")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Bank Transaction CSV File", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        column_mapping, missing_columns = auto_map_columns(data)

        if missing_columns:
            st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
        else:
            st.sidebar.success("Dataset Uploaded and Columns Mapped Successfully!")

            # Preprocess and analyze
            st.header("Key Metrics")
            data = preprocess_and_analyze(data, column_mapping)
            if data is not None:
                # Calculate metrics
                total_rows, total_columns = data.shape
                total_missing_values = data.isnull().sum().sum()
                avg_transaction = data[column_mapping['TransactionAmount']].mean()
                anomalies_detected = data['AnomalyFlag'].sum()

                # Display metrics
                st.metric(label="Total Transactions (Rows)", value=f"{total_rows}")
                st.metric(label="Total Features (Columns)", value=f"{total_columns}")
                st.metric(label="Total Missing Values", value=f"{total_missing_values}")
                st.metric(label="Average Transaction Amount", value=f"${avg_transaction:.2f}")
                st.metric(label="Anomalies Detected", value=f"{anomalies_detected}")

                # Display Dataset Preview
                display_dataset_preview(data)


                # Display Statistical Analysis Table
                display_statistical_analysis(data, column_mapping)

                # Visualizations
                st.header("Visualizations")
                st.subheader("Transaction Amount Distribution")
                plot_transaction_distribution(data, column_mapping["TransactionAmount"])

                st.subheader("Time-Series of Transactions")
                plot_time_series(data, column_mapping["TransactionDate"], column_mapping["TransactionAmount"])

                st.subheader("Correlation Heatmap")
                plot_correlation_heatmap(data)

                st.header("Anomalies Detected")
                st.subheader("Anomaly Detection")
                plot_anomalies(data, column_mapping["TransactionAmount"])

                # Display anomaly transactions table
                display_anomaly_table(data)

                # Download PDF report
                st.sidebar.subheader("Generate Report")
                download_pdf_report(data, column_mapping)

if __name__ == "__main__":
    main()
