import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import IsolationForest
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
import matplotlib.pyplot as plt

# Required columns for processing
REQUIRED_COLUMNS = {
    "TransactionAmount": ["TransactionAmount", "Amount", "TxnAmount"],
    "TransactionDate": ["TransactionDate", "Date", "TxnDate"],
    "PreviousTransactionDate": ["PreviousTransactionDate", "PrevDate", "LastTxnDate"],
    "AccountBalance": ["AccountBalance", "Balance", "AcctBalance"],
}

OPTIONAL_COLUMNS = {
    "MerchantID": ["MerchantID", "Merchant", "Vendor"],
    "Latitude": ["Latitude", "Lat"],
    "Longitude": ["Longitude", "Lon", "Lng"],
}

# Automatically map columns based on the dataset
def auto_map_columns(data):
    column_mapping = {}
    missing_columns = []

    # Map required columns
    for key, possible_names in REQUIRED_COLUMNS.items():
        for name in possible_names:
            if name in data.columns:
                column_mapping[key] = name
                break
        else:
            missing_columns.append(key)

    # Map optional columns
    for key, possible_names in OPTIONAL_COLUMNS.items():
        for name in possible_names:
            if name in data.columns:
                column_mapping[key] = name
                break

    return column_mapping, missing_columns

# Preprocessing and anomaly detection using Isolation Forest
def preprocess_and_analyze(data, column_mapping, contamination=0.05):
    # Check for required columns
    required_columns = ["TransactionAmount", "TransactionDate", "PreviousTransactionDate", "AccountBalance"]
    for col in required_columns:
        if col not in column_mapping:
            st.error(f"Required column '{col}' not found in the dataset.")
            return None

    # Map columns
    amount_col = column_mapping["TransactionAmount"]
    date_col = column_mapping["TransactionDate"]
    prev_date_col = column_mapping["PreviousTransactionDate"]
    balance_col = column_mapping["AccountBalance"]

    # Convert dates to datetime
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data[prev_date_col] = pd.to_datetime(data[prev_date_col], errors='coerce')

    # Calculate time between transactions in minutes
    data['TimeBetweenTransactions'] = (data[date_col] - data[prev_date_col]).dt.total_seconds() / 60

    # Select only numeric columns for imputation
    numeric_columns = data.select_dtypes(include=['number']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    # Feature selection for anomaly detection
    features = [amount_col, 'TimeBetweenTransactions', balance_col]
    feature_data = data[features].fillna(0)

    # Apply Isolation Forest
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
    data['AnomalyScore'] = isolation_forest.fit_predict(feature_data)
    data['AnomalyFlag'] = data['AnomalyScore'].apply(lambda x: 1 if x == -1 else 0)

    return data

# Function to display dataset preview
def display_dataset_preview(data):
    st.header("Dataset Preview")
    options = ["Top 5 Rows", "Last 5 Rows", "Full Dataset", "Custom"]
    selected_option = st.radio("Select an option to view the dataset:", options)

    if selected_option == "Top 5 Rows":
        st.write(data.head())
    elif selected_option == "Last 5 Rows":
        st.write(data.tail())
    elif selected_option == "Full Dataset":
        st.write(data)
    elif selected_option == "Custom":
        # Input for number of rows
        num_rows = st.number_input("Enter the number of rows to display:", min_value=1, max_value=len(data), value=5)

        # Multi-select for column names
        selected_columns = st.multiselect(
            "Select columns to display:",
            options=data.columns.tolist(),
            default=data.columns.tolist()  # Default to all columns
        )

        # Display the dataset with custom rows and columns
        st.write(data.loc[:num_rows - 1, selected_columns])


# Function to display the statistical analysis table
def display_statistical_analysis(data, column_mapping):
    st.header("Statistical Analysis Table")
    amount_col = column_mapping["TransactionAmount"]

    # Calculate statistics
    stats = data[amount_col].describe().to_frame()
    stats.loc['median'] = data[amount_col].median()  # Add median separately
    stats = stats.rename(columns={amount_col: "Transaction Amount Statistics"})

    # Display table
    st.table(stats)

# Visualizations
def plot_transaction_distribution(data, amount_col):
    fig = px.histogram(data, x=amount_col, nbins=30, title="Transaction Amount Distribution", color_discrete_sequence=['blue'])
    st.plotly_chart(fig)
    st.write("""
    **Insights:**  
    This histogram shows the distribution of transaction amounts. Most transactions occur within a lower range, with a few high-value outliers indicating potential areas of interest.
    """)


def plot_time_series(data, date_col, amount_col):
    time_series = data.groupby(date_col)[amount_col].sum().reset_index()
    fig = px.line(time_series, x=date_col, y=amount_col, title="Time-Series of Transactions")
    st.plotly_chart(fig)
    st.write("""
    **Insights:**  
    This visualization highlights transaction patterns over time. Spikes may indicate periods of increased activity or special events.
    """)

def plot_correlation_heatmap(data):
    numeric_data = data.select_dtypes(include=['number'])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)
    st.write("""
    **Insights:**  
    The heatmap identifies correlations between numerical features. High correlations can be leveraged for predictive models or identifying influencing factors.
    """)

def plot_anomalies(data, amount_col):
    fig = px.scatter(data, x=amount_col, y='TimeBetweenTransactions', color='AnomalyFlag', title="Anomaly Detection (Isolation Forest)",
                     labels={amount_col: 'Transaction Amount', 'TimeBetweenTransactions': 'Time Between Transactions (minutes)'},
                     color_discrete_map={0: 'blue', 1: 'red'})
    st.plotly_chart(fig)


# Function to display anomaly transactions
def display_anomaly_table(data):
    st.subheader("Anomaly Transactions Table")
    anomalies = data[data['AnomalyFlag'] == 1]
    st.write(anomalies)

    # Download anomaly transactions as a CSV
    csv = anomalies.to_csv(index=False)
    st.download_button(
        label="Download Anomaly Transactions CSV",
        data=csv,
        file_name="anomaly_transactions.csv",
        mime="text/csv",
    )

# Function to create a high-resolution plot and save as an image
def save_high_res_plot(fig, filename):
    fig.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    return filename

# Function to create a professional PDF report
def create_pdf_report(data, column_mapping, metrics, visualizations):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Add title
    styles = getSampleStyleSheet()
    title = Paragraph("<b>Financial Activity Analysis Report</b>", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 20))

    # Add metrics
    elements.append(Paragraph("<b>Key Metrics:</b>", styles["Heading2"]))
    metrics_table_data = [[key, value] for key, value in metrics.items()]
    metrics_table = Table(metrics_table_data, colWidths=[200, 200])
    metrics_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ])
    )
    elements.append(metrics_table)
    elements.append(Spacer(1, 20))

    # Add visualizations
    elements.append(Paragraph("<b>Visualizations:</b>", styles["Heading2"]))

    for title, fig in visualizations.items():
        # Save high-resolution image
        plot_filename = f"/tmp/{title.replace(' ', '_')}.png"
        save_high_res_plot(fig, plot_filename)

        # Add section header
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        elements.append(Spacer(1, 10))

        # Add image
        elements.append(Image(plot_filename, width=400, height=300))
        elements.append(Spacer(1, 20))

    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Function to trigger PDF download in Streamlit
def download_pdf_report(data, column_mapping):
    metrics = {
        "Total Transactions": f"{data.shape[0]}",
        "Average Transaction Amount": f"${data[column_mapping['TransactionAmount']].mean():.2f}",
        "Anomalies Detected": f"{data['AnomalyFlag'].sum()}",
    }

    # Generate visualizations
    visualizations = {}

    # Transaction Distribution
    fig, ax = plt.subplots()
    sns.histplot(data[column_mapping["TransactionAmount"]], bins=30, kde=True, ax=ax)
    ax.set_title("Transaction Amount Distribution")
    visualizations["Transaction Amount Distribution"] = fig

    # Time-Series
    fig, ax = plt.subplots()
    time_series = data.groupby(column_mapping["TransactionDate"])[column_mapping["TransactionAmount"]].sum()
    time_series.plot(ax=ax)
    ax.set_title("Time-Series of Transactions")
    visualizations["Time-Series of Transactions"] = fig

    # Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_data = data.select_dtypes(include=['number'])
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    visualizations["Correlation Heatmap"] = fig

    # Anomalies
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=data,
        x=column_mapping["TransactionAmount"],
        y='TimeBetweenTransactions',
        hue='AnomalyFlag',
        palette={0: 'blue', 1: 'red'},
        ax=ax
    )
    ax.set_title("Anomaly Detection (Isolation Forest)")
    visualizations["Anomaly Detection"] = fig

    pdf_buffer = create_pdf_report(data, column_mapping, metrics, visualizations)
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="financial_analysis_report.pdf",
        mime="application/pdf"
    )
