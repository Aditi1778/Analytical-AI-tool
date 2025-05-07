
# Ai analytical tool using gradio and pandas 
import gradio as gr
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from datetime import datetime 
from urllib.parse import quote 

from dotenv import load_dotenv 

load_dotenv()


print("Starting the AI Auto Analytics Tool...")

# Function to connect to MySQL database and fetch data
def connect_to_mysql(host, user, password, database, query):
    try:
        # Create connection string
        encoded_password = quote(password)
        connection_string = f"mysql+mysqlconnector://{user}:{encoded_password}@{host}/{database}"
        engine = create_engine(connection_string)
        
        # Fetch data using the provided query
        df = pd.read_sql(query, engine)
        return df, "Successfully connected to MySQL database and fetched data."
    except Exception as e:
        return None, f"Error connecting to MySQL: {str(e)}"

# Function to read uploaded file (CSV or Excel)
def read_uploaded_file(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.name)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."
        return df, "Successfully loaded the uploaded file."
    except Exception as e:
        return None, f"Error reading file: {str(e)}"

# Function to perform basic data analysis
def perform_analysis(df):
    if df is None:
        return None, None, None, None, "No data to analyze."
    
    analysis_results = []
    
    # Basic statistics
    analysis_results.append("### Basic Statistics")
    analysis_results.append(df.describe().to_string())
    
    # Missing values
    analysis_results.append("\n### Missing Values")
    analysis_results.append(df.isnull().sum().to_string())
    
    # Data types
    analysis_results.append("\n### Data Types")
    analysis_results.append(df.dtypes.to_string())
    
    # Generate visualizations
    plots = []
    
    # Histogram for numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols[:3]:  # Limit to first 3 numerical columns
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        # Save plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plots.append(f'<img src="data:image/png;base64,{plot_data}" alt="{col} histogram">')
        plt.close()
    
    # Correlation heatmap for numerical columns
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[numerical_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        
        # Save plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plots.append(f'<img src="data:image/png;base64,{plot_data}" alt="Correlation heatmap">')
        plt.close()
    
    # Combine analysis results
    analysis_text = "\n".join(analysis_results)
    
    return analysis_text, plots, df.head().to_html(), df.shape, "Analysis completed successfully."

# Main function to handle input and trigger analysis
def analyze_data(input_type, host=None, user=None, password=None, database=None, query=None, file=None):
    df = None
    message = ""
    
    if input_type == "MySQL":
        if not all([host, user, password, database, query]):
            return None, None, None, None, "Please provide all MySQL connection details and query."
        df, message = connect_to_mysql(host, user, password, database, query)
    elif input_type == "File":
        if file is None:
            return None, None, None, None, "Please upload a file."
        df, message = read_uploaded_file(file)
    
    if df is None:
        return None, None, None, None, message
    
    # Perform analysis
    analysis_text, plots, data_preview, data_shape, analysis_message = perform_analysis(df)
    
    return analysis_text, plots, data_preview, data_shape, analysis_message

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Auto Analytics App")
    gr.Markdown("Upload a CSV/Excel file or connect to a MySQL database to analyze your data.")
    
    # Input type selection
    input_type = gr.Radio(choices=["MySQL", "File"], label="Select Input Type", value="File")
    
    # MySQL inputs
    with gr.Group(visible=False) as mysql_group:
        host = gr.Textbox(label="MySQL Host", placeholder="e.g., localhost")
        user = gr.Textbox(label="MySQL User", placeholder="e.g., root")
        password = gr.Textbox(label="MySQL Password", type="password")
        database = gr.Textbox(label="MySQL Database", placeholder="e.g., mydb")
        query = gr.Textbox(label="SQL Query", placeholder="e.g., SELECT * FROM mytable")
    
    # File upload
    with gr.Group(visible=True) as file_group:
        file = gr.File(label="Upload CSV or Excel File")
    
    # Button to trigger analysis
    analyze_button = gr.Button("Analyze Data")
    
    # Outputs
    analysis_output = gr.Textbox(label="Analysis Results", lines=20)
    plot_output = gr.HTML(label="Visualizations")
    data_preview = gr.HTML(label="Data Preview (First 5 Rows)")
    data_shape = gr.Textbox(label="Data Shape (Rows, Columns)")
    message = gr.Textbox(label="Status Message")
    
    # Dynamic visibility based on input type
    def toggle_inputs(input_type):
        return {
            mysql_group: gr.update(visible=input_type == "MySQL"),
            file_group: gr.update(visible=input_type == "File")
        }
    
    input_type.change(fn=toggle_inputs, inputs=input_type, outputs=[mysql_group, file_group])
    
    # Connect analyze button to function
    analyze_button.click(
        fn=analyze_data,
        inputs=[input_type, host, user, password, database, query, file],
        outputs=[analysis_output, plot_output, data_preview, data_shape, message]
    )

# Launch the app
demo.launch()