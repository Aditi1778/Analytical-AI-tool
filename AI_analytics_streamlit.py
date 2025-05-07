import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from groq import Groq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import streamlit as st

class AnalyticsAssistant:

    def __init__(self, model_provider='openai'):
       self.model_provider = model_provider
       self.data = None
       self.file_path = None
       self.file_type = None
       self.setup_llm()

    def setup_llm(self):
        if self.model_provider == 'openai':
            global OPENAI_API_KEY
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)
        elif self.model_provider == 'groq':
            global GROQ_API_KEY
            api_key = GROQ_API_KEY
            if not api_key:
               raise ValueError("GROQ_API_KEY not found in environment variables")
            self.llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=api_key)
        elif self.model_provider == 'google':
            global GEMINI_API_KEY
            api_key = GEMINI_API_KEY
            if not api_key:
               raise ValueError("GEMINI_API_KEY not found in environment variables")
            self.llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.0-flash", api_key=api_key)
        elif self.model_provider == 'openrouter':
            global OPENROUTER_API_KEY
            api_key = OPENROUTER_API_KEY
            if not api_key:
               raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            self.llm = ChatOpenAI(
                temperature=0,
                model="openai/gpt-4.1",  # Specify GPT-4.1 model
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1")  # OpenRouter's base URL
    def load_data(self, file_path):
        # if not file_path:
        #     raise ValueError("File path not provided")
        self.file_path = file_path
        self.file_type = file_path.split('.')[-1].lower()

        if self.file_type == 'csv':
            self.data = pd.read_csv(file_path)
        elif self.file_type in ['xls', 'xlsx']:
            self.data = pd.read_excel(file_path)
        elif self.file_type == 'json':
            self.data = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

        print(f"Loaded data with {self.data.shape[0]} rows and {self.data.shape[1]} columns")
        return self.data.head()

    def create_agent(self):
        """Create a Langchain agent for the loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        return create_pandas_dataframe_agent(
            self.llm,
            self.data,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code = True
        )

    def analyze(self, query):

        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        agent = self.create_agent()
        response = agent.run(query)
        return response

    def visualize(self, query):
        """Generate a visualization based on a natural language query.

        Args:
            query: The natural language query to visualize the data
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        visualization_prompt = f"""
        Based on the user query: "{query}", generate Python code to create a high-quality visualization using Matplotlib and/or Seaborn.
        Follow these guidelines:

           1. **Visualization Type**: Choose the most suitable visualization type (e.g., bar, line, scatter, histogram, boxplot, heatmap, violin plot) based on the query and data characteristics (e.g., categorical, numerical, time-series).
           2. **Styling**:
              - Apply a professional Seaborn theme (e.g., 'darkgrid', 'whitegrid') or Matplotlib style (e.g., 'ggplot', 'seaborn').
              - Use a visually appealing color palette (e.g., Seaborn's 'deep', 'muted', or custom Matplotlib colormaps).
              - Incorporate patterns (e.g., hatches for bars) or gradients where appropriate.
           3. **Labels and Annotations**:
              - Include a clear, descriptive title reflecting the query.
              - Label x- and y-axes with meaningful names and units (infer from column names if needed).
              - Add a legend if multiple data series are plotted.
              - Use gridlines, annotations, or data labels for clarity where helpful.
           4. **Robustness**:
              - Handle missing or invalid data gracefully (e.g., drop NaNs or filter data).
              - Ensure the plot is readable (e.g., adjust font sizes, figure size, or rotation for labels).
           5. **Output**: Return only the Python code block, ready to execute, without explanations or comments.
           6. **Example Considerations**:
              - For bar plots, use distinct colors or hatches for categories.
              - For line plots, add markers or dashed lines for multiple series.
              - For heatmaps, include a colorbar and adjust cell annotations.
              - For boxplots or violin plots, highlight outliers or medians.

           The code should produce a complete, publication-quality visualization with appropriate styling and design.
           """
        agent = self.create_agent()
        code = agent.run(visualization_prompt)

        # Extract code block if it's wrapped in ```
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        # Log the generated code for debugging
        print("Generated Visualization Code:\n", code)

        # Basic syntax fixes (e.g., missing parentheses)
        if code.count('(') > code.count(')'):
            code += ')'
        if code.count('[') > code.count(']'):
            code += ']'

        # Execute the visualization code
        try:
            # Setup locals with the dataframe
            locals_dict = {'df': self.data, 'plt': plt, 'sns': sns, 'np': np, 'pd': pd}

            # Execute the code
            exec(code, globals(), locals_dict)

            # Save the figure to a file
            plt.savefig('visualization.png')
            print("Visualization saved to 'visualization.png'")

            # Display the plot
            plt.show()

            return "Visualization completed successfully."
        except Exception as e:
            return f"Error generating visualization: {str(e)}"

    def get_data_summary(self):
        """Get a summary of the loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        summary = {"shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "numeric_summary": self.data.describe().to_dict()
        }
        return summary
# Streamlit web app
def run_streamlit_app():
    """Run a Streamlit web app for the Analytics Assistant."""
    st.title("Analytics Assistant")
    st.subheader("Analyze and visualize data using natural language")

    # Sidebar for model selection
    with st.sidebar:
        st.header("Settings")
        model_provider = st.selectbox(
            "Select Model Provider",
            options=["openai", "groq", "anthropic"],
            index=0
        )

    # Initialize the assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = AnalyticsAssistant(model_provider=model_provider)

    # File upload
    uploaded_file = st.file_uploader("Upload a data file (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the data
        try:
            data_head = st.session_state.assistant.load_data(uploaded_file.name)
            st.write("Data Preview:")
            st.dataframe(data_head)

            # Data summary
            if st.checkbox("Show Data Summary"):
                summary = st.session_state.assistant.get_data_summary()
                st.write("Data Shape:", summary["shape"])
                st.write("Columns:", summary["columns"])
                st.write("Missing Values:")
                st.write(pd.Series(summary["missing_values"]))
                st.write("Numeric Summary:")
                st.dataframe(pd.DataFrame(summary["numeric_summary"]))
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    # Query input
    query = st.text_area("Enter your query", height=100)

    col1, col2 = st.columns(2)
    with col1:
        analyze_button = st.button("Analyze")
    with col2:
        visualize_button = st.button("Visualize")

    # Process query
    if st.session_state.assistant.data is not None:
        if analyze_button and query:
            with st.spinner("Analyzing data..."):
                try:
                    result = st.session_state.assistant.analyze(query)
                    st.subheader("Analysis Result")
                    st.write(result)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

        if visualize_button and query:
            with st.spinner("Generating visualization..."):
                try:
                    result = st.session_state.assistant.visualize(query)
                    st.subheader("Visualization")
                    if os.path.exists('visualization.png'):
                        st.image('visualization.png')
                    st.write(result)
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")

# Command-line interface
def run_cli():
    """Run a command-line interface for the Analytics Assistant."""
    print("Analytics Assistant CLI")
    print("=======================")

    # Select model provider
    print("\nSelect model provider:")
    print("1. OpenAI")
    print("2. groq")
    print("3. google")
    print("4. openrouter")
    provider_choice = input("Enter choice (1-4): ")

    providers = {
        "1": "openai",
        "2": "groq",
        "3": "google",
        "4": "openrouter"
    }

    model_provider = providers.get(provider_choice, "openai")

    # Initialize the assistant
    assistant = AnalyticsAssistant(model_provider=model_provider)

    # Load data
    file_path = input("\nEnter path to data file (CSV, Excel, JSON): ")
    try:
        data_head = assistant.load_data(file_path)
        print("\nData Preview:")
        print(data_head)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Interactive loop
    while True:
        print("\nWhat would you like to do?")
        print("1. Analyze data")
        print("2. Visualize data")
        print("3. Get data summary")
        print("4. Exit")

        choice = input("Enter choice (1-4): ")

        if choice == "1":
            query = input("\nEnter your analysis query: ")
            try:
                result = assistant.analyze(query)
                print("\nAnalysis Result:")
                print(result)
            except Exception as e:
                print(f"Error during analysis: {str(e)}")

        elif choice == "2":
            query = input("\nEnter your visualization query: ")
            try:
                result = assistant.visualize(query)
                print(result)
            except Exception as e:
                print(f"Error generating visualization: {str(e)}")

        elif choice == "3":
            try:
                summary = assistant.get_data_summary()
                print("\nData Summary:")
                print(f"Shape: {summary['shape']}")
                print(f"Columns: {summary['columns']}")
                print("\nMissing Values:")
                for col, count in summary['missing_values'].items():
                    print(f"{col}: {count}")
                print("\nNumeric Summary:")
                for col, stats in summary['numeric_summary'].items():
                    print(f"{col}: {stats}")
            except Exception as e:
                print(f"Error getting data summary: {str(e)}")

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")





if __name__ == "__main__":
    try:
        # This will only run when you use Streamlit
        run_streamlit_app()
    except ImportError:
        # Fallback for CLI interface
        print("Streamlit not detected or not running as web app.")
        print("Choose interface:")
        print("1. Command-line")
        print("2. Web app (Streamlit)")
        interface_choice = input("Enter choice (1-2): ")

        if interface_choice == "2":
            print("Starting Streamlit app...")
            print("Run the following command in your terminal:")
            print("streamlit run analytical_tools.py")
        else:
            run_cli()
