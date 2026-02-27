# 📊 Data Analysis Agent

The data analysis Agent built using the Agno Agent framework. This agent helps users analyze their data (CSV, Excel files) through natural language queries, powered by **OpenAI** and **Groq** language models and **DuckDB** for efficient data processing - making data analysis accessible to users regardless of their SQL expertise.

## ✨ Features

### 📤 File Upload Support
- Upload CSV and Excel files
- Automatic data type detection and schema inference
- Support for multiple file formats

### 💬 Natural Language Queries
- Convert natural language questions into SQL queries
- Get instant answers about your data
- No SQL knowledge required

### 🔍 Advanced Analysis
- Perform complex data aggregations
- Filter and sort data
- Generate statistical summaries
- Create data visualizations

### 🎯 Interactive UI
- User-friendly Streamlit interface
- Real-time query processing
- Clear result presentation
- **LLM Provider Selection**: Choose between OpenAI and Groq and dynamically select different models right from the sidebar!

## 🚀 How to Run

### Setup Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/Thilak30/Data-Analysis-Agent.git
   cd Data-Analysis-Agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Run the Application

```bash
python -m streamlit run Data_Analysis_agent.py
```

*(Note: We recommend running with `python -m streamlit` if `streamlit` is not directly available in your PATH).*

## 💡 Usage

1. Launch the application using the command above.
2. In the Streamlit sidebar settings, select your preferred provider (**OpenAI** or **Groq**).
3. Provide your corresponding API key and select a model from the dropdown.
4. Upload your CSV or Excel file through the Streamlit interface.
5. Ask questions about your data in natural language!
6. View the results and generated SQL querying instantly.
