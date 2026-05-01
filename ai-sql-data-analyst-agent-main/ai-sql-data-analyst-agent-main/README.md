# AI SQL Data Analyst Agent

A Streamlit app for uploading a CSV file, converting it into an in-memory SQLite table, asking natural-language questions, and receiving a SQL query, result table, final answer, and optional visualization.

## Features

- CSV upload and schema preview
- SQLite in-memory database creation
- Groq LLM SQL generation through LangChain
- Read-only SELECT query validation
- Query result table
- LLM-generated explanation
- Plotly bar, line, or scatter visualization

## Local Setup

```powershell
cd D:\ai-sql-data-analyst-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edit `.streamlit/secrets.toml` and add your Groq API key:

```toml
GROQ_API_KEY = "your_key_here"
```

Run the app:

```powershell
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this folder to its own GitHub repository.
2. Open Streamlit Community Cloud.
3. Create a new app and select this repository.
4. Set the main file path to `app.py`.
5. Add `GROQ_API_KEY` in Streamlit Cloud app secrets.
6. Deploy.

## Assignment Mapping

- Frontend: Streamlit
- LLM: Groq Llama 3
- Framework: LangChain
- Database: SQLite
- Data Handling: Pandas
- Visualization: Plotly
