from __future__ import annotations

import re
import sqlite3
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

APP_TITLE = "AI SQL Data Analyst Agent"
TABLE_NAME = "uploaded_data"

st.set_page_config(page_title=APP_TITLE, page_icon=":bar_chart:", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1d2b3a 0, #0c1117 34%, #070b10 100%);
        color: #e7edf5;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111927 0%, #0a0f16 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    .hero {
        padding: 28px 30px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.18), rgba(15, 23, 42, 0.88));
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
        margin-bottom: 20px;
    }
    .hero h1 {
        color: #f8fafc;
        font-size: 2.05rem;
        margin: 0 0 8px;
        letter-spacing: 0;
    }
    .hero p {
        color: #b7c4d4;
        font-size: 1.02rem;
        margin: 0;
        max-width: 860px;
    }
    .metric-card {
        padding: 16px 18px;
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.72);
    }
    .metric-label {
        color: #91a3b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .section-note {
        color: #9fb0c5;
        font-size: 0.92rem;
        margin-top: -8px;
        margin-bottom: 14px;
    }
    div[data-testid="stFileUploader"] section {
        background: rgba(15, 23, 42, 0.72);
        border: 1px dashed rgba(56, 189, 248, 0.45);
        border-radius: 12px;
    }
    div[data-testid="stFileUploader"] section * {
        color: #dbeafe;
    }
    div[data-testid="stFileUploader"] button {
        color: #0f172a;
        background: #e0f2fe;
        border: 0;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def clean_column_name(name: object, index: int) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip().lower()).strip("_")
    if not cleaned:
        cleaned = f"column_{index + 1}"
    if cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def dedupe_columns(columns: Iterable[object]) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for index, column in enumerate(columns):
        base = clean_column_name(column, index)
        count = seen.get(base, 0)
        seen[base] = count + 1
        output.append(base if count == 0 else f"{base}_{count + 1}")
    return output


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df.columns = dedupe_columns(df.columns)
    return df


def dataframe_to_sqlite(df: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql(TABLE_NAME, conn, index=False, if_exists="replace")
    return conn


def schema_text(df: pd.DataFrame) -> str:
    lines = [f"Table name: {TABLE_NAME}", "Columns:"]
    for column, dtype in df.dtypes.items():
        sample_values = df[column].dropna().astype(str).head(3).tolist()
        sample = ", ".join(sample_values) if sample_values else "no non-null sample"
        lines.append(f"- {column} ({dtype}) examples: {sample}")
    return "\n".join(lines)


def missing_value_count(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null": [int(df[column].notna().sum()) for column in df.columns],
            "missing": [int(df[column].isna().sum()) for column in df.columns],
            "unique": [int(df[column].nunique(dropna=True)) for column in df.columns],
        }
    )
    return summary


def build_llm() -> ChatGroq | None:
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        return None

    return ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",  # ✅ use this
        temperature=0
    )

def strip_sql_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"(?is)\bselect\b.+", text)
    if match:
        text = match.group(0).strip()
    return text.rstrip(";")


def validate_select_sql(sql: str) -> tuple[bool, str]:
    normalized = re.sub(r"\s+", " ", sql.strip(), flags=re.MULTILINE).lower()
    if not normalized.startswith("select "):
        return False, "Only SELECT queries are allowed."
    blocked = [" insert ", " update ", " delete ", " drop ", " alter ", " create ", " replace ", " attach ", " pragma "]
    padded = f" {normalized} "
    if any(token in padded for token in blocked):
        return False, "The generated SQL contains a blocked database operation."
    if ";" in sql.strip().rstrip(";"):
        return False, "Only one SQL statement can be executed."
    if TABLE_NAME not in normalized:
        return False, f"The query must read from the {TABLE_NAME} table."
    return True, ""


def generate_sql(llm: ChatGroq, schema: str, question: str) -> str:
    messages = [
        SystemMessage(
            content=(
                "You are an expert SQLite analyst. Generate exactly one read-only SQLite SELECT query. "
                "Use only the table and columns in the schema. Return SQL only, without markdown."
            )
        ),
        HumanMessage(content=f"Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"),
    ]
    response = llm.invoke(messages)
    return strip_sql_fences(str(response.content))


def explain_answer(llm: ChatGroq, question: str, sql: str, result_df: pd.DataFrame) -> str:
    preview = result_df.head(20).to_csv(index=False)
    messages = [
        SystemMessage(content="Explain SQL query results in concise business-friendly language."),
        HumanMessage(
            content=(
                f"Question: {question}\nSQL: {sql}\nResult preview CSV:\n{preview}\n\n"
                "Write the final answer in 3-5 sentences."
            )
        ),
    ]
    response = llm.invoke(messages)
    return str(response.content).strip()


def render_chart(result_df: pd.DataFrame) -> None:
    if result_df.empty:
        st.info("No rows returned, so there is nothing to visualize.")
        return

    numeric_columns = result_df.select_dtypes(include="number").columns.tolist()
    non_numeric_columns = [column for column in result_df.columns if column not in numeric_columns]

    if not numeric_columns:
        st.info("The result has no numeric columns for a chart.")
        return

    with st.expander("Visualization", expanded=True):
        chart_type = st.selectbox("Chart type", ["Bar", "Line", "Scatter"], index=0)
        y_axis = st.selectbox("Numeric value", numeric_columns)
        x_options = non_numeric_columns + [column for column in result_df.columns if column != y_axis]
        x_axis = st.selectbox("Category / x-axis", x_options, index=0 if x_options else None)

        if not x_axis:
            st.info("Add at least one non-value column to visualize this result.")
            return

        if chart_type == "Line":
            fig = px.line(result_df, x=x_axis, y=y_axis, markers=True)
        elif chart_type == "Scatter":
            fig = px.scatter(result_df, x=x_axis, y=y_axis)
        else:
            fig = px.bar(result_df, x=x_axis, y=y_axis)
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.62)",
            font_color="#dbeafe",
        )
        st.plotly_chart(fig, use_container_width=True)


st.markdown(
    """
    <div class="hero">
        <h1>AI SQL Data Analyst Agent</h1>
        <p>Upload real CSV data, convert it into a temporary SQLite workspace, ask business questions in plain English, and review the generated SQL before using the insight.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Analyst Console")
    if get_secret("GROQ_API_KEY"):
        st.success("Groq API key loaded")
    else:
        st.warning("Add GROQ_API_KEY in Streamlit secrets to enable AI SQL generation.")
    st.markdown("**SQLite table:** `uploaded_data`")
    st.markdown("**Query policy:** read-only `SELECT` statements")
    st.divider()
    st.markdown("**Good questions to try**")
    st.caption("Which segment has the highest revenue?")
    st.caption("Show monthly sales trend.")
    st.caption("Find the top 10 products by profit.")
    st.caption("Compare average values by category.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="Use a clean tabular CSV with headers in the first row.")

if not uploaded_file:
    st.info("Upload a CSV file to open the analyst workspace.")
    st.stop()

try:
    df = load_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

schema = schema_text(df)
conn = dataframe_to_sqlite(df)

metric_cols = st.columns(4)
metrics = [
    ("Rows", f"{len(df):,}"),
    ("Columns", f"{len(df.columns):,}"),
    ("Missing Cells", f"{missing_value_count(df):,}"),
    ("SQLite Table", TABLE_NAME),
]
for col, (label, value) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

preview_col, schema_col = st.columns([1.35, 1])
with preview_col:
    st.subheader("Data Preview")
    st.markdown('<div class="section-note">First 100 rows from the uploaded dataset.</div>', unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True)
with schema_col:
    st.subheader("Column Profile")
    st.markdown('<div class="section-note">Data types, completeness, and cardinality.</div>', unsafe_allow_html=True)
    st.dataframe(column_summary(df), use_container_width=True, hide_index=True)

with st.expander("SQLite schema prompt sent to the LLM"):
    st.code(schema, language="text")

question = st.text_area(
    "Ask a business question about this dataset",
    placeholder="Example: Which category has the highest total sales, and what is the total?",
    height=100,
)

run_col, policy_col = st.columns([0.35, 0.65])
with run_col:
    run_analysis = st.button("Generate SQL Insight", type="primary", disabled=not question.strip(), use_container_width=True)
with policy_col:
    st.caption("The app validates the model output and executes only one read-only SELECT query against the temporary SQLite database.")

if run_analysis:
    llm = build_llm()
    if llm is None:
        st.error("Missing GROQ_API_KEY. Add it to .streamlit/secrets.toml locally or Streamlit Cloud secrets.")
        st.stop()

    with st.spinner("Generating SQL and analyzing the data..."):
        try:
            sql = generate_sql(llm, schema, question.strip())
            is_valid, validation_error = validate_select_sql(sql)
            if not is_valid:
                st.error(validation_error)
                st.code(sql, language="sql")
                st.stop()

            result_df = pd.read_sql_query(sql, conn)
            answer = explain_answer(llm, question.strip(), sql, result_df)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    sql_col, answer_col = st.columns([1, 1])
    with sql_col:
        st.subheader("Generated SQL")
        st.code(sql, language="sql")
    with answer_col:
        st.subheader("Executive Answer")
        st.write(answer)

    st.subheader("Result")
    st.dataframe(result_df, use_container_width=True)

    render_chart(result_df)
