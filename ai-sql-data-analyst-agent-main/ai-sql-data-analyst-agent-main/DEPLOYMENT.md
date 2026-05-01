# Deployment Notes

This project is Streamlit Cloud ready.

## Build

Streamlit Cloud installs dependencies from `requirements.txt` automatically.

## App Entry Point

```text
app.py
```

## Required Secret

```toml
GROQ_API_KEY = "your_key_here"
```

## Local Command

```powershell
streamlit run app.py
```
