# Medical Document Classifier

A Streamlit app that sends an uploaded medical document image to an OpenAI vision model and classifies it into one of:

- `prescription`
- `lab_report`
- `ct_scan`
- `xray`
- `bill`
- `others`

## Run

From the repository root:

```powershell
uv run streamlit run medical_document_classifier/app.py
```

Set `OPENAI_API_KEY` in the repository `.env` file before starting the app.
