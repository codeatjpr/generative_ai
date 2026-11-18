# Logistics Document Classifier

A Streamlit app that sends an uploaded logistics or transportation document image to an OpenAI vision model and classifies it into one of:

- `load_doc`
- `pod_doc`
- `odometer_reading`
- `driver_id`
- `invoice`
- `others`

## Run

From the repository root:

```powershell
uv run streamlit run logistics_document_classifier/app.py
```

Set `OPENAI_API_KEY` in the repository `.env` file before starting the app.
