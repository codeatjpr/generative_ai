import base64
import json

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv(find_dotenv())

st.set_page_config(page_title="Medical Document Classifier", page_icon="🏥", layout="wide")
st.title("🏥 Medical Document Classifier")

llm = ChatOpenAI(model="gpt-5.4-mini")

system_prompt = SystemMessage(
    content="""You are an expert medical document classifier.

Classify the uploaded document into exactly one of these categories:
- prescription: Doctor handwritten or printed medicine orders, Rx forms, or doctor notes.
- lab_report: Blood tests, urine tests, CBC reports, thyroid panels, LFT/KFT reports, or diagnostic results.
- ct_scan: CT scan images or reports such as brain, chest, or abdominal CT scans.
- xray: X-ray images or reports such as chest X-rays, bone X-rays, or dental X-rays.
- bill: Hospital invoices, payment summaries, receipts, or hospital billing documents.
- others: Any document that does not clearly belong to the previous five categories.

Respond strictly in JSON format:
{"category": "<one exact category>", "confidence": <number from 0 to 1>}

Do not add markdown, explanation, or text outside the JSON object.
Use a lowercase category exactly as listed."""
)


def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")


def parse_result(content):
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )
        result = json.loads(cleaned)
        category = str(result.get("category", "others")).lower()
        confidence = float(result.get("confidence", 0))

        allowed_categories = {
            "prescription",
            "lab_report",
            "ct_scan",
            "xray",
            "bill",
            "others",
        }
        if category not in allowed_categories:
            category = "others"

        return {
            "category": category,
            "confidence": round(max(0.0, min(1.0, confidence)), 2),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


uploaded_file = st.file_uploader(
    "Upload a medical document image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded medical document", use_container_width=True)

if st.button("Classify document", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("Please upload a JPG, JPEG, or PNG image first.")
    else:
        image_base64 = encode_image(uploaded_file)
        image_mime = uploaded_file.type or "image/png"

        user_prompt = HumanMessage(
            content=[
                {"type": "text", "text": "Classify this medical document."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_mime};base64,{image_base64}",
                    },
                },
            ]
        )

        try:
            with st.spinner("Analyzing the medical document..."):
                response = llm.invoke([system_prompt, user_prompt])

            result = parse_result(response.content)
            if result is None:
                st.error("The model did not return valid JSON.")
                st.text(response.content)
            else:
                st.subheader("Classification Result")
                st.json(result)
        except Exception as exc:
            st.error(f"Classification failed: {exc}")
