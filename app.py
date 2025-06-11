import streamlit as st
import fitz  # PyMuPDF
import docx
import pandas as pd
import io
from transformers import pipeline

# Streamlit page config
st.set_page_config(page_title="Pre-Sales Assistant", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_model()

def extract_text(file):
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "".join(page.get_text() for page in doc)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(io.BytesIO(file.read()))
        return "\n".join(p.text for p in doc.paragraphs)
    elif file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        xls = pd.ExcelFile(file)
        return "\n\n".join(df.to_string(index=False) for df in [xls.parse(s) for s in xls.sheet_names])
    return ""

def split_chunks(text, max_len=1000):
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_len)
        last_dot = text[start:end].rfind(".")
        end = start + last_dot + 1 if last_dot != -1 else end
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

def summarize(text, fast=False):
    chunks = split_chunks(text)
    if fast:
        chunks = chunks[:2]
    summaries = []
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Summarizing part {i+1}/{len(chunks)}..."):
            try:
                out = summarizer(chunk, max_length=150, min_length=80, do_sample=False)
                summaries.append(out[0]['summary_text'])
            except:
                summaries.append("")
    return " ".join(summaries)

st.title("🧠 Pre-Sales Assistant")
uploaded = st.file_uploader("Upload document (PDF, DOCX, XLSX)", type=["pdf", "docx", "xls", "xlsx"])

if uploaded:
    text = extract_text(uploaded)
    if not text or len(text.strip()) < 50:
        st.warning("File too short or could not extract meaningful content.")
    else:
        if st.button("Generate Summary"):
            result = summarize(text, fast=True)
            st.subheader("📝 Summary")
            st.write(result)
else:
    st.info("📁 Upload a document to get started.")
