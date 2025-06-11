import streamlit as st
import fitz  # PyMuPDF
import docx
import pandas as pd
import io
import re
from transformers import pipeline

# Configure the Streamlit page
st.set_page_config(page_title="Pre-Sales Assistant", layout="centered")

# Load summarizer model only once
@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="google/pegasus-xsum")

summarizer = load_summarizer()

# Extract text from various supported file types
def extract_text(file):
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "".join(page.get_text() for page in doc)

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(io.BytesIO(file.read()))
        return "\n".join(para.text for para in doc.paragraphs)

    elif file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        xls = pd.ExcelFile(file)
        texts = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            texts.append(df.to_string(index=False))
        return "\n\n".join(texts)

    return ""

# Split text into reasonably sized chunks
def split_into_chunks(text, max_len=1200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        snippet = text[start:end]
        last_period = snippet.rfind('.')
        if last_period != -1:
            end = start + last_period + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

# Generate a summary with optional fast mode
def generate_summary(text, fast_mode=False):
    chunks = split_into_chunks(text)
    if not chunks:
        return "The document is empty or could not be processed."

    if fast_mode:
        chunks = chunks[:3]  # Only summarize first 3 chunks for speed

    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            with st.spinner(f"Summarizing part {i + 1} of {len(chunks)}..."):
                result = summarizer(chunk, max_length=150, min_length=80, do_sample=False)
                summaries.append(result[0]['summary_text'])
        except Exception:
            summaries.append("")

    combined = " ".join(summaries).strip()
    if not combined:
        return "Could not generate summary from document content."

    try:
        final = summarizer(combined, max_length=180, min_length=100, do_sample=False)
        return final[0]['summary_text']
    except Exception:
        return combined

# Simple keyword-based QA using original text chunks
def find_answer(question, text_chunks):
    question_words = set(re.findall(r'\w+', question.lower()))
    best_chunk = None
    max_matches = 0

    for chunk in text_chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        common_words = question_words.intersection(chunk_words)
        if len(common_words) > max_matches:
            max_matches = len(common_words)
            best_chunk = chunk

    return best_chunk or "Sorry, I couldn't find the answer in the document."

# UI layout starts here
st.title("🤖 Pre-Sales Assistant")

uploaded_file = st.file_uploader("📄 Upload a document (PDF, DOCX, XLSX)", type=['pdf', 'docx', 'xls', 'xlsx'])

if uploaded_file:
    full_text = extract_text(uploaded_file)

    if len(full_text.strip()) < 50:
        st.warning("⚠️ The document appears too short or empty to summarize.")
    else:
        fast_mode = st.checkbox("⚡ Enable Fast Summary Mode (Quick but Less Detailed)", value=True)

        if st.button("Generate Summary"):
            with st.spinner("⏳ Summarizing the document, please wait..."):
                summary = generate_summary(full_text, fast_mode)

            st.subheader("📝 Document Summary")
            st.write(summary)

            st.download_button(
                label="📥 Download Summary",
                data=summary,
                file_name="document_summary.txt",
                mime="text/plain"
            )

        text_chunks = split_into_chunks(full_text, max_len=1200)

        question = st.text_input("💬 Ask a question about the original document:")
        if st.button("Get Answer") and question.strip():
            answer = find_answer(question, text_chunks)
            st.subheader("🧠 Answer")
            st.write(answer)
else:
    st.info("📁 Please upload a document to begin.")
