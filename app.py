import streamlit as st
import pdfplumber
import docx
import pandas as pd
import io
import re
from transformers import pipeline

# --- Page Setup ---
st.set_page_config(page_title="Pre-Sales Assistant", layout="centered")

# --- Load Summarizer Model ---
@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # For better results, you can use: "facebook/bart-large-cnn"

summarizer = load_summarizer()

# --- Clean and Filter Junk Text ---
def clean_text(text):
    lines = text.splitlines()
    cleaned = []
    seen = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower() in seen:
            continue
        if any(bad in line.lower() for bad in ['cnn.com', 'gallery.com', 'ireport.com', 'visit', 'next week']):
            continue
        cleaned.append(line)
        seen.add(line.lower())
    return "\n".join(cleaned)

# --- Extract Text from Upload ---
def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
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

# --- Split Text into Manageable Chunks ---
def split_into_chunks(text, max_words=250):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = []

    for sentence in sentences:
        words = sentence.strip().split()
        if not words:
            continue
        if sum(len(s.split()) for s in current) + len(words) <= max_words:
            current.append(sentence.strip())
        else:
            chunks.append(" ".join(current).strip())
            current = [sentence.strip()]

    if current:
        chunks.append(" ".join(current).strip())

    return chunks

# --- Summarize Text Chunks ---
def generate_summary(text, fast_mode=False):
    chunks = split_into_chunks(text)
    if not chunks:
        return "The document is empty or could not be processed."

    if fast_mode and len(chunks) > 4:
        chunks = chunks[:4]

    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            with st.spinner(f"Summarizing part {i + 1} of {len(chunks)}..."):
                result = summarizer(
                    chunk,
                    max_length=200,
                    min_length=100,
                    do_sample=False
                )
                summaries.append(result[0]['summary_text'])
        except Exception:
            summaries.append("")

    combined = " ".join(summaries).strip()

    if len(combined.split()) < 300:
        return combined

    try:
        with st.spinner("🔄 Refining final summary..."):
            final = summarizer(
                combined,
                max_length=250,
                min_length=150,
                do_sample=False
            )
            return final[0]['summary_text']
    except Exception:
        return combined

# --- Simple Q&A via Keyword Matching ---
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

    return best_chunk or "❌ Sorry, I couldn't find the answer in the document."

# --- UI ---
st.title("🤖 Pre-Sales Assistant")

uploaded_file = st.file_uploader("📄 Upload a document (PDF, DOCX, XLSX)", type=['pdf', 'docx', 'xls', 'xlsx'])

if uploaded_file:
    raw_text = extract_text(uploaded_file)
    full_text = clean_text(raw_text)

    if len(full_text.strip()) < 50:
        st.warning("⚠️ The document appears too short or empty to summarize.")
    else:
        with st.expander("🔍 Preview Extracted Text"):
            st.text_area("Raw Extracted Text", full_text[:4000], height=300)

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

        text_chunks = split_into_chunks(full_text)

        question = st.text_input("💬 Ask a question about the original document:")
        if st.button("Get Answer") and question.strip():
            answer = find_answer(question, text_chunks)
            st.subheader("🧠 Answer")
            st.write(answer)
else:
    st.info("📁 Please upload a document to begin.")
