import streamlit as st

st.title("Local Pre-Sales Assistant")
st.write("This is your bot UI.")

import streamlit as st
import re

def split_into_chunks(text, max_len=500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        snippet = text[start:end]
        last_period = snippet.rfind('.')
        if last_period != -1:
            end = start + last_period + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks

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

st.title("Local Pre-Sales Assistant")

pdf_text = st.text_area("Paste the extracted PDF text here:", height=300)

question = st.text_input("Ask a question:")

if st.button("Ask"):
    if not pdf_text.strip():
        st.warning("Please paste the PDF text first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        chunks = split_into_chunks(pdf_text)
        answer = find_answer(question, chunks)
        st.markdown("### Answer:")
        st.write(answer)
