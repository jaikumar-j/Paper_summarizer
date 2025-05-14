import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import PyPDF2
import os

embedder = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def read_pdf(file):
    text = ""
    with open(file, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

st.title("Research Paper Summarizer")
st.write("Upload your research paper (PDF or text file) to summarize its content.")

uploaded_file = st.file_uploader("Choose a PDF or text file", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        paper_text = read_pdf(uploaded_file)
    else:
        paper_text = uploaded_file.read().decode("utf-8")
    st.write("### Extracted Text:")
    st.write(paper_text)

    paper_embeddings = embedder.encode([paper_text], convert_to_tensor=False)
    paper_embeddings = np.array(paper_embeddings)
    paper_embeddings = paper_embeddings / np.linalg.norm(paper_embeddings, axis=1, keepdims=True)

    embedding_dim = paper_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(paper_embeddings)

    query = st.text_input("Query:", "What are the main findings of the paper?")

    if st.button("Summarize"):
        if query:
            with st.spinner("Searching and summarizing..."):
                query_embedding = embedder.encode([query], convert_to_tensor=False)
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

                distances, indices = faiss_index.search(query_embedding, 1)

                relevant_paper = paper_text  
                summary = summarizer(relevant_paper, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                
                st.write("### Summary:")
                st.write(summary)
        else:
            st.warning("Please enter a query.")
