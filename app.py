import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

embedder = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

research_papers = [
    "Paper 1 text: Natural language processing (NLP) has been gaining traction in various fields...",
    "Paper 2 text: Deep learning has revolutionized computer vision by providing powerful tools for image analysis...",
    "Paper 3 text: Quantum computing is an emerging field that promises to solve complex problems faster than classical computers...",
    # Add more papers here
]

paper_embeddings = embedder.encode(research_papers, convert_to_tensor=False)
paper_embeddings = np.array(paper_embeddings)
paper_embeddings = paper_embeddings / np.linalg.norm(paper_embeddings, axis=1, keepdims=True)

embedding_dim = paper_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(paper_embeddings)

def search_and_summarize(query, top_k=3):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    relevant_papers = [research_papers[idx] for idx in indices[0]]
    
    summaries = []
    for paper in relevant_papers:
        summary = summarizer(paper, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    import word_tokenize
from nltk.corpus import stopwords
import string

#nltk.download('punkt')
#nltk.download('stopwords')

openai.api_key = "API"
# Function to get embeddings from OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']

# Function to calculate keyword overlap (basic semantic relevance check)
def calcul
    return summaries

st.title("Research Paper Summarizer")
st.write("Enter your query to summarize relevant research papers.")

query = st.text_input("Query:", "What are the latest advancements in quantum computing?")

if st.button("Summarize"):
    if query:
        with st.spinner("Searching and summarizing..."):
            summaries = search_and_summarize(query)
            st.write("### Summaries:")
            for i, summary in enumerate(summaries):
                st.write(f"**Summary {i + 1}:** {summary}")
    else:
        st.warning("Please enter a query.")
