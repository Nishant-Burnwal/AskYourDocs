import os
import tempfile
from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline

# Load .env for future HuggingFace tokens or keys (if needed)
load_dotenv()

# Streamlit app config
st.set_page_config(page_title="📄 Ask Your Documents", layout="wide")
st.title("📄 Ask Your Documents (HuggingFace + Streamlit Chat)")

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF(s)
uploaded_files = st.file_uploader("📤 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_documents = []

    for uploaded_file in uploaded_files:
        if uploaded_file.size == 0:
            continue  # skip empty files

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            all_documents.extend(loader.load())

    # Split PDFs into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_documents(all_documents)

    # HuggingFace sentence-transformer
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Chroma vector DB
    vectorstore = Chroma.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever()

    # Use small, fast text2text generation model
    hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # QA chain setup
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    # Ask user query
    query = st.chat_input("Ask a question about your PDFs...")

    if query:
        result = qa_chain.invoke(query)
        answer = result["result"]
        answer = " ".join(dict.fromkeys(answer.split()))  # Deduplicate repeated tokens
        st.session_state.chat_history.append((query, answer))

    # Display chat
    for i, (question, response) in enumerate(st.session_state.chat_history):
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="📘"):
            st.markdown(response)
