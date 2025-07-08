import os
from dotenv import load_dotenv

# ✅ LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ✅ HuggingFace transformers
from transformers import pipeline

# ✅ Load env vars
load_dotenv()

# ✅ Load the PDF
pdf_path = "data/Daily_Exercise_Routine.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# ✅ Split PDF into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# ✅ Use a free embedding model from HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Store in Chroma vector DB
vectorstore = Chroma.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever()

# ✅ Load small HF text2text QA model
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)

# ✅ Define a custom QA chain using LangChain + HuggingFace pipeline
from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ✅ RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ✅ Ask questions
while True:
    query = input("\nAsk something about the PDF (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    answer = qa_chain.invoke(query)
    print("\nAnswer:", answer)
