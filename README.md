# 📄 Ask Your Documents (HuggingFace + Streamlit Chat)

A local chatbot app to interact with your **PDF documents** using **free HuggingFace models** — no OpenAI or external APIs needed. Upload PDFs, ask questions, and get answers in a clean chat UI powered by **LangChain**, **ChromaDB**, and **Streamlit**.

---

## 🚀 Features

- Chat with multiple PDF files
- Local embeddings + LLM (no API keys required)
- HuggingFace models (MiniLM + FLAN-T5-base)
- Fast, minimal RAM usage (runs on CPU)
- Chat-style UI with history
- Streamlit web interface

---

## 🔧 Installation

1. Clone the repo  
2. Install dependencies:

```bash
pip install -r requirements.txt
Run the app:

bash
Copy code
streamlit run streamlit_app.py 
📂 File Structure
bash
Copy code
mini_rag/
├── streamlit_app.py       # Main Streamlit app
├── requirements.txt       # All Python dependencies
├── .env                   # For future use (e.g., HuggingFace token)
└── data/                  # (Optional) Folder for sample PDFs
🧠 Models Used
Embeddings: sentence-transformers/all-MiniLM-L6-v2

LLM: google/flan-t5-base (CPU-friendly text generator)

✅ Example Queries
“Summarize the book.”

“What are the key takeaways?”

“List characters and their roles.”

“Who is the author?”

📌 Notes
Works fully offline after model download

Designed for 8GB–16GB RAM systems

Keeps recent chat history in session

No OpenAI required — 100% HuggingFace-powered

