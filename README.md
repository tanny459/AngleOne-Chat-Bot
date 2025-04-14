# AngleOne-Chat-Bot
AI-powered chatbot built with LangChain, FAISS, and Groq's LLaMA-3 to answer user queries based on AngelOne support documents and insurance PDFs. Uses RAG (Retrieval-Augmented Generation) to ensure accurate, context-aware responses.

# 🤖 AngelOne & Insurance RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot trained on AngelOne's support documents and insurance PDFs. It helps users get quick, accurate answers to their questions — and says **"I don't know"** if the answer doesn't exist in the knowledge base.

---

## 💡 Features

- ✅ Retrieval-Augmented Generation using FAISS & MiniLM
- 🤝 Integrated with Groq's LLaMA-3 for fast, intelligent responses
- 🧠 Chat history memory support using Streamlit session state
- 📃 Uses custom-cleaned Q&A extracted from `angelone_support_pages.txt`
- 🧼 "I don't know" fallback for out-of-scope questions
- 🌐 Web-based interface with Streamlit

---

## 🛠️ Tech Stack

- [Langchain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Groq API](https://console.groq.com/)
- [Streamlit](https://streamlit.io/)
- [HuggingFace MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/angelone-rag-chatbot.git
cd angelone-rag-chatbot
