# GyroGlove Research Assistant – RAG-Based Chatbot using Gemini LLM 
A RAG-based chatbot that uses Gemini LLM and Astra DB to answer user queries from a research paper on the GyroGlove. It performs semantic search using Hugging Face embeddings, retrieves relevant chunks, and responds with context-aware answers via Streamlit UI — in both text and speech.

##  Overview

This project implements a **Retrieval Augmented Generation (RAG)** pipeline that allows users to interact with a research paper on the **GyroGlove** through a chatbot interface. By combining **semantic search**, **embeddings**, **vector database**, and the **Gemini LLM**, the chatbot can answer complex queries using contextual information extracted from the paper — with both **text and audio responses**.

---

##  Objective

To build a chatbot that can:
- Understand and search through a technical research paper on GyroGlove
- Retrieve contextually relevant information
- Generate human-like, accurate answers using an LLM (Gemini)
- Respond in **both text and audio**
- Provide a deployed, easy-to-use GUI with Streamlit

---

##  System Architecture

###  End-to-End Pipeline

[PDF of Research Paper]
↓
[Sentence Transformer (Hugging Face)]
↓ (Chunk + Embed)
[Astra DB Vector Store]
↓ (Query)
[User Input → Embedding → Semantic Search]
↓
[Retriever → Relevant Chunks]
↓
[Gemini LLM for Answer Generation]
↓
[Text + Audio Response → Streamlit UI]


---

##  Tech Stack

| Component | Technology |
|----------|-------------|
| Embedding Model | `all-MiniLM-L6-v2` (Hugging Face) |
| Vector DB | **Astra DB** |
| LLM | **Gemini (Google Generative AI)** |
| Frontend | **Streamlit** |
| Audio | `pyttsx3` / `gTTS` |
| File Handling | `PyMuPDF` or `pdfplumber` |

---

