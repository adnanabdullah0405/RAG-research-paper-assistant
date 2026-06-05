# 🔬 GyroGlove Research Assistant — Advanced RAG Chatbot

> **Advanced RAG pipeline** built with LangGraph — answers complex 
> queries about the GyroGlove tremor stabilization research paper 
> using semantic search, self-correcting generation loops, 
> conversation memory, and text-to-speech output.

![Gemini](https://img.shields.io/badge/Gemini_LLM-4285F4?style=flat-square&logo=google&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-000000?style=flat-square&logo=python&logoColor=white)
![AstraDB](https://img.shields.io/badge/AstraDB-000000?style=flat-square&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

## 📊 Key Capabilities

| Feature | Detail |
|---------|--------|
| Knowledge source | GyroGlove tremor stabilization research paper |
| RAG technique | Advanced — query transformation, document grading, self-correction |
| Response format | Text + Audio (TTS) |
| Memory | Conversation memory with update loop |
| Vector DB | AstraDB with HuggingFace embeddings |
| Interface | Streamlit UI |

---

## 🏗️ LangGraph Architecture

![System Architecture](architecture/system-architecture.png)

> **Self-correcting RAG loop** — if retrieved documents are 
> irrelevant or generation quality is poor, the system 
> automatically rewrites the query and re-retrieves. 
> Only finalizes when generation meets quality threshold.

---

## ⚡ Advanced RAG Techniques

This is not a basic RAG pipeline. It implements:

- **Query transformation** — rewrites user query for better retrieval before embedding
- **Document grading** — evaluates retrieved chunks for relevance before generation
- **Generation grading** — scores the LLM output for accuracy and completeness
- **Self-correction loop** — if generation fails quality check, rewrites query and re-retrieves automatically
- **Conversation memory** — updates memory after each interaction for context-aware responses
- **Text-to-speech output** — responds in both text and audio

---


## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Pipeline Orchestration | LangGraph |
| LLM | Gemini (Google Generative AI) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Database | AstraDB |
| Text-to-Speech | pyttsx3 / gTTS |
| PDF Processing | pdfplumber / PyMuPDF |
| Frontend | Streamlit |
| Language | Python |

---

## 📸 Demo

> 🎥 [Watch Demo](results/demo-recording.mp4)

---

## 📫 Contact

**Adnan Abdullah** — Agentic AI Engineer & AI Team Lead

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/adnan-abdullah-700899b)
[![Email](https://img.shields.io/badge/Gmail-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:muhammad.adnannust@gmail.com)

---

*Built with LangGraph + Gemini + AstraDB · Advanced RAG with 
self-correction loops · Text + Audio responses*
