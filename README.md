# 🎓 LectureScribe

> **AI-Powered Lecture Summarizer & Intelligent PDF Q&A**  
> Transform lecture PDFs into interactive study assistants with RAG-based retrieval, instant summaries, and context-aware answers.

---

## ✨ Features

- **📄 Smart PDF Ingestion** — Upload lecture notes, research papers, or any PDF; auto-extracts text with layout preservation
- **🤖 AI Summarization** — Get concise, structured summaries powered by Groq's lightning-fast LLM inference
- **💬 RAG-Based Q&A** — Ask questions about your content and receive accurate, citation-backed answers
- **🔍 Semantic Search** — ChromaDB vector store enables precise retrieval of relevant passages
- **🧠 Concept Extraction** — Automatically identifies key terms, definitions, and relationships
- **⚡ Optimized Performance** — Lightweight embeddings (sentence-transformers) + cached inference for fast responses
- **🔒 Privacy-First** — All processing happens locally; your data never leaves your machine

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 (required for compatibility)
- Git
- A [Groq API key](https://console.groq.com) (free tier available)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Gagan-Veeravelly/Lecture-Scribe-Final.git
cd Lecture-Scribe-Final

# 2. Create and activate virtual environment
python3.11 -m venv venv311
source venv311/bin/activate  # macOS/Linux
# OR
venv311\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
mkdir -p .streamlit
echo 'GROQ_API_KEY = "gsk_your_actual_key_here"' > .streamlit/secrets.toml

# 5. Launch the app
streamlit run app.py
