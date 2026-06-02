LectureScribe – AI-Powered Lecture Assistant

LectureScribe is an AI-powered learning platform that helps students transform lecture materials into structured study resources. The system processes lecture PDFs and transcripts, generates summaries, creates knowledge representations, stores information in a vector database, and enables intelligent question-answering over lecture content.

What the Project Does
1. Lecture Summarization
Extracts content from lecture PDFs.
Uses Large Language Models (LLMs) to generate:
Key concepts
Important definitions
Concise summaries
Exam preparation questions
2. Knowledge Retrieval
Splits lecture content into chunks.
Converts text into embeddings using Sentence Transformers.
Stores embeddings in ChromaDB vector database.
Enables semantic search over lecture materials.
3. AI-Powered Q&A
Allows users to ask questions about lecture content.
Retrieves the most relevant chunks using vector similarity search.
Supports Retrieval-Augmented Generation (RAG)-style workflows.
4. Knowledge Graph Generation
Extracts relationships between concepts.
Visualizes connections through an interactive knowledge graph.
Helps students understand topic dependencies and concept relationships.
5. Interactive Learning Interface
Built using Streamlit.
Provides a modern web-based interface.
Includes visualization and study-assistance features.
AI Concepts Used

This project demonstrates knowledge of:

Large Language Models (LLMs)
Llama 3.2
Groq API integration
Retrieval-Augmented Generation (RAG)
Document chunking
Embedding generation
Semantic retrieval
Context-aware responses
Vector Databases
ChromaDB
Embedding Models
Sentence Transformers
all-MiniLM-L6-v2
Natural Language Processing (NLP)
Text summarization
Information extraction
Semantic search
Knowledge Graphs
Entity relationship extraction
Graph visualization
Technologies Used
Python
Streamlit
ChromaDB
Sentence Transformers
LangChain
Groq API
Llama 3.2
Plotly
Graph Visualization Libraries
