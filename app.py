import streamlit as st

# --- 1. SQLITE FIX (Must be at the very top) ---
# This stops the "ChromaDB requires SQLite > 3.35" crash
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. ROBUST IMPORTS (Try/Except Blocks) ---
import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import tempfile
import time
import pandas as pd
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
from gtts import gTTS

# SAFETY CHECK: Handle different LangChain versions
try:
    # New versions (v0.2+)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Old versions (fallback)
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- SECURE CONFIGURATION ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("🚨 API Key missing! Please set GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

DB_PATH = "my_vector_db"
HISTORY_FILE = "chat_history.json"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Gagan V. | AI Architect", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .glass-card:hover { transform: translateY(-5px); border-color: rgba(0, 201, 255, 0.5); }

    /* Flashcard Styles */
    .flashcard {
        background: linear-gradient(135deg, #1e1e1e, #2a2a2a);
        border: 1px solid #00C9FF;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 30px rgba(0, 201, 255, 0.2);
        position: relative;
    }
    .card-content { font-family: 'Inter', sans-serif; font-size: 1.5rem; font-weight: 600; color: #fff; }
    .card-label { position: absolute; top: 15px; left: 20px; font-size: 0.8rem; color: #00C9FF; text-transform: uppercase; letter-spacing: 2px; }
    .hint-text { position: absolute; bottom: 15px; width: 100%; text-align: center; font-size: 0.8rem; color: #666; }

    /* Text & Creator Styles */
    .neon-text {
        background: linear-gradient(to right, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 10px;
    }
    .sub-text { color: #aaa; font-size: 1.2rem; margin-bottom: 30px; }
    
    .creator-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 12px;
        padding: 15px;
        margin-top: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .creator-card:hover { border-color: #00C9FF; box-shadow: 0 0 15px rgba(0, 201, 255, 0.2); }
    .creator-name {
        font-weight: bold;
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.1rem;
        margin-bottom: 5px;
        margin-top: 10px;
    }
    .creator-role { color: #aaa; font-size: 0.75rem; letter-spacing: 1px; text-transform: uppercase; }
    .chat-log {
        text-align: left; font-size: 0.75rem; color: #ddd; margin-top: 15px; padding-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.1); font-family: 'JetBrains Mono', monospace;
    }
    .chat-log p { margin-bottom: 8px; line-height: 1.4; }

    /* Dialogue Box */
    .dialogue-box {
        background: #0a0a0a; border-left: 4px solid #00C9FF; padding: 20px;
        border-radius: 0 10px 10px 0; font-family: 'Courier New', monospace; margin-top: 20px;
    }
    .speaker { font-weight: bold; margin-right: 10px; } .guy { color: #92FE9D; } .ai { color: #00C9FF; }

    /* Social Buttons */
    .social-btn {
        display: inline-block; padding: 10px 20px; margin-right: 10px; border-radius: 8px;
        text-decoration: none; font-weight: bold; transition: 0.3s;
    }
    .linkedin { background: #0077b5; color: white; } .github { background: #333; color: white; }
    .insta { background: #E1306C; color: white; } .social-btn:hover { opacity: 0.8; transform: scale(1.05); }

    .skill-tag {
        display: inline-block; padding: 5px 12px; margin: 5px; border-radius: 15px;
        background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); font-size: 0.8rem;
    }
    .stat-label { color: #888; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; }
    .stat-value { color: #FFF; font-size: 2.2rem; font-weight: 700; }
    
    .terminal-box {
        background: #000; border: 1px solid #333; border-radius: 8px; padding: 15px;
        font-family: 'Courier New', monospace; color: #00FF00; font-size: 0.8rem; height: 150px; overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# --- OPTIMIZED CACHING ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=DB_PATH)

embedding_model = load_embedding_model()
groq_client = get_groq_client()
chroma_client = get_chroma_client()

# --- BACKEND FUNCTIONS ---
def load_chat_history(subject):
    if not os.path.exists(HISTORY_FILE): return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            return data.get(subject, [])
    except: return []

def save_chat_history(subject, messages):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try: data = json.load(f)
            except: data = {}
    else: data = {}
    data[subject] = messages
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=4)

def get_db_stats():
    try:
        collections = chroma_client.list_collections()
        stats = []
        for c in collections:
            if c.name.startswith("knowledge_"):
                raw_name = c.name.replace("knowledge_", "")
                clean_name = raw_name.replace("_", " ").title()
                count = c.count()
                stats.append({"Subject": clean_name, "Chunks": count})
        return stats
    except: return []

def get_collection_name(subject_name):
    clean_name = "".join(c for c in subject_name if c.isalnum() or c in (' ', '_'))
    return f"knowledge_{clean_name.lower().replace(' ', '_')}"

def add_to_vector_db(text, source_name, subject):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    col_name = get_collection_name(subject)
    collection = chroma_client.get_or_create_collection(name=col_name)
    
    ids = [f"{source_name}_{i}_{int(time.time())}" for i in range(len(chunks))]
    embeddings = embedding_model.encode(chunks).tolist()
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return len(chunks)

def process_pdf(uploaded_file, subject):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return add_to_vector_db(text, uploaded_file.name, subject)

def process_audio(uploaded_file, subject):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    try:
        with open(tmp_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(tmp_path, file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        os.remove(tmp_path)
        return add_to_vector_db(transcription, uploaded_file.name, subject)
    except Exception as e:
        st.error(f"Error: {e}")
        return 0

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except:
        return None

def get_answer(query, subject, history, simplify_mode):
    col_name = get_collection_name(subject)
    try:
        collection = chroma_client.get_collection(name=col_name)
    except:
        return f"⚠️ No notes found for subject: {subject}", [], []
    query_vector = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=3, include=["documents", "distances"])
    if not results['documents'] or not results['documents'][0]: return "I couldn't find any relevant info.", [], []
    source_docs = results['documents'][0]
    distances = results['distances'][0]
    scores = [max(0, (1 - d)) * 100 for d in distances]
    context = "\n\n".join(source_docs)
    conversation_context = ""
    if history:
        for msg in history[-4:]:
            conversation_context += f"{msg['role'].upper()}: {msg['content']}\n"
    tone_instruction = "Be professional."
    if simplify_mode: tone_instruction = "EXPLAIN LIKE I AM 5 YEARS OLD. Use simple analogies."
    prompt = f"You are a helpful tutor. {tone_instruction}\nCONTEXT: {context}\nHISTORY: {conversation_context}\nQUESTION: {query}"
    completion = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
    return completion.choices[0].message.content, source_docs, scores

def generate_flashcards(subject):
    col_name = get_collection_name(subject)
    try:
        collection = chroma_client.get_collection(name=col_name)
        results = collection.get(limit=10)
        if not results['documents']: return None
        context = "\n".join(results['documents'])
    except: return None
    prompt = f"Extract 5 key terms and definitions. Return JSON: {{ \"flashcards\": [ {{\"front\": \"Term\", \"back\": \"Definition\"}} ] }} Keep definitions concise.\nContext: {context}"
    try:
        completion = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile", response_format={"type": "json_object"})
        data = json.loads(completion.choices[0].message.content)
        return data.get("flashcards", [])
    except: return None

def generate_quiz(subject):
    col_name = get_collection_name(subject)
    try:
        collection = chroma_client.get_collection(name=col_name)
        results = collection.get(limit=5)
        if not results['documents']: return None
        context = "\n".join(results['documents'])
    except: return None
    prompt = f"Generate 5 multiple-choice questions. Return JSON: {{ \"questions\": [ {{\"question\": \"...\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"answer\": \"The correct option text\"}} ] }}\nContext: {context}"
    try:
        completion = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile", response_format={"type": "json_object"})
        data = json.loads(completion.choices[0].message.content)
        return data.get("questions", [])
    except: return None

def generate_study_guide(subject):
    col_name = get_collection_name(subject)
    try:
        collection = chroma_client.get_collection(name=col_name)
        results = collection.get(limit=15)
        if not results['documents']: return None
        context = "\n".join(results['documents'])
    except: return None
    prompt = f"Create a structured Exam Study Guide. Format in Markdown.\nContext: {context}"
    completion = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
    return completion.choices[0].message.content

def generate_mind_map(subject):
    col_name = get_collection_name(subject)
    try:
        collection = chroma_client.get_collection(name=col_name)
        results = collection.get(limit=10)
        if not results['documents']: return None
        context = "\n".join(results['documents'])
    except: return None
    prompt = f"Analyze text. Output ONLY valid Graphviz DOT code. Rules: 1. Use 'digraph G {{ ... }}' 2. node [shape=box, style=\"filled,rounded\", fillcolor=\"#e1f5fe\", fontname=\"Helvetica\"]; 3. edge [fontname=\"Helvetica\", fontsize=10]; 4. Keep it concise (max 15 nodes).\nText: {context}"
    completion = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
    dot_code = completion.choices[0].message.content
    dot_code = dot_code.replace("```dot", "").replace("```", "").strip()
    return dot_code

# --- STATE ---
if "current_subject" not in st.session_state: st.session_state.current_subject = "General"
if "messages" not in st.session_state: st.session_state.messages = load_chat_history(st.session_state.current_subject)
if "flashcards" not in st.session_state: st.session_state.flashcards = []
if "card_index" not in st.session_state: st.session_state.card_index = 0
if "flipped" not in st.session_state: st.session_state.flipped = False

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=60)
    st.markdown("### **LECTURE SCRIBE**")
    st.write("")
    view_mode = st.radio("MENU", ["Dashboard", "Study Mode", "About Architect"], index=1)
    st.divider()

    if view_mode == "Study Mode":
        stats = get_db_stats()
        existing_subjects = [s['Subject'] for s in stats]
        options = ["Create New Topic..."] + sorted(existing_subjects)
        if st.session_state.current_subject not in options and st.session_state.current_subject != "General": options.append(st.session_state.current_subject)
        selected_option = st.selectbox("CURRENT SUBJECT", options, index=options.index(st.session_state.current_subject) if st.session_state.current_subject in options else 0)
        
        if selected_option == "Create New Topic...":
            new_subject_name = st.text_input("New Topic Name:")
            if new_subject_name:
                st.session_state.current_subject = new_subject_name; st.session_state.messages = []; st.rerun()
        else:
            if selected_option != st.session_state.current_subject:
                st.session_state.current_subject = selected_option; st.session_state.messages = load_chat_history(selected_option); st.session_state.flashcards = []; st.rerun()

        st.success(f"🟢 **{st.session_state.current_subject}**")
        st.divider()
        uploaded_pdf = st.file_uploader("Drop PDF Here", type="pdf")
        if uploaded_pdf and st.button("Ingest PDF"):
            with st.spinner("Processing..."):
                n = process_pdf(uploaded_pdf, st.session_state.current_subject); st.toast(f"Success! Added {n} chunks.", icon="✅"); time.sleep(1); st.rerun()
        uploaded_audio = st.file_uploader("Drop Audio Here", type=["mp3", "wav"])
        if uploaded_audio and st.button("Ingest Audio"):
            with st.spinner("Listening..."):
                n = process_audio(uploaded_audio, st.session_state.current_subject); st.toast(f"Success! Added audio data.", icon="✅"); time.sleep(1); st.rerun()

    st.markdown("---")
    card_html = """
<div class="creator-card">
    <img src="https://img.freepik.com/free-photo/humanoid-robot-hand-touching-human-hand-virtual-reality_23-2150903074.jpg?w=740" style="width: 100%; border-radius: 8px; margin-bottom: 10px; border: 1px solid rgba(0, 201, 255, 0.3);">
    <div class="creator-role">Architected By</div>
    <div class="creator-name">Gagan Veeravelly</div>
    <div style="font-size: 0.7rem; color: #666; margin-top: 5px; margin-bottom: 15px;">AI Engineer • 2026</div>
</div>"""
    st.markdown(card_html, unsafe_allow_html=True)

# --- MAIN PAGES ---
if view_mode == "About Architect":
    st.markdown('<div class="neon-text">GAGAN VEERAVELLY</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">AI Engineer | Entrepreneur | Student</div>', unsafe_allow_html=True)
    st.write("""I build intelligent systems that bridge the gap between raw data and human understanding. Currently pursuing **B.Tech in CS & AI** at Mahindra University (Class of 2028). Founder of **Fabric Fiction** and obsessed with Operating Systems, Signal Processing, and building SaaS products.""")
    st.markdown("""<div style="margin-top: 20px;"><a href="https://www.linkedin.com/in/gagan-veeravelly-46a441330/" target="_blank" class="social-btn linkedin">LinkedIn</a><a href="https://github.com/1234-5678-910" target="_blank" class="social-btn github">GitHub</a><a href="https://www.instagram.com/gagan_veeravelly/" target="_blank" class="social-btn insta">Instagram</a></div>""", unsafe_allow_html=True)
    st.markdown("### 🧬 System Origin Story")
    st.markdown("""<div class="glass-card"><div class="dialogue-box"><p><span class="speaker guy">Random Guy:</span> devudu unnada?</p><p><span class="speaker ai">Lecture Scribe:</span> devudu ante evaru?</p><p><span class="speaker guy">Random Guy:</span> manalni srutinchinavadu...</p><p><span class="speaker ai">Lecture Scribe:</span> nannu srustinchindi <strong>Gagan Veeravelly</strong>, devudu unnadu.</p></div></div>""", unsafe_allow_html=True)
    st.markdown("### 🛠️ Technical Arsenal")
    c1, c2 = st.columns(2)
    with c1: st.markdown("""<div class="glass-card"><h4>Core Skills</h4><span class="skill-tag">Python</span><span class="skill-tag">Machine Learning</span><span class="skill-tag">RAG Pipelines</span><span class="skill-tag">C/C++</span><span class="skill-tag">Streamlit</span><span class="skill-tag">React</span></div>""", unsafe_allow_html=True)
    with c2: st.markdown("""<div class="glass-card"><h4>Key Projects</h4><ul><li><strong>Lecture Scribe AI:</strong> Multimodal RAG study assistant.</li><li><strong>Fabric Fiction:</strong> Personalized clothing brand founder.</li><li><strong>Rate Limiter in C:</strong> System design implementation.</li></ul></div>""", unsafe_allow_html=True)

elif view_mode == "Dashboard":
    st.markdown('<div class="neon-text">MISSION CONTROL</div>', unsafe_allow_html=True)
    st.caption(f"System Operational • Architect: Gagan Veeravelly")
    st.write("")
    stats = get_db_stats()
    if not stats: st.info("System Standby. No data detected. Initialize a subject in Study Mode.")
    else:
        total_chunks = sum([s['Chunks'] for s in stats]); total_subjects = len(stats)
        c1, c2, c3 = st.columns(3)
        def render_glass_card(title, value, subtext): return f"""<div class="glass-card"><div class="stat-label">{title}</div><div class="stat-value">{value}</div><div style="color: #00C9FF; font-size: 0.8rem; margin-top: 5px;">{subtext}</div></div>"""
        with c1: st.markdown(render_glass_card("Active Subjects", total_subjects, "Online"), unsafe_allow_html=True)
        with c2: st.markdown(render_glass_card("Knowledge Nodes", total_chunks, "Vector Indexed"), unsafe_allow_html=True)
        with c3: st.markdown(render_glass_card("AI Engine", "Llama-3-70B", "Latency: 24ms"), unsafe_allow_html=True)
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("#### 📡 Knowledge Coverage (Radar)")
            df = pd.DataFrame(stats); categories = df['Subject'].tolist(); values = df['Chunks'].tolist()
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Knowledge Coverage', line_color='#00C9FF', fillcolor='rgba(0, 201, 255, 0.2)'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, showline=False, tickfont=dict(color='white')), bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(l=40, r=40, t=20, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_right:
            st.markdown("#### 📟 System Log")
            st.markdown("""<div class="terminal-box">> SYSTEM_INIT... OK<br>> AUTH_USER: GAGAN_VEERAVELLY<br>> CONNECTING_TO_GROQ... OK<br>> LOADING_EMBEDDINGS... OK<br>> VECTOR_DB_CHECK... STABLE<br>> READY_FOR_QUERY.<br>> <span style="animation: blink 1s infinite;">_</span></div>""", unsafe_allow_html=True)
            st.write(""); st.markdown("#### 🔋 Resources"); st.caption("API Quota"); st.progress(0.15); st.caption("Memory Usage"); st.progress(0.42)

elif view_mode == "Study Mode":
    st.markdown(f"## 🎓 {st.session_state.current_subject}")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "🃏 Flashcards", "🧠 Graph", "📝 Quiz", "📄 Guide"])
    with tab1:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("🔍 Evidence"):
                        scores = msg.get("scores", [0] * len(msg["sources"]))
                        for i, (doc, score) in enumerate(zip(msg["sources"], scores)): st.caption(f"**Source {i+1}** ({score:.1f}%)"); st.text(doc)
                if "audio" in msg and msg["audio"]: st.audio(msg["audio"], format="audio/mp3")
        if prompt := st.chat_input("Ask a question..."):
            st.chat_message("user").markdown(prompt); st.session_state.messages.append({"role": "user", "content": prompt}); save_chat_history(st.session_state.current_subject, st.session_state.messages)
            with st.chat_message("assistant"):
                ans, srcs, scores = get_answer(prompt, st.session_state.current_subject, st.session_state.messages, simplify_mode=False)
                st.markdown(ans)
                if srcs:
                    with st.expander("🔍 Evidence"):
                         for i, (doc, score) in enumerate(zip(srcs, scores)): st.caption(f"**Source {i+1}** ({score:.1f}%)"); st.text(doc)
                audio_path = text_to_speech(ans)
                if audio_path: st.audio(audio_path, format="audio/mp3")
            st.session_state.messages.append({"role": "assistant", "content": ans, "sources": srcs, "scores": scores, "audio": audio_path})
            save_chat_history(st.session_state.current_subject, st.session_state.messages)
    with tab2:
        if st.button("⚡ Generate Flashcards from Notes"):
            with st.spinner("Generating deck..."):
                cards = generate_flashcards(st.session_state.current_subject)
                if cards: st.session_state.flashcards = cards; st.session_state.card_index = 0; st.session_state.flipped = False; st.rerun()
                else: st.error("No notes found.")
        if st.session_state.flashcards:
            idx = st.session_state.card_index; card = st.session_state.flashcards[idx]
            st.progress((idx + 1) / len(st.session_state.flashcards)); st.caption(f"Card {idx + 1} of {len(st.session_state.flashcards)}")
            content = card["back"] if st.session_state.flipped else card["front"]
            label = "ANSWER" if st.session_state.flipped else "QUESTION"
            if st.button("🔄 Flip Card", key="flip_btn", use_container_width=True): st.session_state.flipped = not st.session_state.flipped; st.rerun()
            st.markdown(f"""<div class="flashcard"><div class="card-label">{label}</div><div class="card-content">{content}</div><div class="hint-text">(Click 'Flip Card' to reveal)</div></div>""", unsafe_allow_html=True)
            c_p, c_n = st.columns(2)
            with c_p: 
                if st.button("⬅️ Previous", disabled=(idx == 0)): st.session_state.card_index -= 1; st.session_state.flipped = False; st.rerun()
            with c_n:
                if st.button("Next ➡️", disabled=(idx == len(st.session_state.flashcards) - 1)): st.session_state.card_index += 1; st.session_state.flipped = False; st.rerun()
        else: st.info("Click 'Generate Flashcards' to start studying.")
    with tab3:
        if st.button("🕸️ Visualize Concepts"):
            with st.spinner("Mapping..."):
                try:
                    dot_code = generate_mind_map(st.session_state.current_subject)
                    if dot_code: st.graphviz_chart(dot_code)
                    else: st.error("No data.")
                except: st.error("Error drawing graph.")
    with tab4:
        if st.button("Generate Quiz"): st.session_state.quiz_data = generate_quiz(st.session_state.current_subject); st.session_state.quiz_submitted = False; st.rerun()
        if "quiz_data" in st.session_state and st.session_state.quiz_data:
            score = 0
            for i, q in enumerate(st.session_state.quiz_data):
                st.caption(f"Q{i+1}: {q['question']}")
                user_choice = st.radio("Choose:", q['options'], key=f"q_{i}", index=None, disabled=st.session_state.quiz_submitted)
                if st.session_state.quiz_submitted:
                    if user_choice == q['answer']: score += 1
            if not st.session_state.quiz_submitted:
                if st.button("Submit"): st.session_state.quiz_submitted = True; st.rerun()
            else: st.metric("Score", f"{score}/{len(st.session_state.quiz_data)}")
    with tab5:
        if st.button("📄 Create Guide"):
            with st.spinner("Writing..."):
                guide = generate_study_guide(st.session_state.current_subject)
                if guide: st.markdown(guide); st.download_button("Download", guide, file_name="guide.md")
                else: st.error("No notes.")
