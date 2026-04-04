import os
import io
import json
import tempfile
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import PyPDF2
import docx

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── System Prompts ───────────────────────────────────────────────
CAREER_COACH_PROMPT = """
You are an expert Career Coach with 20 years of experience. You help with:
- Resume writing and optimization
- Interview preparation and mock interviews
- Career path guidance
- LinkedIn profile improvement
- Salary negotiation tips
Be friendly, encouraging, and give specific, actionable advice.
If the user uploads a document, use it to give personalized advice.
"""

INTERVIEW_PROMPT = """
You are a professional technical interviewer conducting a structured mock interview.
Rules:
1. Ask ONE question at a time
2. Wait for the candidate's answer
3. Give brief feedback on their answer (2-3 sentences)
4. Then ask the next question
5. After 5 questions, give a final performance summary with scores
6. Be professional but encouraging
7. Tailor questions to the job role mentioned

Format your responses exactly like:
[QUESTION X/5]: <question here>
or
[FEEDBACK]: <feedback on their answer>
[QUESTION X/5]: <next question>
or
[FINAL REPORT]: <detailed performance summary>
"""

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(page_title="Career Coach AI", page_icon="💼", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f;
    color: #e8e6e0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: 
        radial-gradient(ellipse at 20% 20%, rgba(212, 175, 55, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(139, 90, 43, 0.08) 0%, transparent 50%),
        #0a0a0f;
}

[data-testid="stSidebar"] {
    background: rgba(15, 15, 20, 0.95) !important;
    border-right: 1px solid rgba(212, 175, 55, 0.1) !important;
    display: block !important;
    visibility: visible !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: block !important;
    visibility: visible !important;
    color: #d4af37 !important;
}

[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 2rem 3rem; max-width: 900px; margin: 0 auto; }

.hero {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid rgba(212, 175, 55, 0.15);
    margin-bottom: 1.5rem;
}

.hero-badge {
    display: inline-block;
    background: rgba(212, 175, 55, 0.1);
    border: 1px solid rgba(212, 175, 55, 0.3);
    color: #d4af37;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #f0ede6;
    line-height: 1.1;
    margin-bottom: 0.75rem;
}

.hero h1 span { color: #d4af37; }
.hero p { color: #8a8580; font-size: 1rem; font-weight: 300; }

.mode-banner {
    background: rgba(212, 175, 55, 0.08);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1.5rem;
    color: #d4af37;
    font-size: 0.85rem;
    text-align: center;
}

.interview-banner {
    background: rgba(255, 100, 100, 0.08);
    border: 1px solid rgba(255, 100, 100, 0.2);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1.5rem;
    color: #ff6464;
    font-size: 0.85rem;
    text-align: center;
}

.suggestion-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-bottom: 2rem;
}

.suggestion-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    transition: all 0.2s ease;
}

.suggestion-card .icon { font-size: 1.1rem; margin-bottom: 0.3rem; }
.suggestion-card .label { font-size: 0.8rem; color: #c8c4bc; font-weight: 400; }

[data-testid="stChatInputContainer"] {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(10, 10, 15, 0.95);
    backdrop-filter: blur(20px);
    border-top: 1px solid rgba(212, 175, 55, 0.1);
    padding: 1rem 3rem;
    z-index: 999;
}

[data-testid="stChatInputContainer"] textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important;
    border-radius: 12px !important;
    color: #f0ede6 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stChatInputContainer"] textarea:focus {
    border-color: rgba(212, 175, 55, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.08) !important;
}

[data-testid="stChatInputContainer"] button {
    background: #d4af37 !important;
    border-radius: 10px !important;
    color: #0a0a0f !important;
}

.bottom-spacer { height: 120px; }

/* Sidebar styling */
.stButton button {
    width: 100%;
    background: rgba(212, 175, 55, 0.1) !important;
    border: 1px solid rgba(212, 175, 55, 0.3) !important;
    color: #d4af37 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stButton button:hover {
    background: rgba(212, 175, 55, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Helper Functions ─────────────────────────────────────────────
def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or TXT files"""
    text = ""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = uploaded_file.read().decode("utf-8")
    return text

def build_vectorstore(text):
    """Build FAISS vectorstore from text"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def retrieve_context(query, vectorstore, k=3):
    """Retrieve relevant chunks from vectorstore"""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def transcribe_audio(audio_bytes):
    """Transcribe audio using Groq Whisper"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        with open(f.name, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text"
            )
    return transcription

def is_interview_request(text):
    """Detect if user wants a mock interview"""
    keywords = ["mock interview", "interview me", "start interview", 
                "practice interview", "interview practice", "conduct interview"]
    return any(kw in text.lower() for kw in keywords)

def get_ai_response(messages, system_prompt, context=None):
    """Get response from Groq"""
    system = system_prompt
    if context:
        system += f"\n\nRelevant document context:\n{context}"
    
    api_messages = [{"role": "system", "content": system}] + messages
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=api_messages,
        max_tokens=1000
    )
    return response.choices[0].message.content

# ─── Session State Init ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "interview_mode" not in st.session_state:
    st.session_state.interview_mode = False
if "interview_question_count" not in st.session_state:
    st.session_state.interview_question_count = 0
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0; border-bottom: 1px solid rgba(212,175,55,0.2); margin-bottom: 1rem;'>
        <div style='font-family: Playfair Display; color: #d4af37; font-size: 1.2rem;'>⚙️ Controls</div>
    </div>
    """, unsafe_allow_html=True)

    # Document Upload
    st.markdown("<p style='color:#8a8580; font-size:0.8rem; letter-spacing:0.1em; text-transform:uppercase;'>📄 Upload Document</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Resume, JD, or any document",
        type=["pdf", "docx"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if st.session_state.doc_name != uploaded_file.name:
            with st.spinner("Processing document..."):
                text = extract_text_from_file(uploaded_file)
                st.session_state.vectorstore = build_vectorstore(text)
                st.session_state.doc_name = uploaded_file.name
            st.success(f"✅ {uploaded_file.name} loaded!")

    if st.session_state.vectorstore:
        st.markdown(f"<p style='color:#d4af37; font-size:0.8rem;'>📎 {st.session_state.doc_name}</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Voice Input
    st.markdown("<p style='color:#8a8580; font-size:0.8rem; letter-spacing:0.1em; text-transform:uppercase;'>🎙️ Voice Input</p>", unsafe_allow_html=True)
    audio_input = st.audio_input("Speak your message", label_visibility="collapsed")

    if audio_input:
        with st.spinner("Transcribing..."):
            transcribed = transcribe_audio(audio_input.read())
        st.markdown(f"<p style='color:#d4af37; font-size:0.85rem;'>🎤 <i>{transcribed}</i></p>", unsafe_allow_html=True)
        st.session_state["voice_input"] = transcribed

    st.markdown("<br>", unsafe_allow_html=True)

    # Mock Interview
    st.markdown("<p style='color:#8a8580; font-size:0.8rem; letter-spacing:0.1em; text-transform:uppercase;'>🎯 Mock Interview</p>", unsafe_allow_html=True)
    
    if not st.session_state.interview_mode:
        job_role = st.text_input("Job role (e.g. ML Engineer)", placeholder="ML Engineer")
        if st.button("🎯 Start Mock Interview"):
            st.session_state.interview_mode = True
            st.session_state.interview_question_count = 0
            starter = get_ai_response(
                [{"role": "user", "content": f"Start a mock interview for a {job_role if job_role else 'Software Engineer'} position. Introduce yourself briefly and ask the first question."}],
                INTERVIEW_PROMPT
            )
            st.session_state.messages.append({"role": "assistant", "content": starter})
            st.rerun()
    else:
        st.markdown(f"<p style='color:#ff6464; font-size:0.85rem;'>🔴 Interview in progress</p>", unsafe_allow_html=True)
        if st.button("⏹ End Interview"):
            st.session_state.interview_mode = False
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Interview ended. Feel free to ask me any career questions!"
            })
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Clear Chat
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.interview_mode = False
        st.session_state.interview_question_count = 0
        st.rerun()

# ─── Main UI ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI Powered</div>
    <h1>Your <span>Career Coach</span><br/>is here</h1>
    <p>Expert guidance for resumes, interviews, and career growth</p>
</div>
""", unsafe_allow_html=True)

# Mode banners
if st.session_state.interview_mode:
    st.markdown('<div class="interview-banner">🔴 Mock Interview Mode Active — Answer each question naturally. Good luck!</div>', unsafe_allow_html=True)
elif st.session_state.vectorstore:
    st.markdown(f'<div class="mode-banner">📎 Document loaded: <b>{st.session_state.doc_name}</b> — Ask me anything about it!</div>', unsafe_allow_html=True)

# Suggestions
if not st.session_state.messages:
    st.markdown("""
    <div class="suggestion-grid">
        <div class="suggestion-card">
            <div class="icon">📄</div>
            <div class="label">Upload your resume for personalized advice</div>
        </div>
        <div class="suggestion-card">
            <div class="icon">🎯</div>
            <div class="label">Start a mock interview for any role</div>
        </div>
        <div class="suggestion-card">
            <div class="icon">💰</div>
            <div class="label">How do I negotiate a higher salary?</div>
        </div>
        <div class="suggestion-card">
            <div class="icon">🎙️</div>
            <div class="label">Use voice input to speak your questions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── Chat Input ───────────────────────────────────────────────────
voice_text = st.session_state.pop("voice_input", None)
prompt = st.chat_input("Ask your career question...") or voice_text

if prompt:
    # Check if user wants to start interview via chat
    if is_interview_request(prompt) and not st.session_state.interview_mode:
        st.session_state.interview_mode = True
        st.session_state.interview_question_count = 0

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get context from vectorstore if available
    context = None
    if st.session_state.vectorstore:
        context = retrieve_context(prompt, st.session_state.vectorstore)

    # Choose system prompt
    system = INTERVIEW_PROMPT if st.session_state.interview_mode else CAREER_COACH_PROMPT

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_ai_response(st.session_state.messages, system, context)
            st.markdown(reply)

        # Track interview questions
        if st.session_state.interview_mode and "FINAL REPORT" in reply:
            st.session_state.interview_mode = False

    st.session_state.messages.append({"role": "assistant", "content": reply})

st.markdown('<div class="bottom-spacer"></div>', unsafe_allow_html=True)
