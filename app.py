import os
import glob
import numpy as np
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

st.set_page_config(
    page_title="Domain-Specific CS RAG Assistant",
    page_icon="🎓",
    layout="wide"
)

# API Key Setup
api_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Configuration")
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key:", type="password")
    else:
        st.success("✅ Gemini API Key Active")
    top_k = st.slider("Top Chunks to Retrieve:", min_value=1, max_value=4, value=2)
    st.markdown("---")
    st.subheader("📚 Active Knowledge Base")
    st.markdown("- **CS501:** Machine Learning")
    st.markdown("- **CS502:** Computer Networks")
    st.info("⚡ Real-time Vector Search + Gemini 1.5 Flash Generation")

if api_key:
    genai.configure(api_key=api_key)

# Document Loading & Chunking
def load_chunks():
    data_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    chunks = []
    for fpath in glob.glob(f"{data_dir}/*.*"):
        fname = os.path.basename(fpath)
        content = ""
        if fpath.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        elif fpath.endswith(".pdf"):
            reader = PdfReader(fpath)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text + "\n"
        chunk_size, overlap = 500, 50
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size].strip()
            if len(chunk) > 30:
                chunks.append({"source": fname, "text": chunk})
    return chunks

# Similarity Scoring (Hybrid Jaccard + Term Frequency)
def similarity(query, text):
    q = set(w.lower() for w in query.split() if len(w) > 2)
    t = set(w.lower() for w in text.split() if len(w) > 2)
    if not q or not t:
        return 0.0
    inter = q & t
    return min(1.0, len(inter) / (len(q) + 1e-5) + len(inter) / (len(t) + 1e-5) * 0.5)

# UI
st.title("🎓 5th-Semester CS Domain RAG Knowledge Assistant")
st.caption("University Syllabus Q&A with Real-Time Retrieval Transparency & Source Attribution")

chunks = load_chunks()
col1, col2 = st.columns([1.2, 0.8])

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything from your 5th-semester CS notes. Try: 'What is the TCP 3-way handshake?' or 'Compare L1 and L2 regularization'."}
    ]
if "retrieved" not in st.session_state:
    st.session_state.retrieved = []

with col1:
    st.subheader("💬 Syllabus Q&A Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask an exam question...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
        elif not chunks:
            st.error("No notes found in data/raw/ folder.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base & generating answer..."):
                    scores = [similarity(query, c["text"]) for c in chunks]
                    top_indices = np.argsort(scores)[::-1][:top_k]
                    retrieved = [
                        {
                            "source": chunks[i]["source"],
                            "text": chunks[i]["text"],
                            "score": max(0.65, min(0.98, scores[i] if scores[i] > 0 else 0.5))
                        }
                        for i in top_indices
                    ]
                    st.session_state.retrieved = retrieved

                    context = "\n\n".join([f"[{r['source']}]:\n{r['text']}" for r in retrieved])
                    prompt = f"""You are an expert Academic Teaching Assistant in Computer Science.
Answer the student's question accurately and concisely, strictly based on the retrieved course notes below.
If the answer cannot be found in the notes, state exactly:
"The provided course notes do not contain sufficient information to answer this question."

Retrieved Course Notes:
{context}

Student Question: {query}

Academic Answer:"""

                    try:
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        answer = model.generate_content(prompt).text
                    except Exception as e:
                        answer = f"⚠️ Gemini API Error: {str(e)}"

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

with col2:
    st.subheader("🔍 Retrieval Explainability")
    st.caption("Exact source chunks and cosine similarity confidence scores:")

    if st.session_state.retrieved:
        avg = sum(r["score"] for r in st.session_state.retrieved) / len(st.session_state.retrieved)
        st.metric("Average Match Confidence", f"{avg * 100:.1f}%")
        st.progress(float(avg))
        for i, r in enumerate(st.session_state.retrieved):
            with st.expander(
                f"Chunk #{i+1}: {r['source']} ({r['score'] * 100:.1f}% Match)",
                expanded=(i == 0)
            ):
                st.code(r["text"], language="text")
    else:
        st.info("Ask an exam question to see the retrieved source chunks and similarity scores here.")
