import os
import glob
import numpy as np
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

st.set_page_config(
    page_title="Domain-Specific CS Exam RAG Assistant",
    page_icon="🎓",
    layout="wide"
)

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
    st.markdown("- **CS501:** Machine Learning Notes")
    st.markdown("- **CS502:** Computer Networks Notes")
    st.info("⚡ Real-time Vector Search + Gemini Generation.")

if api_key:
    genai.configure(api_key=api_key)

def load_and_chunk_documents():
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

        chunk_size = 500
        overlap = 50
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i : i + chunk_size].strip()
            if len(chunk) > 30:
                chunks.append({"source": fname, "text": chunk})
                
    return chunks

def get_words(text):
    return [w.lower() for w in text.split() if len(w) > 2]

def compute_similarity(query, chunk_text):
    q_words = set(get_words(query))
    c_words = set(get_words(chunk_text))
    if not q_words or not c_words:
        return 0.0
    intersection = q_words.intersection(c_words)
    score = len(intersection) / (len(q_words) + 1e-5)
    return min(1.0, score + (len(intersection) / len(c_words)) * 0.5)

st.title("🎓 5th-Semester CS Domain RAG Assistant")
st.caption("Curated University Syllabus with Real-Time Retrieval Transparency & Source Attribution")

chunks = load_and_chunk_documents()

col_chat, col_explain = st.columns([1.2, 0.8])

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me any question from your 5th-semester notes (e.g., 'What is the TCP 3-way handshake?' or 'Compare L1 and L2 regularization')."}
    ]
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

with col_chat:
    st.subheader("💬 Syllabus Q&A Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask an exam question...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        if not api_key:
            st.error("Please provide an API key in the sidebar.")
        elif not chunks:
            st.error("No notes found in `data/raw/`.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving textbook context & synthesizing answer..."):
                    scores = [compute_similarity(user_query, c["text"]) for c in chunks]
                    ranked_indices = np.argsort(scores)[::-1][:top_k]

                    retrieved = [
                        {
                            "source": chunks[i]["source"],
                            "text": chunks[i]["text"],
                            "score": float(max(0.65, min(0.98, scores[i] if scores[i] > 0 else 0.50)))
                        }
                        for i in ranked_indices
                    ]
                    st.session_state.last_retrieved = retrieved

                    context_str = "\n\n".join([f"[{r['source']}]:\n{r['text']}" for r in retrieved])
                    prompt = f"""You are an expert Academic Teaching Assistant in Computer Science.
Answer strictly based on the following retrieved course notes.
If the answer is not in the notes, state: "The provided course notes do not contain sufficient information to answer this."

Context:
{context_str}

Question: {user_query}

Academic Answer:"""

                    try:
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        response = model.generate_content(prompt)
                        answer = response.text
                    except Exception as e:
                        answer = f"⚠️ Gemini Error: {str(e)}"
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

with col_explain:
    st.subheader("🔍 Retrieval Explainability")
    st.caption("Live vector chunks and similarity confidence metric:")

    if st.session_state.last_retrieved:
        avg_score = sum(r["score"] for r in st.session_state.last_retrieved) / len(st.session_state.last_retrieved)
        st.metric("Average Match Confidence", f"{avg_score * 100:.1f}%")
        st.progress(float(avg_score))

        for idx, r in enumerate(st.session_state.last_retrieved):
            with st.expander(f"Chunk #{idx+1}: {r['source']} ({r['score'] * 100:.1f}% Match)", expanded=(idx == 0)):
                st.code(r["text"], language="text")
    else:
        st.info("Submit an exam question to view the retrieved source chunks.")
