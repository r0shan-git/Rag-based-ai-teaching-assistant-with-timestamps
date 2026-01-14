import streamlit as st
import requests
import joblib
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------
# CONFIG
# --------------------------------
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "llama3.2"
TOP_K = 5
DATA_FILE = "embeddings.joblib"   # <-- change only if filename differs

# --------------------------------
# PAGE SETUP
# --------------------------------
st.set_page_config(
    page_title="🎓 RAG-based AI Teaching Assistant",
    layout="wide"
)

# --------------------------------
# SIDEBAR
# --------------------------------
with st.sidebar:
    st.title("🎓 RAG-based AI Teaching Assistant")
    st.markdown("""
    **On Your Own Course Videos**

    ---
    ### 🚀 Features
    - Exercise & practice locator  
    - Timestamp-level answers  
    - Vector similarity search  
    - LLM-guided explanation  

    ---
    ### 🧠 How It Works
    1. Videos → Audio  
    2. Audio → Text  
    3. Text → Chunks  
    4. Chunks → Embeddings  
    5. Query → Similarity Search  
    6. Exact exercise location returned  
    """)

# --------------------------------
# LOAD DATA (SAFE)
# --------------------------------
@st.cache_resource
def load_dataframe():
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ File not found: `{DATA_FILE}`")
        st.stop()
    return joblib.load(DATA_FILE)

df = load_dataframe()

# --------------------------------
# FUNCTIONS
# --------------------------------
def create_embeddings(text_list):
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "input": text_list},
        timeout=120
    )
    return r.json()["embeddings"]

def inference(prompt):
    r = requests.post(
        OLLAMA_GEN_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    return r.json()["response"]

def seconds_to_mmss(seconds):
    """Convert any seconds (even >3600) to standard mm:ss"""
    seconds = int(seconds)
    minutes = seconds // 60
    sec = seconds % 60
    return f"{minutes:02d}:{sec:02d}"

# --------------------------------
# MAIN UI
# --------------------------------
st.title("🎥 Video Exercise Locator")
st.markdown(
    "Ask questions like **“Where is Exercise 1 taught?”** and get "
    "**exact video + timestamp (standard clock format)**."
)

st.divider()

query = st.text_input(
    "🔍 Ask your question (example: Where is Exercise 1 taught?)",
    placeholder="Where is Exercise 1 taught?"
)

# --------------------------------
# SEARCH & ANSWER
# --------------------------------
if st.button("🔎 Search") and query.strip():

    # Bias query toward exercises
    search_query = query + " exercise practice hands-on implementation"

    with st.spinner("Creating query embedding..."):
        query_embedding = create_embeddings([search_query])[0]

    with st.spinner("Finding relevant video segments..."):
        similarity = cosine_similarity(
            np.vstack(df["embedding"]),
            [query_embedding]
        ).flatten()

        top_indices = similarity.argsort()[::-1][:TOP_K]
        results_df = df.loc[top_indices]

    st.subheader("📌 Retrieved Video Segments (Exercise Related)")

    for i, row in enumerate(results_df.itertuples(), start=1):
        start_time = seconds_to_mmss(row.start)
        end_time = seconds_to_mmss(row.end)

        st.markdown(f"### Result {i}")
        st.write(f"📺 **Video Title:** {row.title}")
        st.write(f"🔢 **Video Number:** {row.number}")
        st.write(f"⏱ **Timestamp:** {start_time} – {end_time}")
        st.write(f"📝 **Content:** {row.text}")
        st.divider()

    # --------------------------------
    # STRICT EXERCISE-FOCUSED PROMPT
    # --------------------------------
    prompt = f"""
You are an AI Teaching Assistant for the Sigma Web Development Course.

Your ONLY task is to locate WHERE an EXERCISE is TAUGHT.

Rules (STRICT):
- Focus ONLY on exercises, practice, hands-on, or implementation
- Ignore theory-only explanations
- Do NOT guess
- If Exercise 1 is NOT found, clearly say:
  "Exercise 1 is not taught in the provided video segments."

You are given video data containing:
- Video title
- Video number
- Start time (seconds)
- End time (seconds)
- What is taught

Course Data:
{results_df[["title","number","start","end","text"]].to_json(orient="records")}

User Question:
"{query}"

Answer Format (FOLLOW EXACTLY):

📌 Exercise Location:
• Video Title:
• Video Number:
• Timestamp (mm:ss – mm:ss)

📖 What is practiced:
• One short line

👉 Guidance:
• Tell the user exactly which video to open and from where

DO NOT:
- Explain theory
- Mention subtitles, chunks, JSON, or datasets
"""

    with st.spinner("Generating precise exercise location..."):
        response = inference(prompt)

    st.subheader("🤖 AI Answer (Exercise Location)")
    st.write(response)

else:
    st.info("👆 Enter a question and click **Search**")
