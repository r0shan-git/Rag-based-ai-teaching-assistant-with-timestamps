# 🎓 RAG-Based AI Teaching Assistant with Video Timestamps

This project is an **end-to-end Retrieval-Augmented Generation (RAG) system** built on **course videos**.  
It allows users to ask questions like **“Where is Exercise 1 taught?”** and get the **exact video name, video number, and timestamp (mm:ss)** where the topic or exercise is explained.

The system is implemented step-by-step using **Whisper, embeddings, vector similarity search, and LLMs**, and is finally exposed through an interactive **Streamlit web app**.

---

## 🚀 What This Project Does

- Converts course videos → audio (MP3)
- Transcribes and translates audio using **Whisper**
- Splits transcripts into **timestamped chunks**
- Creates **vector embeddings** using `bge-m3` (Ollama)
- Performs **semantic similarity search**
- Uses an **LLM (llama3.2)** to return:
  - 📺 Video Title  
  - 🔢 Video Number  
  - ⏱ Exact Timestamp (mm:ss – mm:ss)  
  - 📖 What is practiced  
  - 👉 Clear guidance on where to watch  

---

## 🧠 Complete Workflow (Step-by-Step)

1. Video → MP3 (FFmpeg)  
2. MP3 → Text (Whisper transcription + translation)  
3. Text → Chunks (with start & end timestamps)  
4. Chunks → Embeddings (Ollama + bge-m3)  
5. Query → Similarity Search (cosine similarity)  
6. LLM response with precise video & timestamp  
7. Streamlit UI for interactive querying  

---

## 🛠 Tech Stack

- Frontend: Streamlit  
- Speech-to-Text: OpenAI Whisper  
- Embeddings: bge-m3 (via Ollama)  
- LLM: llama3.2 (via Ollama)  
- Vector Search: scikit-learn (cosine similarity)  
- Data Handling: Pandas, Joblib  
- Language: Python  

---

## 📂 Project Structure

.
├── step1_video_to_mp3.py          # Video → Audio  
├── step2_mp3_to_json.py           # Audio → Transcription  
├── step3_preprocess_json.py       # Chunking with timestamps  
├── step4_code.py                  # Embedding generation  
├── step5_process_incoming.py      # Query processing  
├── step6_app.py                   # Streamlit application  
│
├── audios/                        # Extracted audio files  
├── jsons/                         # Transcript & embedding JSONs  
├── whisper/                       # Whisper outputs  
├── embeddings.joblib              # Vector store (ignored)  
├── prompt.txt                     # LLM prompt  
└── response.txt                   # Model response  

---


## ✨ Key Highlights
- End-to-end RAG pipeline on real course videos
- Timestamp-level answers (mm:ss accuracy)
- Fully offline system (no paid APIs)
- Exercise & practice–focused retrieval
- Built with Whisper + Ollama + Streamlit

## 🎯 Problem Statement
Long course videos make it difficult to quickly find
where a specific topic or exercise is explained.
This project solves that by enabling semantic search
and exact timestamp-based navigation.

## 🧪 Example Output

Question:
Where is Exercise 1 taught?

Answer:
• Video Title: HTML Forms Tutorial  
• Video Number: 07  
• Timestamp: 12:30 – 18:45  
• Guidance: Open video 07 and start watching from 12:30

## ⚙️ How to Update Videos
1. Add new videos to `video/`
2. Run:
   - step1_video_to_mp3.py
   - step2_mp3_to_json.py
   - step3_process_incoming.py
3. Restart Streamlit app


## 🔐 Design Decisions
- Used cosine similarity for fast semantic matching
- Chunk-based retrieval for precise timestamps
- Local LLM via Ollama for privacy & cost efficiency
- Strict prompt to avoid hallucinations

## 🔄 Pipeline

1. 🎥 **Video Ingestion**  
   Input course videos are collected for processing.

2. 🎧 **Audio Extraction**  
   Videos are converted into audio files (MP4 → MP3).

3. 📝 **Speech-to-Text**  
   Audio is transcribed using Whisper with timestamps.

4. ✂️ **Text Chunking**  
   Transcripts are split into meaningful chunks while preserving start and end times.

5. 🧩 **Embedding Generation**  
   Each text chunk is converted into semantic embeddings.

6. 📦 **Vector Storage**  
   Embeddings along with metadata are stored for fast retrieval.

7. 🔍 **Query Processing**  
   User query is embedded and compared using similarity search.

8. ⏱️ **Timestamp Retrieval**  
   Most relevant video segment with exact timestamp is identified.

9. 🌐 **User Interface**  
   Streamlit app displays the answer and navigates to the correct video time.


## 🚀 Future Improvements

- Improve timestamp accuracy using finer text chunking  
- Enhance semantic search with better embedding models  
- Add keyword + semantic hybrid search  
- Optimize performance for faster query response  
- Support multiple languages  
- Develop a user-friendly web interface  
- Enable cloud-based storage and processing  
- Add user feedback to improve result relevance  
