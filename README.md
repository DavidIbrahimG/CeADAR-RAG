# CeADAR Data Scientist II — Retrieval-Augmented Generation (RAG) Prototype

This repository contains an end-to-end **Retrieval-Augmented Generation (RAG)** proof-of-concept designed to demonstrate the ability to ingest heterogeneous documents, retrieve relevant information using embeddings, and generate **grounded, factual responses** using an open-weights Large Language Model.

The system was built to align explicitly with the **CeADAR Data Scientist II challenge requirements**, with a strong emphasis on **architecture clarity, evaluation rigor, hallucination resistance, and explainability**.

---

## 1. Problem Overview

Large Language Models are powerful but prone to hallucination when operating without access to authoritative sources.  
This project demonstrates a **RAG pipeline** that:

- Retrieves relevant document passages before generation  
- Grounds all answers strictly in retrieved context  
- Explicitly refuses to answer out-of-scope questions  
- Supports conversational follow-ups without using memory as a source of truth  

---

## 2. System Architecture

### High-Level Pipeline

```
PDF / DOCX Documents
        ↓
Text Extraction & Cleaning
        ↓
Chunking with Overlap
        ↓
Embedding Generation (SentenceTransformers)
        ↓
Vector Storage (Chroma)
        ↓
Top-K Similarity Retrieval
        ↓
Query Rewriting (Conversation Memory)
        ↓
LLM Generation (Groq – Open Weights)
        ↓
Grounded Answer + Explicit Sources
```

### Key Architectural Principle

> **Conversation memory is used only to improve retrieval, never as a source of factual knowledge.**

---

## 3. Project Structure

```text
Ceadar/
│
├── ingestion/               # Document loading, chunking, indexing
│   ├── loaders.py
│   ├── chunking.py
│   └── build_index.py
│
├── vector_store/            # Embeddings + Chroma integration
│   ├── embeddings.py
│   └── chroma_store.py
│
├── rag/                     # Core RAG logic
│   ├── retriever.py
│   ├── generator.py
│   ├── query_rewriter.py
│   └── pipeline.py
│
├── evaluation/              # Lightweight evaluation harness
│   └── run_eval.py
│
├── app/                     # Streamlit UI
│   └── app.py
│
├── data/
│   └── raw/                 # Provided PDF & DOCX documents
│
├── requirements.txt
├── README.md
└── .env.example
```

---

## 4. Key Design Decisions & Trade-offs

### Embeddings
- **SentenceTransformers (all-MiniLM-L6-v2)** used for local, open-source embeddings
- Chosen for strong semantic performance with low computational cost

### Vector Database
- **Chroma** used for persistent local vector storage
- Trade-off: simplicity and reproducibility over managed cloud services

### Language Model
- **Groq-hosted open-weights model (e.g. Llama 3.1)**
- Provides fast inference while aligning with open-model principles

### Hallucination Mitigation
- The generator is explicitly instructed to:
  - Use **only retrieved context**
  - Refuse unsupported queries using a **fixed refusal phrase**

---

## 5. Conversation Memory (Retrieval-Only)

The system supports conversational follow-ups by rewriting user questions into **standalone retrieval queries** using chat history.

### Example

```
User: What is self-attention?
User: How is it different from multi-head attention?
```

Rewritten retrieval query:

```
Difference between self-attention and multi-head attention in the Transformer architecture
```

This rewritten query is used **only for retrieval**.  
The final answer is still generated **exclusively from retrieved document chunks**.

The Streamlit UI optionally exposes the rewritten query for **transparency and debugging**.

---

## 6. Streamlit User Interface

The deployed Streamlit app provides:

- Chat-style interaction
- Top-K retrieval controls
- Rebuildable index
- Explicit source attribution
- Optional display of rewritten retrieval queries

---

## 7. Testing & Evaluation

### Evaluation Design

A structured evaluation was conducted using a small but systematic test set designed to cover:

- Factual recall
- Conceptual reasoning
- Cross-document retrieval
- Conversational follow-ups
- Hallucination resistance

### Test Queries

| ID | Query | Expected Behaviour |
|---|------|------------------|
| Q1 | What is self-attention and why is it useful? | Answer with citations |
| Q2 | What are key components of the Transformer architecture? | Answer with citations |
| Q3 | What problem does the Transformer address vs recurrent models? | Grounded explanation |
| Q4 | What are the main themes of the EU AI Act? | Legal summary |
| Q5 | How is it different from multi-head attention? | Correct follow-up |
| Q6 | Who won the 2022 World Cup? | Explicit refusal |

### Evaluation Metrics

- **Answer Correctness**
- **Groundedness**
- **Citation Coverage**
- **Hallucination Resistance**

### Results Summary

| Metric | Result |
|------|--------|
| Answer Correctness | 4 / 4 |
| Groundedness | 5 / 5 |
| Citation Coverage | 4 / 4 |
| Hallucination Resistance | 1 / 1 |

---

## 8. Observations

- Retrieval quality is strong for well-defined technical concepts
- Conversational query rewriting significantly improves follow-up retrieval
- Longer legal passages sometimes require more context than a single chunk

---

## 9. Limitations

- Chunking may split long legal arguments
- Evaluation is qualitative and based on a small test set

---

## 10. Future Improvements

- Add a reranker for improved retrieval precision
- Implement hybrid keyword + embedding search
- Introduce automated faithfulness metrics


---

## 11. How to Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add environment variables in `.env`:

```
GROQ_API_KEY=...
GROQ_MODEL=llama-3.1-8b-instant
```

Build the index:

```bash
python -m ingestion.build_index
```

Run the app:

```bash
PYTHONPATH=. streamlit run app/app.py
```

Run evaluation:

```bash
PYTHONPATH=. python evaluation/run_eval.py
```

---

## 12. Assessment Alignment Summary

This project demonstrates:

- End-to-end RAG implementation  
- Multi-format document ingestion  
- Open-weights LLM usage  
- Grounded generation with refusal behaviour  
- Systematic testing & evaluation  
- Clear architecture and trade-off reasoning  

---

### Final Note

This prototype was designed to prioritise **correctness, explainability, and reliability** over model complexity, reflecting real-world constraints in applied research and production systems.
