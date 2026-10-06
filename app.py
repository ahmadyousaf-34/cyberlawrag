import os
import re
import io
import hashlib
import requests
import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# ============================================================
# PECA Pakistan Legal RAG Assistant
# Files required: app.py, requirements.txt, readme.md
# ============================================================

st.set_page_config(
    page_title="Pakistan Cyber Law RAG Assistant",
    page_icon="⚖️",
    layout="wide",
)

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Official Pakistan Code PDF used as the default startup source.
# You can replace this in Streamlit Secrets with PECA_PDF_URL.
DEFAULT_PDF_URL = (
    "https://www.pakistancode.gov.pk/pdffiles/"
    "administrator6a061efe0ed5bd153fa8b79b8eb4cba7.pdf"
)

DISCLAIMER = (
    "This application is an educational information/retrieval tool, not a lawyer "
    "or a substitute for professional legal advice. Answers are grounded in the "
    "loaded PECA source and may not reflect later amendments, court decisions, "
    "rules, notifications, or facts outside the source."
)


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, None)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def section_label(text: str) -> str:
    # Try to identify a PECA section number from the beginning of a chunk.
    patterns = [
        r"(?im)^\s*(\d+[A-Z]?)\.\s+",
        r"(?im)\bsection\s+(\d+[A-Z]?)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return f"Section {m.group(1)}"
    return "Source passage"


def split_into_chunks(page_text: str, page_no: int, max_chars: int = 1400,
                       overlap: int = 220):
    """Chunk text while retaining page number and likely section labels."""
    text = clean_text(page_text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
            continue

        if current:
            chunks.append(current)

        # Long paragraphs are split by sentence-ish boundaries.
        if len(para) > max_chars:
            start = 0
            while start < len(para):
                end = min(start + max_chars, len(para))
                piece = para[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(para):
                    break
                start = max(0, end - overlap)
            current = ""
        else:
            # Keep a small tail to preserve legal context across chunks.
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}".strip()

    if current:
        chunks.append(current)

    records = []
    for i, chunk in enumerate(chunks):
        records.append(
            {
                "text": chunk,
                "page": page_no,
                "section": section_label(chunk),
                "chunk_id": f"p{page_no}-c{i}",
            }
        )
    return records


@st.cache_data(show_spinner=False)
def download_pdf(pdf_url: str) -> bytes:
    response = requests.get(
        pdf_url,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 PECA-RAG-App"},
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if not response.content.startswith(b"%PDF") and "pdf" not in content_type:
        raise ValueError("The supplied URL did not return a PDF file.")
    return response.content


@st.cache_resource(show_spinner=False)
def build_vector_store(pdf_url: str):
    pdf_bytes = download_pdf(pdf_url)

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    records = []

    for page_index in range(len(document)):
        page = document.load_page(page_index)
        text = page.get_text("text")
        records.extend(split_into_chunks(text, page_index + 1))

    document.close()

    if not records:
        raise ValueError("No readable text was extracted from the PDF.")

    texts = [r["text"] for r in records]

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    source_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]

    return {
        "index": index,
        "records": records,
        "embedder": embedder,
        "source_hash": source_hash,
        "pages": len(document) if False else max(r["page"] for r in records),
    }


def retrieve(store, query: str, top_k: int, min_score: float):
    query_vector = store["embedder"].encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = store["index"].search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        record = store["records"][int(idx)].copy()
        record["score"] = float(score)
        if score >= min_score:
            results.append(record)

    return results


def format_context(results):
    if not results:
        return "NO_RELEVANT_SOURCE_PASSAGES_FOUND"

    blocks = []
    for n, item in enumerate(results, start=1):
        blocks.append(
            f"[SOURCE {n} | {item['section']} | PDF page {item['page']} | "
            f"similarity {item['score']:.3f}]\n{item['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(question, results, language, technicality, response_size,
                    category, reasoning_effort, conversation):
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets or as an "
            "environment variable."
        )

    client = Groq(api_key=api_key)

    context = format_context(results)

    size_instructions = {
        "Short": "Answer in about 120-180 words.",
        "Medium": "Answer in about 250-400 words.",
        "Detailed": "Answer in about 500-750 words.",
        "Very detailed": "Answer in about 800-1100 words.",
    }

    technicality_instructions = {
        "Beginner": "Use simple language and explain legal/technical terms briefly.",
        "Intermediate": "Use moderate legal and technical terminology with concise explanations.",
        "Advanced": "Use precise legal and cybersecurity terminology and discuss relevant distinctions.",
        "Law-student": "Use structured legal terminology, section references, elements of an offence, and procedural distinctions where the source supports them.",
    }

    language_instruction = {
        "English": "Respond in English.",
        "Urdu": "Respond in Urdu script where practical.",
        "Roman Urdu": "Respond in Roman Urdu.",
    }[language]

    history = ""
    if conversation:
        recent = conversation[-6:]
        history = "\n\nRECENT CONVERSATION:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent
        )

    system_prompt = f"""
You are a Pakistan cyber-law retrieval assistant.

PRIMARY SOURCE RULE:
- The retrieved passages below are the controlling source for this answer.
- Answer only from the supplied PECA source passages.
- Do NOT invent section numbers, punishments, procedures, definitions, exceptions,
  legal conclusions, or amendments.
- If the retrieved passages do not contain enough information, explicitly say:
  "The supplied PECA source does not provide enough information to answer this."
- If the question is outside the loaded Act/source, say that it is outside the
  available source instead of filling the gap with general legal knowledge.
- When making a legal claim, cite the relevant source as [Section X, PDF page Y].
- If a source passage has no recognizable section number, cite its PDF page.
- Distinguish clearly between what the Act says and what would require a lawyer,
  court, or another law.
- Do not present the answer as a guaranteed legal outcome.
- Do not provide instructions for committing cybercrime. For a harmful or illegal
  how-to question, provide a lawful, safety-focused explanation of the relevant
  legal provision instead.

ANSWER STYLE:
{language_instruction}
{technicality_instructions[technicality]}
{size_instructions[response_size]}
Question category selected by user: {category}
Reasoning effort requested: {reasoning_effort}

For legal questions, prefer this structure when applicable:
1. Direct answer
2. Relevant PECA section(s)
3. What the source says
4. Important limitation / caveat

Always use the exact section number supported by the retrieved text.

{history}
"""

    user_prompt = f"""
USER QUESTION:
{question}

RETRIEVED PECA SOURCE:
{context}

Now answer the user's question strictly using the retrieved source.
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens={
            "Short": 500,
            "Medium": 900,
            "Detailed": 1500,
            "Very detailed": 2200,
        }[response_size],
        reasoning_effort=reasoning_effort.lower(),
    )

    return completion.choices[0].message.content


# --------------------------- UI ---------------------------

st.title("⚖️ Pakistan Cyber Law RAG Assistant")
st.caption("PECA-focused legal information retrieval using FAISS + Sentence Transformers + Groq GPT-OSS 20B")

with st.sidebar:
    st.header("⚙️ Settings")

    language = st.selectbox(
        "Response language",
        ["English", "Roman Urdu", "Urdu"],
        index=0,
    )

    technicality = st.selectbox(
        "Technicality level",
        ["Beginner", "Intermediate", "Advanced", "Law-student"],
        index=1,
    )

    response_size = st.selectbox(
        "Response size",
        ["Short", "Medium", "Detailed", "Very detailed"],
        index=1,
    )

    category = st.selectbox(
        "Question category",
        [
            "General",
            "Offence / Section",
            "Punishment / Fine",
            "Cybercrime scenario",
            "Investigation / Procedure",
            "Evidence / Search / Seizure",
            "Rights / Remedies",
            "Other",
        ],
    )

    top_k = st.slider("Retrieved passages", 3, 10, 6)
    min_score = st.slider("Minimum similarity", 0.10, 0.60, 0.20, 0.05)

    reasoning_effort = st.selectbox(
        "GPT-OSS reasoning effort",
        ["Low", "Medium", "High"],
        index=1,
        help="Higher reasoning can improve complex legal comparisons but may use more tokens.",
    )

    show_sources = st.checkbox("Show retrieved source passages", value=True)
    use_history = st.checkbox("Use recent chat context", value=True)

    st.divider()
    st.subheader("📄 Source PDF")

    pdf_url = st.text_input(
        "PDF URL",
        value=get_secret("PECA_PDF_URL", DEFAULT_PDF_URL),
        help="The PDF is downloaded and embedded automatically when the app starts.",
    )

    if st.button("🔄 Rebuild PDF index"):
        download_pdf.clear()
        build_vector_store.clear()
        st.rerun()

    st.caption("Embedding model: all-MiniLM-L6-v2")
    st.caption(f"LLM: {MODEL_NAME}")

st.info(DISCLAIMER)

# Build index on startup.
with st.spinner("Downloading PECA PDF and building FAISS embeddings..."):
    try:
        store = build_vector_store(pdf_url.strip())
    except Exception as exc:
        st.error(f"Could not build the legal knowledge base: {exc}")
        st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Indexed passages", len(store["records"]))
c2.metric("PDF pages", store["pages"])
c3.metric("Source hash", store["source_hash"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a question about Pakistan cyber law / PECA..."
)

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching relevant PECA sections..."):
        results = retrieve(
            store,
            question,
            top_k=top_k,
            min_score=min_score,
        )

    if use_history:
        history_for_model = st.session_state.messages
    else:
        history_for_model = []

    with st.chat_message("assistant"):
        with st.spinner("Generating source-grounded answer..."):
            try:
                answer = answer_question(
                    question=question,
                    results=results,
                    language=language,
                    technicality=technicality,
                    response_size=response_size,
                    category=category,
                    reasoning_effort=reasoning_effort,
                    conversation=history_for_model,
                )
                st.markdown(answer)
            except Exception as exc:
                st.error(str(exc))
                answer = "I could not generate an answer because the Groq request failed."

        if show_sources:
            with st.expander("🔎 Retrieved PECA evidence"):
                if results:
                    for item in results:
                        st.markdown(
                            f"**{item['section']} — PDF page {item['page']} "
                            f"— similarity {item['score']:.3f}**"
                        )
                        st.write(item["text"])
                else:
                    st.warning(
                        "No passage crossed the similarity threshold. "
                        "Try lowering 'Minimum similarity' or rephrasing the question."
                    )

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.messages:
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.divider()
st.caption(
    "Source-grounded educational tool. Verify current Pakistani law and obtain "
    "professional legal advice for real cases."
)
