# Pakistan Cyber Law RAG Assistant

A free-to-run Retrieval-Augmented Generation (RAG) application for asking questions about Pakistan's **Prevention of Electronic Crimes Act (PECA)**.

The app uses:

- **Python**
- **Streamlit**
- **FAISS** for vector search
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for free local embeddings
- **PyMuPDF** for PDF extraction
- **Groq API**
- **`openai/gpt-oss-20b`** for answer generation

Only three project files are required:

```text
app.py
requirements.txt
readme.md
```

## How the RAG pipeline works

```text
PECA PDF URL
     ↓
Download PDF on startup
     ↓
PyMuPDF text extraction
     ↓
Page-aware chunking
     ↓
Sentence Transformer embeddings
     ↓
FAISS IndexFlatIP
     ↓
User question
     ↓
Question embedding
     ↓
Top relevant PECA passages
     ↓
Groq GPT-OSS 20B
     ↓
Source-grounded answer + section/page references
```

The PDF is downloaded and the FAISS index is built automatically when the app starts.

## 1. Get a Groq API key

Create a Groq API key from your Groq account.

Do **not** put the API key directly inside `app.py`.

The app reads:

```text
GROQ_API_KEY
```

## 2. Run locally

Create a folder containing:

```text
app.py
requirements.txt
readme.md
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Groq key.

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

### Linux / macOS

```bash
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Run:

```bash
streamlit run app.py
```

## 3. Google Colab

This project does not require a paid service for embeddings.

In Colab:

```python
!pip install -r requirements.txt
```

Then:

```python
!streamlit run app.py
```

If you want to open the Streamlit interface publicly from Colab, use a tunneling service separately. The application itself does not require a paid tunnel.

## 4. Streamlit Community Cloud

Push these three files to GitHub:

```text
app.py
requirements.txt
readme.md
```

Then create a Streamlit Community Cloud app using `app.py` as the main file.

Add your API key under:

```text
Settings → Secrets
```

Use:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Optional: if you want to use a different PECA PDF URL:

```toml
PECA_PDF_URL = "https://example.com/your-peca.pdf"
```

If `PECA_PDF_URL` is not supplied, the app uses the default Pakistan Code PDF configured in `app.py`.

## 5. Source PDF

The default source is the Prevention of Electronic Crimes Act, 2016 PDF published through Pakistan Code.

You can change the PDF URL from the sidebar or with the `PECA_PDF_URL` secret.

For a deployment based exactly on your own supplied PDF, host that PDF at a stable HTTPS URL and set:

```toml
PECA_PDF_URL = "YOUR_PDF_URL"
```

Then press **Rebuild PDF index** in the sidebar.

## 6. UI controls

The application includes:

### Response language

- English
- Roman Urdu
- Urdu

### Technicality

- Beginner
- Intermediate
- Advanced
- Law-student

### Response size

- Short
- Medium
- Detailed
- Very detailed

### Question category

- General
- Offence / Section
- Punishment / Fine
- Cybercrime scenario
- Investigation / Procedure
- Evidence / Search / Seizure
- Rights / Remedies
- Other

### Retrieval controls

- Number of retrieved passages
- Minimum similarity threshold

### GPT-OSS reasoning

- Low
- Medium
- High

### Other options

- Show retrieved PECA evidence
- Use recent conversation context
- Rebuild the PDF/FAISS index
- Clear conversation

## 7. Important RAG behavior

This application is intentionally designed to reduce hallucination.

The model is instructed to:

1. Use the retrieved PECA passages as the primary source.
2. Cite the relevant section and PDF page when supported.
3. Never invent a section number or punishment.
4. Say when the supplied source does not contain enough information.
5. Avoid presenting an answer as a guaranteed court/legal outcome.
6. Redirect harmful cybercrime how-to requests toward the relevant legal provision rather than providing instructions for wrongdoing.

## 8. Free components

The following parts can run without paid infrastructure:

- Streamlit
- FAISS
- Sentence Transformers
- PyMuPDF
- NumPy
- Requests

Groq API usage is separate and depends on the availability/limits of your Groq account. The selected model is:

```text
openai/gpt-oss-20b
```

## 9. Rebuilding embeddings

The app caches the downloaded PDF and FAISS resources during the Streamlit session.

If the PDF changes, use:

```text
Sidebar → Rebuild PDF index
```

This clears the cached PDF and vector store and rebuilds them.

## 10. Legal disclaimer

This is an educational RAG/retrieval application, not a law firm or legal adviser.

PECA can be amended, interpreted by courts, supplemented by rules/notifications, and interact with other Pakistani laws. A real legal matter should therefore be checked against the current applicable law and professional legal advice.

The application intentionally answers from the loaded PDF rather than pretending that the model knows every current Pakistani legal development.
