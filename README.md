# 📄 RAG Pipeline POC - PDF Document Analysis

A complete, production-ready RAG (Retrieval-Augmented Generation) pipeline that enables intelligent Q&A, summarization, and analysis of PDF documents.

## 🎯 Features

- **Streamlit Web UI** - Interactive web interface for uploading handbooks/PDFs and chatting
- **Smart PDF Loading** - Automatically loads and processes PDFs
- **Intelligent Chunking** - Splits documents into optimal-sized chunks with overlap
- **Vector Embeddings** - Uses HuggingFace embeddings (sentence-transformers)
- **Local Vector DB** - Chroma for fast similarity search
- **Google Gemini LLM** - Powered by Google Gemini for intelligent responses
- **Interactive CLI & Web Chat** - Ask questions in real-time with source citations
- **Batch Processing** - Process multiple questions at once
- **Source Attribution** - Know which parts of the document support answers

## 🏗️ Architecture

```
PDF File
   ↓
Load & Extract Text (PyPDF)
   ↓
Split into Chunks (RecursiveCharacterTextSplitter)
   ↓
Create Embeddings (HuggingFace Transformers)
   ↓
Store in Vector DB (Chroma)
   ↓
User Query
   ↓
Semantic Search (find relevant chunks)
   ↓
Send to Claude LLM with Context
   ↓
Return Intelligent Answer + Sources
```

## 📋 Prerequisites

- Python 3.8+
- Anthropic API Key (get from https://console.anthropic.com/)
- ~2GB disk space (for embeddings model)

## 🚀 Quick Start

### 1. Clone/Setup

```bash
# Create project directory
mkdir rag-pipeline
cd rag-pipeline

# Copy files
# (Place the following files in this directory:
#  - rag_pipeline.py
#  - example_usage.py
#  - requirements.txt
#  - .env.example
#  - Chitrakavi_Lakshmi_Anuhya_-_Offer_Letter.pdf
# )
```

### 2. Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

⏳ **First-time setup note**: The first run will download the HuggingFace embeddings model (~400MB). This happens once and is cached.

### 3. Setup API Key

Create a `.env` file in your project directory:

```bash
# Copy the example
cp .env.example .env

# Edit .env and add your Anthropic API key
nano .env  # or use your favorite editor
```

Or set it as an environment variable:

```bash
# Linux/Mac
export ANTHROPIC_API_KEY='sk-ant-xxxxxxxxxxxxx'

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY='sk-ant-xxxxxxxxxxxxx'

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
```

### 4. Run It!

**Interactive Mode** (chat with your PDF):

```bash
python rag_pipeline.py
```

Then start asking questions:
```
🔍 Your query: What is the salary structure?
⏳ Processing your query...
✅ Answer: The offer provides a CTC of 2.4 Lakhs per annum...

🔍 Your query: summarize
📝 Generating summary...
```

**Batch Mode** (ask multiple questions):

```bash
python rag_pipeline.py "What is the CTC?" "When is the join date?" "What benefits are provided?"
```

**Run Examples**:

```bash
python example_usage.py
```

## 📝 Usage Examples

### Example 1: Simple Question

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline("Chitrakavi_Lakshmi_Anuhya_-_Offer_Letter.pdf")

result = rag.query("What is the probation period?")
print(result["answer"])
# Output: The employee will be under 3 months probation period...
```

### Example 2: Get Summary

```python
summary = rag.summarize()
print(summary)
# Comprehensive summary of the entire document
```

### Example 3: Multiple Questions

```python
questions = [
    "What salary does the employee get?",
    "What leaves are available?",
    "What is the service bond?",
]

results = rag.batch_query(questions)
for r in results:
    print(f"Q: {r['question']}\nA: {r['answer']}\n")
```

### Example 4: With Source Attribution

```python
result = rag.query("What benefits are offered?")

print("Answer:")
print(result["answer"])

print("\nSources:")
for doc in result["sources"]:
    page = doc.metadata.get("page", "N/A")
    print(f"  - Page {page}")
```

## 🔧 Configuration

Edit these variables in `rag_pipeline.py` to customize:

```python
PDF_PATH = "Chitrakavi_Lakshmi_Anuhya_-_Offer_Letter.pdf"  # Your PDF file
CHROMA_DB_PATH = "./chroma_db"                              # Vector DB location
CHUNK_SIZE = 500                                             # Characters per chunk
CHUNK_OVERLAP = 50                                           # Overlap between chunks
```

### Advanced: Change Embedding Model

```python
# In rag_pipeline.py, line ~88
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Larger, more accurate
)
```

Available models:
- `all-MiniLM-L6-v2` (default, fast, 22MB) ✨
- `all-mpnet-base-v2` (more accurate, 438MB)
- `multi-qa-mpnet-base-dot-v1` (best for Q&A, 438MB)

## 📊 How It Works

### Step-by-Step

1. **Load PDF**: Extracts text from all pages
2. **Chunk**: Splits into ~500 character pieces with overlap
3. **Embed**: Converts chunks to 384-dimensional vectors using HuggingFace
4. **Store**: Saves vectors in local Chroma database
5. **Query**: When user asks, finds 3 most relevant chunks
6. **Generate**: Sends relevant chunks + question to Claude
7. **Answer**: Returns intelligent response with source references

### Why RAG?

- ✅ Uses your specific document content
- ✅ Cites where answers come from
- ✅ Reduces hallucination (no making up info)
- ✅ Faster than fine-tuning
- ✅ Works with local embeddings (no API cost)

## 🎓 Understanding the Code

### Main Components

| Component | Purpose | Library |
|-----------|---------|---------|
| PDF Loader | Extract text from PDF | PyPDF |
| Text Splitter | Break into chunks | LangChain |
| Embeddings | Convert to vectors | HuggingFace |
| Vector Store | Semantic search | Chroma |
| LLM | Generate answers | Anthropic (Claude) |
| QA Chain | Orchestrate everything | LangChain |

### Key Methods

```python
# Initialize
rag = RAGPipeline("file.pdf")

# Query with sources
result = rag.query("Your question?")
result["answer"]        # The answer text
result["sources"]       # Source documents

# Summarize entire document
rag.summarize()

# Multiple questions
results = rag.batch_query(["Q1?", "Q2?"])
```

## 🐛 Troubleshooting

### Error: "ANTHROPIC_API_KEY not found"
**Solution**: Set your API key in `.env` file or environment variable
```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

### Error: "PDF not found"
**Solution**: Ensure the PDF file is in the same directory as the script
```bash
ls Chitrakavi_Lakshmi_Anuhya_-_Offer_Letter.pdf
```

### Slow First Run
**Solution**: Normal! HuggingFace downloads embedding model (~400MB) first time only

### Memory Issues
**Solution**: Reduce `CHUNK_SIZE` or use a smaller embedding model
```python
CHUNK_SIZE = 300  # Smaller chunks use less memory
```

### Poor Answer Quality
**Solution**: Adjust search parameters in `_create_qa_chain()`:
```python
retriever=self.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Get more context (default: 3)
)
```

## 📈 Performance Tips

| Optimization | Impact | Implementation |
|--------------|--------|-----------------|
| Increase search results (k) | Better answers | Change `k` from 3 to 5 |
| Better embedding model | Smarter retrieval | Use `all-mpnet-base-v2` |
| Smaller chunks | More precise | Set `CHUNK_SIZE = 300` |
| Larger chunks | More context | Set `CHUNK_SIZE = 1000` |

## 🔐 Cost Optimization

- **Embeddings**: FREE (local HuggingFace model, runs on your machine)
- **Vector DB**: FREE (Chroma runs locally)
- **LLM Queries**: You pay per query to Anthropic
  - ~$0.01 per query for this small document
  - Use batch mode to combine questions

## 📚 What You Learn

- How RAG pipelines work end-to-end
- Vector embeddings and semantic search
- LangChain framework (most popular RAG library)
- Chroma vector database
- Prompt engineering for Q&A
- Integration with Claude API

## 🚀 Next Steps / Enhancements

1. **Multiple PDFs**: Extend to handle document collections
2. **Web Interface**: Add FastAPI or Streamlit UI
3. **Persistence**: Save and load existing vector stores
4. **Streaming**: Stream responses for better UX
5. **Caching**: Cache similar queries
6. **Different LLMs**: Swap Claude for local Ollama

## 💡 Example Prompts for Your PDF

Try these questions:
- "Summarize the offer letter"
- "What is the CTC?"
- "When should I join?"
- "What are the benefits?"
- "What is the service bond?"
- "What's included in the salary breakdown?"
- "What happens after probation?"
- "What learning opportunities are available?"

## 📞 Support

- Anthropic API Docs: https://docs.anthropic.com
- LangChain Docs: https://python.langchain.com
- Chroma Docs: https://docs.trychroma.com
- HuggingFace Docs: https://huggingface.co/docs

## 📄 License

Free to use for learning and development!

---

**Happy RAG-ing! 🚀** Build something awesome with your documents!
