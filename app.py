import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .doc-pill {
        background-color: #e2e8f0;
        border-radius: 12px;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin: 2px 4px 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper directory for uploads
UPLOAD_DIR = Path("./uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)


def init_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "current_files" not in st.session_state:
        st.session_state.current_files = []
    if "summary_text" not in st.session_state:
        st.session_state.summary_text = None
    if "file_stats" not in st.session_state:
        st.session_state.file_stats = {"docs": 0, "pages": 0, "chunks": 0}


from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource(show_spinner="🧠 Loading embedding model into memory...")
def get_cached_embeddings():
    """Load and cache HuggingFace embeddings model in memory once"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

init_session_state()


def load_pipeline(file_paths: list, api_key: str, chunk_size: int, chunk_overlap: int, model_name: str):
    """Load and index multiple PDFs into RAGPipeline with instant cache reuse"""
    num_docs = len(file_paths)
    doc_label = f"{num_docs} document{'s' if num_docs > 1 else ''}"
    
    with st.spinner(f"📄 Checking vector cache & indexing {doc_label}..."):
        try:
            cached_embeds = get_cached_embeddings()
            rag = RAGPipeline(
                pdf_path=file_paths,
                api_key=api_key,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                model_name=model_name,
                embeddings=cached_embeds,
            )
            st.session_state.rag_pipeline = rag
            st.session_state.current_files = [Path(p).name for p in file_paths]
            st.session_state.file_stats = {
                "docs": rag.total_docs,
                "pages": rag.total_pages,
                "chunks": rag.total_chunks
            }
            st.session_state.messages = []  # reset conversation for new docs
            st.session_state.summary_text = None
            st.success(f"✅ Ready! Indexed **{rag.total_docs} document(s)** ({rag.total_pages} total pages, {rag.total_chunks} chunks)")
        except Exception as e:
            st.error(f"❌ Failed to load documents: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Handling
    env_api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_api_key,
        type="password",
        help="Reads from GOOGLE_API_KEY environment variable by default or paste your key here."
    )
    api_key = api_key_input.strip() if api_key_input else env_api_key

    st.divider()

    # Multi-Document Upload Section
    st.subheader("📁 Document / PDF Upload")
    upload_method = st.radio(
        "Select Upload Method",
        ["Upload Files (Browser)", "Local File Path(s)"],
        index=0
    )

    selected_file_paths = []
    trigger_load = False

    if upload_method == "Upload Files (Browser)":
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or multiple PDF documents to analyze together"
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = UPLOAD_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                selected_file_paths.append(str(save_path))
            
            uploaded_names = [f.name for f in uploaded_files]
            if st.session_state.current_files != uploaded_names:
                btn_label = f"🚀 Process & Index {len(uploaded_files)} File{'s' if len(uploaded_files) > 1 else ''}"
                trigger_load = st.button(btn_label, type="primary", use_container_width=True)
    else:
        default_local_path = r"C:\Users\Anuhya\Downloads\Employee_handbook.pdf"
        local_path_input = st.text_area(
            "Local PDF Path(s)",
            value=default_local_path,
            help="Enter one or more absolute paths (one per line) to PDF files on your computer",
            height=85
        )
        if local_path_input:
            lines = [line.strip() for line in local_path_input.splitlines() if line.strip()]
            for p_str in lines:
                if Path(p_str).exists():
                    selected_file_paths.append(p_str)
            
            st.caption(f"✅ Found {len(selected_file_paths)} / {len(lines)} valid file(s) on disk")
            if selected_file_paths:
                btn_label = f"🚀 Load {len(selected_file_paths)} Local File{'s' if len(selected_file_paths) > 1 else ''}"
                trigger_load = st.button(btn_label, type="primary", use_container_width=True)

    st.divider()

    # Advanced Settings Expandable
    with st.expander("🛠️ Advanced Model & Chunk Settings"):
        model_name = st.selectbox(
            "Gemini Model",
            options=["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-latest"],
            index=0
        )
        chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=1000, step=50)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=150, step=10)

    # Document Status
    st.divider()
    st.subheader("📊 Document Status")
    if st.session_state.rag_pipeline:
        st.success(f"🟢 **{st.session_state.file_stats['docs']} Document(s) Active**")
        with st.expander("📄 View Active Files", expanded=False):
            for fname in st.session_state.current_files:
                st.markdown(f"- `{fname}`")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files", st.session_state.file_stats["docs"])
        with col2:
            st.metric("Pages", st.session_state.file_stats["pages"])
        with col3:
            st.metric("Chunks", st.session_state.file_stats["chunks"])
        
        if st.button("🗑️ Reset Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No documents loaded yet. Please upload one or more PDFs.")

# Process Documents if triggered
if trigger_load and selected_file_paths:
    if not api_key:
        st.error("❌ Please provide a Google Gemini API Key in the sidebar or in .env")
    else:
        load_pipeline(selected_file_paths, api_key, chunk_size, chunk_overlap, model_name)


# ─────────────────────────────────────────────────────────────────────────────
# Main Interface
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📚 RAG Multi-Document Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload one or multiple PDF documents and ask questions across all of them with AI-powered semantic search & source citations.</div>', unsafe_allow_html=True)

if not st.session_state.rag_pipeline:
    st.info("👈 **Get Started**: Upload one or more PDF documents from the sidebar on the left and click **Process & Index**.")
    st.stop()

# Display active document tags
st.markdown("**Active Knowledge Base:** " + " ".join([f'<span class="doc-pill">📄 {f}</span>' for f in st.session_state.current_files]), unsafe_allow_html=True)
st.markdown("")

# Tabs for Chat and Summary
tab_chat, tab_summary = st.tabs(["💬 Ask Questions", "📑 Document Summary"])

# ── Tab 1: Chat Interface ─────────────────────────────────────────────────────
with tab_chat:
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            st.info("💡 Type your question in the box at the bottom to start asking questions about your document(s).")
        
        # Display Chat History chronologically (oldest at top, newest at bottom)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander(f"🔍 Source Citations ({len(msg['sources'])} chunks)"):
                        for idx, doc in enumerate(msg["sources"], 1):
                            source_file = doc.metadata.get("source_name", Path(doc.metadata.get("source", "Document")).name)
                            page_num = doc.metadata.get("page", "Unknown")
                            st.markdown(f"**Source {idx} — 📄 `{source_file}` (Page {page_num})**")
                            st.markdown(f"> {doc.page_content.strip()}")
                            st.divider()

# ── Tab 2: Document Summary ───────────────────────────────────────────────────
with tab_summary:
    num_docs = len(st.session_state.current_files)
    st.subheader(f"📑 Summary for {num_docs} Document{'s' if num_docs > 1 else ''}")
    
    if st.session_state.summary_text:
        st.markdown(st.session_state.summary_text)
        if st.button("🔄 Regenerate Summary"):
            with st.spinner("Generating comprehensive summary across all documents..."):
                st.session_state.summary_text = st.session_state.rag_pipeline.summarize()
                st.rerun()
    else:
        st.write("Generate an executive summary covering key sections, numbers, requirements, and policies across all loaded documents.")
        if st.button("✨ Generate Document Summary", type="primary"):
            with st.spinner("Analyzing all context and generating summary..."):
                try:
                    summary = st.session_state.rag_pipeline.summarize()
                    st.session_state.summary_text = summary
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")

# ── Floating Fixed Chat Input at Root Bottom ─────────────────────────────────
user_query = st.chat_input("Ask any question across your uploaded documents...")

if user_query:
    # 1. Append User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. Generate Assistant Response
    with tab_chat:
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and formulating answer..."):
                try:
                    result = st.session_state.rag_pipeline.query(user_query)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander(f"🔍 Source Citations ({len(sources)} chunks)"):
                            for idx, doc in enumerate(sources, 1):
                                source_file = doc.metadata.get("source_name", Path(doc.metadata.get("source", "Document")).name)
                                page_num = doc.metadata.get("page", "Unknown")
                                st.markdown(f"**Source {idx} — 📄 `{source_file}` (Page {page_num})**")
                                st.markdown(f"> {doc.page_content.strip()}")
                                st.divider()
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    error_msg = f"⚠️ Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.rerun()
