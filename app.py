import os
import tempfile
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

# Custom CSS styling for a modern, clean look
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
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .source-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-top: 8px;
        font-size: 0.9rem;
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
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "summary_text" not in st.session_state:
        st.session_state.summary_text = None
    if "file_stats" not in st.session_state:
        st.session_state.file_stats = {"pages": 0, "chunks": 0}


init_session_state()


def load_pipeline(file_path: str, api_key: str, chunk_size: int, chunk_overlap: int, model_name: str):
    """Load and index PDF into RAGPipeline"""
    with st.spinner("📄 Reading document, generating embeddings, and building vector index..."):
        try:
            rag = RAGPipeline(
                pdf_path=file_path,
                api_key=api_key,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                model_name=model_name
            )
            st.session_state.rag_pipeline = rag
            st.session_state.current_file = Path(file_path).name
            st.session_state.file_stats = {
                "pages": rag.total_pages,
                "chunks": rag.total_chunks
            }
            st.session_state.messages = []  # reset conversation for new doc
            st.session_state.summary_text = None
            st.success(f" Successfully processed **{Path(file_path).name}** ({rag.total_pages} pages, {rag.total_chunks} chunks)")
        except Exception as e:
            st.error(f"❌ Failed to load document: {str(e)}")


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

    # Upload Section
    st.subheader("📁 Handbook / PDF Upload")
    upload_method = st.radio(
        "Select Upload Method",
        ["Upload File (Browser)", "Local File Path"],
        index=0
    )

    selected_file_path = None
    trigger_load = False

    if upload_method == "Upload File (Browser)":
        uploaded_file = st.file_uploader(
            "Upload your Handbook (PDF)",
            type=["pdf"],
            help="Upload an employee handbook or any PDF document to analyze"
        )
        if uploaded_file is not None:
            save_path = UPLOAD_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_file_path = str(save_path)
            
            # Button to process if not yet processed
            if st.session_state.current_file != uploaded_file.name:
                trigger_load = st.button("🚀 Process & Index Handbook", type="primary", use_container_width=True)
    else:
        default_local_path = r"C:\Users\Anuhya\Downloads\Employee_handbook.pdf"
        local_path_input = st.text_input(
            "Local PDF Path",
            value=default_local_path,
            help="Absolute path to the handbook PDF on your computer"
        )
        if local_path_input:
            selected_file_path = local_path_input.strip()
            if Path(selected_file_path).exists():
                st.caption(f"✅ Found file on disk ({round(os.path.getsize(selected_file_path) / 1024, 1)} KB)")
            else:
                st.caption("⚠️ File not found at this path")
            trigger_load = st.button("🚀 Load Local Handbook", type="primary", use_container_width=True)

    st.divider()

    # Advanced Settings Expandable
    with st.expander("🛠️ Advanced Model & Chunk Settings"):
        model_name = st.selectbox(
            "Gemini Model",
            options=["gemini-flash-latest", "gemini-pro-latest", "gemini-2.5-flash-lite"],
            index=0
        )
        chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=500, step=50)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)

    # Document Status
    st.divider()
    st.subheader("📊 Document Status")
    if st.session_state.rag_pipeline:
        st.success(f"🟢 Active: **{st.session_state.current_file}**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pages", st.session_state.file_stats["pages"])
        with col2:
            st.metric("Chunks", st.session_state.file_stats["chunks"])
        
        if st.button("🗑️ Reset Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No document loaded yet. Please upload or specify a handbook PDF.")

# Process Document if triggered
if trigger_load and selected_file_path:
    if not api_key:
        st.error("❌ Please provide a Google Gemini API Key in the sidebar or in .env")
    elif not Path(selected_file_path).exists():
        st.error(f"❌ File does not exist: {selected_file_path}")
    else:
        load_pipeline(selected_file_path, api_key, chunk_size, chunk_overlap, model_name)


# ─────────────────────────────────────────────────────────────────────────────
# Main Interface
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📚 Handbook Q&A and Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload your employee handbook or company document and ask questions with AI-powered retrieval & citations.</div>', unsafe_allow_html=True)

if not st.session_state.rag_pipeline:
    st.info("👈 **Get Started**: Upload your Handbook PDF from the sidebar on the left and click **Process & Index Handbook**.")
    
    # Preview sample questions
    st.markdown("### 💡 What you can ask once loaded:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📋 Policies & Guidelines**
        - What are the working hours and attendance policies?
        - What is the code of conduct?
        """)
    with col2:
        st.markdown("""
        **🏖️ Leaves & Benefits**
        - How many annual leaves or sick leaves are allowed?
        - What health insurance benefits are provided?
        """)
    with col3:
        st.markdown("""
        **⚖️ Terms & Roles**
        - What is the notice period and probation duration?
        - What are the remote work / WFH guidelines?
        """)
    st.stop()

# Tabs for Chat, Summary, and Help
tab_chat, tab_summary = st.tabs(["💬 Ask Questions", "📑 Document Summary"])

# ── Tab 1: Chat Interface ─────────────────────────────────────────────────────
with tab_chat:
    # Quick prompt buttons
    st.markdown("##### ⚡ Quick Queries")
    quick_cols = st.columns(4)
    quick_prompts = [
        "What are the leave policies?",
        "What are the working hours & attendance rules?",
        "What are the employee benefits and perks?",
        "What is the probation and notice period?",
    ]
    
    chosen_prompt = None
    for i, prompt_text in enumerate(quick_prompts):
        with quick_cols[i]:
            if st.button(prompt_text, key=f"quick_{i}", use_container_width=True):
                chosen_prompt = prompt_text

    st.markdown("---")

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"🔍 Source Citations ({len(msg['sources'])} chunks)"):
                    for idx, doc in enumerate(msg["sources"], 1):
                        page_num = doc.metadata.get("page", "Unknown")
                        st.markdown(f"**Source {idx} — Page {page_num}**")
                        st.markdown(f"> {doc.page_content.strip()}")
                        st.divider()

    # Handle user input from chat input or quick prompt
    user_query = st.chat_input("Ask any question about your handbook...")
    if chosen_prompt:
        user_query = chosen_prompt

    if user_query:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching document and formulating answer..."):
                try:
                    result = st.session_state.rag_pipeline.query(user_query)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander(f"🔍 Source Citations ({len(sources)} chunks)"):
                            for idx, doc in enumerate(sources, 1):
                                page_num = doc.metadata.get("page", "Unknown")
                                st.markdown(f"**Source {idx} — Page {page_num}**")
                                st.markdown(f"> {doc.page_content.strip()}")
                                st.divider()
                    
                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    error_msg = f"⚠️ Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ── Tab 2: Document Summary ───────────────────────────────────────────────────
with tab_summary:
    st.subheader(f"📑 Document Summary: {st.session_state.current_file}")
    
    if st.session_state.summary_text:
        st.markdown(st.session_state.summary_text)
        if st.button("🔄 Regenerate Summary"):
            with st.spinner("Generating comprehensive summary..."):
                st.session_state.summary_text = st.session_state.rag_pipeline.summarize()
                st.rerun()
    else:
        st.write("Generate an executive summary covering key sections, policies, numbers, and requirements mentioned in the handbook.")
        if st.button("✨ Generate Handbook Summary", type="primary"):
            with st.spinner("Reading full context and generating summary..."):
                try:
                    summary = st.session_state.rag_pipeline.summarize()
                    st.session_state.summary_text = summary
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
