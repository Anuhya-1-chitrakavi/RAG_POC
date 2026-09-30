#!/usr/bin/env python3
"""
RAG Pipeline for PDF Document Analysis
Supports Q&A, summarization, and general queries

"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import warnings

# LangChain imports
# PyPDFLoader has no standalone package yet - suppress the community deprecation warning
warnings.filterwarnings("ignore", message=".*langchain-community.*")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings          # standalone: langchain-huggingface
from langchain_chroma import Chroma                              # standalone: langchain-chroma
from langchain_google_genai import ChatGoogleGenerativeAI        # standalone: langchain-google-genai
from langchain_classic.chains import RetrievalQA                         # langchain_classic
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# ── Logger setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RAGPipeline")
# ─────────────────────────────────────────────────────────────────────────────

# Configuration
PDF_PATH = r'C:\Users\Anuhya\Downloads\Employee_handbook.pdf'
CHROMA_DB_PATH = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

import hashlib
import chromadb

def _compute_collection_id(file_paths, chunk_size, chunk_overlap):
    """Compute a deterministic hash for files and chunk configuration"""
    hasher = hashlib.sha256()
    for fp in sorted(file_paths):
        p = Path(fp)
        if p.exists():
            stat = p.stat()
            hasher.update(p.name.encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime).encode("utf-8"))
    hasher.update(str(chunk_size).encode("utf-8"))
    hasher.update(str(chunk_overlap).encode("utf-8"))
    return "col_" + hasher.hexdigest()[:24]

class RAGPipeline:
    """RAG Pipeline for document analysis (single or multiple documents)"""
    
    def __init__(
        self,
        pdf_path,
        api_key: str = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        collection_name: str = None,
        persist_directory: str = CHROMA_DB_PATH,
        model_name: str = "gemini-flash-lite-latest",
        embeddings = None,
    ):
        """
        Initialize RAG pipeline with automatic embedding cache reuse
        
        Args:
            pdf_path: Path to PDF file, or list of PDF file paths
            api_key: Gemini API key (optional, reads from GOOGLE_API_KEY env var)
            chunk_size: Chunk size for text splitting
            chunk_overlap: Chunk overlap for text splitting
            collection_name: ChromaDB collection name (defaults to content hash)
            persist_directory: Directory to persist ChromaDB
            model_name: Gemini model name (default: gemini-flash-lite-latest)
            embeddings: Optional pre-loaded embeddings instance for instant reuse
        """
        if isinstance(pdf_path, (list, tuple)):
            self.pdf_paths = [str(p) for p in pdf_path]
        else:
            self.pdf_paths = [str(pdf_path)]
        
        self.pdf_path = self.pdf_paths[0] if self.pdf_paths else ""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embeddings = embeddings
        
        # Determine deterministic collection cache name
        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = _compute_collection_id(self.pdf_paths, self.chunk_size, self.chunk_overlap)
        
        if not self.api_key:
            logger.error("API key not found. Set GOOGLE_API_KEY as an environment variable or pass it as argument.")
            raise ValueError(
                "GOOGLE_API_KEY not found. "
                "Please set it as an environment variable or pass it as argument."
            )
        
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.total_docs = len(self.pdf_paths)
        self.total_pages = 0
        self.total_chunks = 0
        self.documents = []
        
        logger.info("🚀 Initializing RAG Pipeline for %d document(s)...", self.total_docs)
        self._setup()
    
    def _setup(self):
        """Setup the complete pipeline with vector cache check"""
        # Step 1: Ensure embedding model is ready
        if self.embeddings is None:
            logger.info("🧠 Loading HuggingFace embedding model: sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("✅ Embedding model loaded")
        
        # Step 2: Check for existing cached collection in ChromaDB
        chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        existing_collections = [c.name for c in chroma_client.list_collections()]
        
        # Load raw PDF pages for document summaries & citations
        self.documents = self._load_pdf()
        self.total_pages = len(self.documents)
        
        if self.collection_name in existing_collections:
            coll = chroma_client.get_collection(self.collection_name)
            if coll.count() > 0:
                self.total_chunks = coll.count()
                logger.info("⚡ CACHE HIT: Reusing %d cached embeddings in ChromaDB collection '%s'. Zero re-embedding required!", self.total_chunks, self.collection_name)
                self.vectorstore = Chroma(
                    client=chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                )
            else:
                self._build_vectorstore(chroma_client)
        else:
            self._build_vectorstore(chroma_client)
        
        # Step 3: Setup LLM
        logger.info("Initializing Google Gemini LLM (model=%s)", self.model_name)
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model=self.model_name,
            temperature=0.7,
        )
        
        # Step 4: Create QA chain
        self._create_qa_chain()
        logger.info("🎉 Pipeline fully initialized and ready!\n")
    
    def _build_vectorstore(self, chroma_client):
        """Split documents and compute embeddings into ChromaDB"""
        logger.info("✂️  Splitting into chunks (size=%d, overlap=%d)", self.chunk_size, self.chunk_overlap)
        chunks = self._split_documents(self.documents)
        self.total_chunks = len(chunks)
        logger.info("✅ Created %d chunks", self.total_chunks)
        
        logger.info("💾 Generating embeddings and saving to ChromaDB collection '%s'...", self.collection_name)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=chroma_client,
            collection_name=self.collection_name
        )
        logger.info("✅ Embeddings stored in cache")
    
    def _load_pdf(self):
        """Load one or multiple PDF documents"""
        all_documents = []
        for file_path in self.pdf_paths:
            p = Path(file_path)
            logger.debug("Checking if PDF exists at path: %s", file_path)
            if not p.exists():
                logger.error("PDF file not found: %s", file_path)
                raise FileNotFoundError(f"PDF not found: {file_path}")
            
            loader = PyPDFLoader(str(p))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_name"] = p.name
            all_documents.extend(docs)
        return all_documents
    
    def _split_documents(self, documents):
        """Split documents into chunks"""
        logger.debug("Splitting %d document(s) using RecursiveCharacterTextSplitter", len(documents))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        return chunks
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt"""
        logger.debug("Building custom PromptTemplate for QA chain")
        prompt_template = """You are an intelligent document analysis assistant. Answer the user's question accurately based on the provided context.

GUIDELINES:
- Provide clear, direct, and well-synthesized answers in your own words.
- Do NOT simply copy-paste, dump, or blindly repeat raw lists and text blocks from the document.
- Summarize, categorize, and explain the relevant points concisely.
- If the information cannot be found in the provided context, clearly state so.

Context:
{context}

Question: {question}

Helpful Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        logger.debug("PromptTemplate created with input_variables: ['context', 'question']")
        
        logger.debug("Creating RetrievalQA chain (chain_type=stuff, retriever k=6)")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}  # Get top 6 relevant chunks
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        logger.debug("RetrievalQA chain created successfully")
    
    def query(self, question: str) -> dict:
        """
        Query the RAG pipeline
        
        Args:
            question: User query
        
        Returns:
            Dictionary with answer and source documents
        """
        clean_q = question.strip().lower().rstrip("?.!")
        summary_triggers = ("summarize", "summary", "overview", "give me a brief", "brief of", "key takeaways", "wrap up", "synopsis")
        # If user is asking for a summary/overview in chat, route to executive summarizer
        if any(trigger in clean_q for trigger in summary_triggers):
            logger.info("Chat query recognized as document summary request ('%s'). Invoking summarize()...", question)
            summary_ans = self.summarize()
            return {
                "answer": summary_ans,
                "sources": self.documents[:6] if hasattr(self, "documents") else []
            }

        logger.info("Received query: %s", question)
        logger.debug("Invoking QA chain with query...")
        result = self.qa_chain.invoke({"query": question})
        logger.info("Query answered. Source documents returned: %d", len(result.get("source_documents", [])))
        return {
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }
    
    def summarize(self) -> str:
        """Generate a comprehensive, analytical executive summary of the entire document(s)"""
        logger.info("Generating full-document executive summary...")
        
        # Aggregate text from all loaded pages
        doc_texts = []
        for doc in self.documents:
            src = doc.metadata.get("source_name", "Document")
            page = doc.metadata.get("page", 0)
            doc_texts.append(f"[{src} - Page {page}]\n{doc.page_content.strip()}")
        
        combined_text = "\n\n".join(doc_texts)
        
        # Limit to safe context length (~250,000 characters if huge)
        if len(combined_text) > 250000:
            combined_text = combined_text[:250000] + "\n...[Additional pages omitted for summary brevity]"
        
        summary_instruction = f"""You are an elite AI analyst. Read the entire provided document content and write a comprehensive, fluid, and intelligent executive summary.

STYLE GUIDELINES:
1. Opening Overview: Write a clear, engaging 2-3 sentence narrative introducing who or what the document is about, their core role/purpose, and background.
2. Thematic Groupings: Logically group key details (e.g. Core Skills / Technical Competencies, Key Projects & Accomplishments, Policies / Guidelines, Education & Certifications).
3. Narrative Insights for Experience / Projects: For each major project or section, do NOT just list bullets. Write a concise narrative explaining WHAT was built/done, HOW it was accomplished, and the IMPACT / core tech used (e.g., "Built the response-generation component producing validated outputs using Pydantic AI and Bedrock LLMs, and reduced redundant calls via Redis caching").
4. Executive Assessment (Overall): Conclude with a strong, thoughtful 2-3 sentence overall takeaway highlighting key strengths, capabilities, and ideal fit / significance.

DOCUMENT CONTENT:
{combined_text}
"""
        try:
            response = self.llm.invoke(summary_instruction)
            summary_content = response.content if hasattr(response, "content") else str(response)
            
            # If response is in list format (e.g. from langchain response parts), extract text
            if isinstance(summary_content, list):
                summary_content = "\n".join(
                    part.get("text", "") if isinstance(part, dict) else str(part) 
                    for part in summary_content
                )
            
            logger.info("Executive summary generated successfully")
            return summary_content
        except Exception as e:
            logger.error("Error generating executive summary: %s", str(e), exc_info=True)
            fallback = self.query("Provide an intelligent executive summary of the document without copying raw text.")
            return fallback["answer"]
    
    def batch_query(self, questions: list) -> list:
        """
        Process multiple questions
        
        Args:
            questions: List of questions
        
        Returns:
            List of results
        """
        logger.info("Starting batch query: %d question(s) to process", len(questions))
        results = []
        for i, question in enumerate(questions, 1):
            logger.info("📌 Batch Question %d/%d: %s", i, len(questions), question)
            try:
                result = self.query(question)
                results.append({
                    "question": question,
                    "answer": result["answer"]
                })
                logger.info("Question %d/%d answered successfully", i, len(questions))
            except Exception as e:
                logger.error("Error processing question %d/%d: %s", i, len(questions), str(e), exc_info=True)
                results.append({
                    "question": question,
                    "answer": f"Error: {str(e)}"
                })
        logger.info("Batch query complete. %d/%d questions answered", len(results), len(questions))
        return results


def interactive_mode(rag: RAGPipeline):
    """Interactive CLI mode"""
    logger.info("Entering interactive mode")
    print("="*60)
    print("RAG Pipeline - Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  'summarize' - Get a summary of the document")
    print("  'quit'      - Exit the program")
    print("  Or ask any question about the document\n")
    
    while True:
        try:
            user_input = input("\nYour query: ").strip()
            
            if not user_input:
                logger.debug("Empty input received, skipping")
                continue
            
            if user_input.lower() == "quit":
                logger.info("User requested exit from interactive mode")
                print("\nGoodbye!")
                break
            
            if user_input.lower() == "summarize":
                logger.info("User requested document summary")
                print("\nGenerating summary...")
                summary = rag.summarize()
                print(f"\n{summary}")
            else:
                logger.info("User query: %s", user_input)
                print("\nProcessing your query...")
                result = rag.query(user_input)
                
                print(f"\nAnswer:\n{result['answer']}")
                
                # Show sources
                if result["sources"]:
                    logger.debug("Displaying %d source document(s)", len(result["sources"]))
                    print("\nSource Documents:")
                    for i, doc in enumerate(result["sources"], 1):
                        print(f"   [{i}] Page {doc.metadata.get('page', 'N/A')}")
        
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Exiting interactive mode.")
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error("Unexpected error in interactive mode: %s", str(e), exc_info=True)
            print(f"\nError: {str(e)}")



def main():
    """Main entry point"""
    import sys
    
    logger.info("RAG Pipeline starting up")
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY environment variable is not set. Cannot proceed.")
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("\nTo set it:")
        print("  Get your free key at: https://aistudio.google.com/app/apikey")
        print("  Then add to .env file: GOOGLE_API_KEY=your-key-here")
        sys.exit(1)
    
    # Initialize pipeline
    logger.info("Initializing RAGPipeline with PDF: %s", PDF_PATH)
    rag = RAGPipeline(PDF_PATH)
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        # Batch mode: pass questions as arguments
        questions = sys.argv[1:]
        logger.info("Batch mode: %d question(s) received via command line", len(questions))
        print(f"Processing {len(questions)} question(s)...\n")
        results = rag.batch_query(questions)
        
        # Print results
        for result in results:
            print("\n" + "="*60)
            print(f"Q: {result['question']}")
            print(f"A: {result['answer']}")
        logger.info("Batch mode complete. Results printed.")
    else:
        # Interactive mode
        logger.info("No CLI arguments provided. Launching interactive mode.")
        interactive_mode(rag)


if __name__ == "__main__":
    main()
