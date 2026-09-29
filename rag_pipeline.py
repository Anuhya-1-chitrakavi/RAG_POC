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
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

class RAGPipeline:
    """RAG Pipeline for document analysis"""
    
    def __init__(
        self,
        pdf_path: str,
        api_key: str = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        collection_name: str = None,
        persist_directory: str = CHROMA_DB_PATH,
        model_name: str = "gemini-flash-latest",
    ):
        """
        Initialize RAG pipeline
        
        Args:
            pdf_path: Path to PDF file
            api_key: Gemini API key (optional, reads from GOOGLE_API_KEY env var)
            chunk_size: Chunk size for text splitting
            chunk_overlap: Chunk overlap for text splitting
            collection_name: ChromaDB collection name
            persist_directory: Directory to persist ChromaDB
            model_name: Gemini model name (default: gemini-flash-latest)
        """
        self.pdf_path = pdf_path
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        # Determine collection name based on file name if not provided
        if collection_name:
            self.collection_name = collection_name
        else:
            base_name = Path(pdf_path).stem
            # Clean collection name for Chroma (alphanumeric, underscores, hyphens, 3-63 chars)
            clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name)
            clean_name = clean_name[:60] if clean_name else "doc_collection"
            if len(clean_name) < 3:
                clean_name = clean_name + "_collection"
            self.collection_name = clean_name
        
        if not self.api_key:
            logger.error("API key not found. Set GOOGLE_API_KEY as an environment variable or pass it as argument.")
            raise ValueError(
                "GOOGLE_API_KEY not found. "
                "Please set it as an environment variable or pass it as argument."
            )
        
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.total_pages = 0
        self.total_chunks = 0
        
        logger.info("🚀 Initializing RAG Pipeline...")
        self._setup()
    
    def _setup(self):
        """Setup the complete pipeline"""
        # Step 1: Load PDF
        logger.info("Step 1/6 | 📄 Loading PDF: %s", self.pdf_path)
        documents = self._load_pdf()
        self.total_pages = len(documents)
        logger.info("Step 1/6 | ✅ Loaded %d page(s) from PDF", self.total_pages)
        
        # Step 2: Split into chunks
        logger.info("Step 2/6 | ✂️  Splitting into chunks (size=%d, overlap=%d)", self.chunk_size, self.chunk_overlap)
        chunks = self._split_documents(documents)
        self.total_chunks = len(chunks)
        logger.info("Step 2/6 | ✅ Created %d chunks", self.total_chunks)
        
        # Step 3: Create embeddings
        logger.info("Step 3/6 | 🧠 Loading HuggingFace embedding model: sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Step 3/6 | ✅ Embedding model loaded successfully")
        
        # Step 4: Create vector store
        logger.info("Step 4/6 | 💾 Creating Chroma vector store at: %s (collection: %s)", self.persist_directory, self.collection_name)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        logger.info("Step 4/6 | ✅ Vector store created with collection '%s'", self.collection_name)
        
        # Step 5: Setup LLM
        logger.info("Step 5/6 | Initializing Google Gemini LLM (model=%s)", self.model_name)
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model=self.model_name,
            temperature=0.7,
        )
        logger.info("Step 5/6 | Gemini LLM initialized (temperature=0.7)")
        
        # Step 6: Create QA chain
        logger.info("Step 6/6 | 🔗 Building RetrievalQA chain")
        self._create_qa_chain()
        logger.info("Step 6/6 | ✅ QA chain ready")
        
        logger.info("🎉 Pipeline fully initialized and ready!\n")
    
    def _load_pdf(self):
        """Load PDF document"""
        logger.debug("Checking if PDF exists at path: %s", self.pdf_path)
        if not Path(self.pdf_path).exists():
            logger.error("PDF file not found: %s", self.pdf_path)
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        return documents
    
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
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, say so clearly.

Context:
{context}

Question: {question}

Provide a helpful and detailed answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        logger.debug("PromptTemplate created with input_variables: ['context', 'question']")
        
        logger.debug("Creating RetrievalQA chain (chain_type=stuff, retriever k=5)")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Get top 5 relevant chunks
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
        logger.info("Received query: %s", question)
        logger.debug("Invoking QA chain with query...")
        result = self.qa_chain.invoke({"query": question})
        logger.info("Query answered. Source documents returned: %d", len(result.get("source_documents", [])))
        return {
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }
    
    def summarize(self) -> str:
        """Generate a summary of the entire document"""
        logger.info("Generating document summary...")
        summary_prompt = """Based on all the content in this document, provide a comprehensive summary. 
Include all key sections, important numbers, dates, and requirements mentioned."""
        
        result = self.query(summary_prompt)
        logger.info("Summary generated successfully")
        return result["answer"]
    
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
