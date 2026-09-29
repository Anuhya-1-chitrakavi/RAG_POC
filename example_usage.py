#!/usr/bin/env python3
"""
Example usage of the RAG pipeline
Shows different ways to use it
"""

import os
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()

def main():
    # Initialize the RAG pipeline
    print("Initializing RAG Pipeline with your PDF...\n")
    rag = RAGPipeline("Chitrakavi Lakshmi Anuhya - Offer Letter.pdf")
    
    # Example 1: Ask specific questions
    print("\n" + "="*60)
    print("EXAMPLE 1: Specific Questions")
    print("="*60)
    
    questions = [
        "What is the salary structure mentioned in the offer letter?",
        "When should the employee join?",
        "What are the main benefits provided?",
        "What is the probation period?",
        "What is the CTC mentioned in the offer?",
    ]
    
    for question in questions:
        print(f"\n❓ {question}")
        result = rag.query(question)
        print(f"✅ {result['answer']}\n")
    
    # Example 2: Get a summary
    print("\n" + "="*60)
    print("EXAMPLE 2: Document Summary")
    print("="*60)
    
    summary = rag.summarize()
    print(f"\n{summary}\n")
    
    # Example 3: Complex multi-part query
    print("\n" + "="*60)
    print("EXAMPLE 3: Complex Query")
    print("="*60)
    
    complex_query = "Compare the probation period with the service bond period and explain what they mean for the employee"
    print(f"\n❓ {complex_query}")
    result = rag.query(complex_query)
    print(f"✅ {result['answer']}\n")


if __name__ == "__main__":
    main()
