"""
Resume Chatbot - A simple RAG chatbot for answering questions about your professional experience.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, DirectoryLoader

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
VECTORSTORE_DIR = Path(__file__).parent / "vectorstore"

# System prompt for the chatbot
SYSTEM_TEMPLATE = """You are a helpful assistant that answers questions about a person's professional experience based on their performance reviews and career documents.

Use the following context to answer the question. The context includes performance reviews (called "Connects") from different fiscal years and halves (e.g., FY25H1 = Fiscal Year 2025, first half).

When answering:
- Be specific and cite time periods when relevant (e.g., "In FY25H1...")
- If asked about accomplishments, projects, or impact, draw from the "Summarize your impact" and "Core priorities" sections
- If asked about skills or growth areas, look at "Opportunities for growth" and manager feedback
- If the information isn't in the context, say so rather than making things up
- Synthesize information across multiple time periods when relevant

Context:
{context}

Question: {question}

Answer:"""


def parse_fiscal_period(filename: str) -> str:
    """Extract fiscal year and half from filename for metadata."""
    match = re.search(r'fy(\d+)h(\d)', filename.lower())
    if match:
        year = match.group(1)
        half = match.group(2)
        return f"FY{year}H{half}"
    return "Unknown Period"


def load_documents():
    """Load all text and markdown files from the data directory."""
    documents = []
    
    # Load .txt files
    txt_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents.extend(txt_loader.load())
    
    # Load .md files
    md_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents.extend(md_loader.load())
    
    # Add fiscal period metadata to each document
    for doc in documents:
        filename = Path(doc.metadata.get("source", "")).name
        period = parse_fiscal_period(filename)
        doc.metadata["period"] = period
        # Prepend period to content so it's included in chunks
        if period != "Unknown Period":
            doc.page_content = f"[{period}]\n\n{doc.page_content}"
    
    return documents


def create_vectorstore(documents):
    """Create a FAISS vector store from documents."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Larger chunks to preserve more context
        chunk_overlap=300,  # More overlap to avoid cutting mid-section
        separators=["\n## ", "\n---\n", "\n\n", "\n", " "],  # Split on markdown headers first
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Save for reuse
    vectorstore.save_local(str(VECTORSTORE_DIR))
    return vectorstore


def load_vectorstore():
    """Load existing vector store or create new one."""
    embeddings = OpenAIEmbeddings()
    
    if VECTORSTORE_DIR.exists():
        return FAISS.load_local(
            str(VECTORSTORE_DIR), 
            embeddings,
            allow_dangerous_deserialization=True
        )
    
    documents = load_documents()
    if not documents:
        raise ValueError("No documents found in data/ directory. Add your resume files first!")
    
    return create_vectorstore(documents)


def create_chatbot():
    """Create the conversational chatbot chain."""
    vectorstore = load_vectorstore()
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,  # Lower temperature for more factual answers
    )
    
    # Custom prompt with system instructions
    prompt = PromptTemplate(
        template=SYSTEM_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),  # Retrieve more chunks
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    
    return chain


def chat(chain, question: str, chat_history: list) -> str:
    """Send a question to the chatbot and get a response."""
    result = chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })
    return result["answer"]


# CLI interface for testing
if __name__ == "__main__":
    print("Loading chatbot...")
    chain = create_chatbot()
    chat_history = []
    
    print("\n🤖 Resume Chatbot Ready!")
    print("Ask questions about professional experience. Type 'quit' to exit.\n")
    
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
            
        answer = chat(chain, question, chat_history)
        print(f"\nBot: {answer}\n")
        chat_history.append((question, answer))
