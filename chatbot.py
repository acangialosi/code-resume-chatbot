"""
Resume Chatbot - A simple RAG chatbot for answering questions about your professional experience.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, DirectoryLoader

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
VECTORSTORE_DIR = Path(__file__).parent / "vectorstore"


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
    
    return documents


def create_vectorstore(documents):
    """Create a FAISS vector store from documents."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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
        temperature=0.7,
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
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
