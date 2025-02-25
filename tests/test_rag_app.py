"""
Author : Janarddan Sarkar
file_name : test_rag_app.py 
date : 25-02-2025
description : 
"""
import os
import pytest
from dotenv import load_dotenv, find_dotenv
from unittest.mock import patch
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

# Sample test data
MOCK_WEB_CONTENT = "This is a test article for the RAG app."

# Load environment variables from .env file
load_dotenv(find_dotenv())

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = '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'

# Load API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("GROQ API Key is missing. Set GROQ_API_KEY as an environment variable.")
    exit()


@pytest.fixture
def mock_documents():
    """Mocked document returned from WebBaseLoader"""
    return [Document(page_content=MOCK_WEB_CONTENT)]


@pytest.fixture
def text_splitter():
    """Returns a text splitter instance"""
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


@pytest.fixture
def vector_store():
    """Returns a mock vector store"""
    return InMemoryVectorStore(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))


def test_web_loader(mock_documents):
    """Test that the web loader correctly fetches content"""
    with patch.object(WebBaseLoader, "load", return_value=mock_documents):
        loader = WebBaseLoader(web_paths=("https://example.com",))
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].page_content == MOCK_WEB_CONTENT


def test_text_splitting(text_splitter, mock_documents):
    """Test that text is split into chunks correctly"""
    splits = text_splitter.split_documents(mock_documents)
    assert len(splits) > 0
    assert MOCK_WEB_CONTENT in splits[0].page_content


def test_vector_store(vector_store, mock_documents):
    """Test that documents are added to the vector store"""
    doc_ids = vector_store.add_documents(documents=mock_documents)
    assert len(doc_ids) > 0


def test_retrieval(vector_store, mock_documents):
    """Test that vector store retrieves relevant documents"""
    vector_store.add_documents(documents=mock_documents)
    retrieved_docs = vector_store.similarity_search("test")
    assert len(retrieved_docs) > 0
    assert MOCK_WEB_CONTENT in retrieved_docs[0].page_content


def test_llm_response():
    """Test the LLM response format"""
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    custom_rag_prompt = PromptTemplate.from_template(
        """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.

        {context}

        Question: {question}

        Helpful Answer:"""
    )

    question = "What is AI?"
    docs_content = "AI is artificial intelligence."

    messages = custom_rag_prompt.invoke({"question": question, "context": docs_content})
    response = llm.invoke(messages)
    response_str = str(response)
    assert response_str.strip() != "", "Response should not be empty"



