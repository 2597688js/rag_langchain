"""
Author : Janarddan Sarkar
file_name : rag_app.py
date : 25-02-2025
description : RAG for a given URL using free Groq models and hugging face embeddings
"""
from pkgutil import find_loader

import streamlit as st
import os
import bs4
from dotenv import load_dotenv, find_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from typing import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Load API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ API Key is missing. Set GROQ_API_KEY as an environment variable.")
    st.stop()

llm = init_chat_model("llama3-8b-8192", model_provider="groq")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)

# Define Prompt Template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Function to retrieve relevant context
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


# Function to generate the answer
def generate(state: State):
    if not state["context"]:
        return {"answer": "I couldn't find relevant information. Please try a different question."}

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = custom_rag_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}


# Streamlit UI
st.title("RAG Web App for URL")
url = st.text_input("Enter URL:")
question = st.text_input("Enter your question:")

if st.button("Submit"):
    try:
        # Load and parse web content
        loader = WebBaseLoader(web_paths=(url,))
        docs = loader.load()

        if not docs:
            st.error("No content found at the given URL")
        else:
            # Clean and extract text using BeautifulSoup
            parsed_texts = []
            for doc in docs:
                soup = bs4.BeautifulSoup(doc.page_content, "html.parser")

                # Remove script, style, and unwanted elements
                for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
                    tag.decompose()

                parsed_texts.append(soup.get_text(separator=" ", strip=True))

            # Combine extracted text
            combined_text = "\n".join(parsed_texts)

            # Convert into Document format
            document = Document(page_content=combined_text)

            # Split content
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_splits = text_splitter.split_documents([document])

            # Index chunks
            vector_store.add_documents(documents=all_splits)

            # Compile application and test
            graph_builder = StateGraph(State).add_sequence([retrieve, generate])
            graph_builder.add_edge(START, "retrieve")
            graph = graph_builder.compile()

            # Invoke graph
            response = graph.invoke({"question": question})

            st.success("Answer:")
            # st.write(response['content'])
            st.write(response['answer'].content)

    except Exception as e:
        st.error(f"Error: {str(e)}")
