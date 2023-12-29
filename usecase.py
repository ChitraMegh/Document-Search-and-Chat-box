# main.py
import streamlit as st
from document_search_chat import search_and_chat
import os
from llamaindex import LlamaIndex
from mistral import Mistral

st.title("Document Search and Chat Application")

user_query = st.text_input("Enter your query: ")

if user_query:
    # Call backend function to get document search results and chat response
    search_results, chat_response = search_and_chat(user_query)

    # Display search results
    st.subheader("Search Results:")
    for result in search_results:
        st.write(result)

    # Display chat response
    st.subheader("Chat Response:")
    st.write(chat_response)


# Load Weaviate for vector database
weaviate = LlamaIndex("weaviate://localhost")

# Load Mistral7b for language understanding and generation
mistral = Mistral("mistral://localhost")

# Function to perform document search and chat
def search_and_chat(user_query):
    # Implement logic to fetch document search results using Weaviate
    search_results = search_documents(user_query)

    # Implement logic to generate chat response using Mistral7b
    chat_response = generate_chat_response(user_query)

    return search_results, chat_response

def search_documents(query):
    # Placeholder logic for document search
    # In a real scenario, you would use Weaviate to retrieve relevant documents
    document_folder = "documents/"
    documents = os.listdir(document_folder)
    search_results = [doc for doc in documents if query.lower() in doc.lower()]
    return search_results

def generate_chat_response(query):
    # Placeholder logic for chat response
    # In a real scenario, you would use Mistral7b to generate a chat-like response
    chat_response = mistral.generate_text(query)
    return chat_response