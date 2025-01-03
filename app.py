import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Page layout
st.set_page_config(page_title="FinanceBot: News Research Tool", page_icon="📊", layout="wide")

# Title and Sidebar
st.title("FinanceBot: News Research Tool 📊")
st.sidebar.title("🔼 Enter News Article URLs")

# URL Input Section
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", placeholder=f"Enter URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("🌄 Process URLs")
file_path = "faiss_store_openai.pkl"

# Main Placeholder Section
main_placeholder = st.empty()

# Set LLM configuration
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    with st.spinner("Processing URLs. Please wait..."):
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        main_placeholder.success("Data Loaded Successfully! 🚀")
        
        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)
        main_placeholder.success("Text Splitter Initialized! 📦")
        
        # Create embeddings and save to FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.success("Embedding Vectors Built! 📊")
        time.sleep(2)

        # Save FAISS index
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)
        main_placeholder.success("FAISS Index Saved Successfully! 📂")

# Query Input Section
query = st.text_input("Ask a question:", placeholder="Type your query here...")
if query:
    if os.path.exists(file_path):
        with st.spinner("Fetching the answer..."):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

            # Display answer
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources")
                st.markdown("\n".join(f"- {source}" for source in sources.split("\n")), unsafe_allow_html=True)
