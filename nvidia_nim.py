import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

# Set up the Streamlit page
st.title("Nvidia NIM Q&A Generator")

# API Key input section
api_key = st.text_input("Enter your NVIDIA API Key", type="password")

# Validate API key format
if not api_key:
    st.warning("Please enter your NVIDIA API key to continue")
    st.stop()
elif not api_key.startswith("nvapi"):
    st.error("Invalid API key format. The key should start with 'nvapi'")
    st.stop()

# Set the API key for the session
os.environ['NVIDIA_API_KEY'] = api_key

# Initialize the LLM
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Function to create vector embeddings
def create_vector_db():
    if "vector_store" not in st.session_state:
        # Create embeddings
        embeddings = NVIDIAEmbeddings()
        
        if uploaded_files:
            # Save uploaded files temporarily
            temp_dir = "./temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            for file in uploaded_files:
                with open(os.path.join(temp_dir, file.name), "wb") as f:
                    f.write(file.getbuffer())
            doc_loader = PyPDFDirectoryLoader(temp_dir)
        else:
            # Default to us_census directory if no files uploaded
            doc_loader = PyPDFDirectoryLoader("./temp")
            
        documents = doc_loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        doc_chunks = text_splitter.split_documents(documents[:30])
        
        # Create vector store
        st.session_state.vector_store = FAISS.from_documents(doc_chunks, embeddings)
        
        # Clean up temp files if they exist
        if uploaded_files and os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

# Create prompt template
qa_prompt = ChatPromptTemplate.from_template(
"""
Generate a number of brief question and answer pairs asked by user and based on the given context.
Please provide the most accurate response based on the context.
<context>
{context}
<context>
Questions:{input}
"""
)
# Load documents - check for uploaded files first
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.button("Prepare Document Database"):
    with st.spinner("Creating vector database..."):
        create_vector_db()
    st.success("Document database is ready for queries")


# User input section
user_question = st.text_input("Enter your the number of Q&A pairs you want to generate:")

# Process user question
if user_question and "vector_store" in st.session_state:
    with st.spinner("Searching for answer..."):
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # Create retriever
        retriever = st.session_state.vector_store.as_retriever()
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Time the response
        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': user_question})
        processing_time = time.process_time() - start_time
        
        # Display answer
        st.subheader("Answer")
        st.write(response['answer'])
        st.caption(f"Processing time: {processing_time:.2f} seconds")

        # Show relevant document chunks
        with st.expander("Document Sources"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Source {i+1}**")
                st.write(doc.page_content)
                st.divider()
elif user_question and "vector_store" not in st.session_state:
    st.warning("Please prepare the document database first by clicking the 'Prepare Document Database' button.")


if __name__ == "__main__":
    main()