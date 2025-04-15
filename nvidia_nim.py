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
try:
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
    st.success("Successfully connected to NVIDIA NIM API")
except Exception as e:
    st.error(f"Error connecting to NVIDIA NIM API: {str(e)}")
    st.info("Please check your API key and try again")
    st.stop()

# Function to create vector embeddings
def create_vector_db():
    if "vector_store" not in st.session_state:
        try:
            # Create embeddings
            embeddings = NVIDIAEmbeddings()
            
            # Check if files were uploaded and process them first
            if uploaded_files and len(uploaded_files) > 0:
                st.info(f"Processing {len(uploaded_files)} uploaded PDF files...")
                # Save uploaded files temporarily
                temp_dir = "./temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                for file in uploaded_files:
                    with open(os.path.join(temp_dir, file.name), "wb") as f:
                        f.write(file.getbuffer())
                doc_loader = PyPDFDirectoryLoader(temp_dir)
                
                documents = doc_loader.load()
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
                doc_chunks = text_splitter.split_documents(documents[:30])
                
                # Create vector store with progress indicator
                with st.spinner("Creating vector embeddings... This may take a moment"):
                    st.session_state.vector_store = FAISS.from_documents(doc_chunks, embeddings)
                
                # Clean up temp files
                if os.path.exists(temp_dir):
                    for file in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, file))
                    os.rmdir(temp_dir)
            else:
                # Default to temp directory if no files uploaded
                st.info("No files uploaded. Using saved files from temp directory...")
                temp_dir = "./temp"
                os.makedirs(temp_dir, exist_ok=True)
                # Check if directory has files
                if not os.listdir(temp_dir):
                    st.error("No files found in the default directory. Please upload PDF files.")
                    return
                    
                doc_loader = PyPDFDirectoryLoader(temp_dir)
                documents = doc_loader.load()
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
                doc_chunks = text_splitter.split_documents(documents[:30])
                
                # Create vector store with progress indicator
                with st.spinner("Creating vector embeddings... This may take a moment"):
                    st.session_state.vector_store = FAISS.from_documents(doc_chunks, embeddings)
                    st.success("Successfully processed files from temp directory")
        except Exception as e:
            st.error(f"Error creating vector database: {str(e)}")
            return

# Create prompt template
qa_prompt = ChatPromptTemplate.from_template(
"""
Generate {input} brief question and answer pairs based on the given context.
Each pair should be factual and directly derived from the provided documents.
Format your response as numbered Q&A pairs with clear separation between questions and answers.

<context>
{context}
</context>

Generate exactly {input} question and answer pairs. Be concise and accurate.
"""
)
# Load documents - check for uploaded files first
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.button("Prepare Document Database"):
    with st.spinner("Creating vector database..."):
        create_vector_db()
    st.success("Document database is ready for queries")


# User input section
user_question = st.text_input("Enter the number of Q&A pairs you want to generate (1-10):")

# Validate user input is a number
if user_question:
    try:
        num_pairs = int(user_question)
        if num_pairs < 1 or num_pairs > 10:
            st.warning("Please enter a number between 1 and 10")
            user_question = None
    except ValueError:
        st.warning("Please enter a valid number")
        user_question = None

# Process user question
if user_question and "vector_store" in st.session_state:
    try:
        with st.spinner("Generating Q&A pairs..."):
            # Create document chain
            document_chain = create_stuff_documents_chain(llm, qa_prompt)
            
            # Create retriever
            retriever = st.session_state.vector_store.as_retriever()
            
            # Create retrieval chain
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Time the response with a timeout mechanism
            start_time = time.process_time()
            
            # Set a timeout for the API call
            try:
                response = retrieval_chain.invoke({'input': user_question})
                processing_time = time.process_time() - start_time
                
                # Display answer
                st.subheader("Generated Q&A Pairs")
                st.write(response['answer'])
                st.caption(f"Processing time: {processing_time:.2f} seconds")

                # Show relevant document chunks
                with st.expander("Document Sources"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Source {i+1}**")
                        st.write(doc.page_content)
                        st.divider()
            except Exception as e:
                st.error(f"Error generating Q&A pairs: {str(e)}")
                st.info("The API request might have timed out. Try with a smaller number of Q&A pairs.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.info("Please try again or check your API key.")
elif user_question and "vector_store" not in st.session_state:
    st.warning("Please prepare the document database first by clicking the 'Prepare Document Database' button.")

# Remove the main() function call as it doesn't exist
# if __name__ == "__main__":
#     main()