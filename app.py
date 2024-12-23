import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Import LangChain's Document class
import os
# from version_check import ensure_sqlite_version
# ensure_sqlite_version()
# import sys
# sys.path.insert(0, "/home/appuser/sqlite/lib")
from chromadb import PersistentClient
from langchain_google_genai import ChatGoogleGenerativeAI
import uuid
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import concurrent.futures
from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
import time
from classify import Query
from langchain_groq import ChatGroq
from faq import LegalFAQProcessor

load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     api_key=os.getenv('GEMINI_API_KEY')
# )


llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key=os.getenv('GROQ_API_KEY')
)

st.set_page_config(
    page_title="Legal Helper",
    page_icon="👨🏻‍⚖️⚖",
    layout="wide" 
)
# Initialize SentenceTransformer for embeddings
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Initialize Chroma Client and Collection
persistent_client = PersistentClient(path="./legal_db")
collection = persistent_client.get_or_create_collection("legal_docs_collection")

faq=LegalFAQProcessor(chroma_client=persistent_client)

vector_store = Chroma(
    client=persistent_client,
    collection_name="legal_docs_collection",
    embedding_function=embedding_model,
)

# Text splitter for better indexing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

# Function to process and upload documents
def upload_documents(uploaded_files):
    clustered_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        pages = []
        for i, doc in enumerate(docs):
            page_content = doc.page_content
            doc.metadata["page"] = i + 1
            metadata = {
                "source": uploaded_file.name,
                "page": i
            }
            page_info = {
                "page_content": page_content,
                "metadata": metadata
            }
            pages.append(page_info)
            print(i + 1)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = [
            Document(page_content=page["page_content"], metadata=page["metadata"])
            for page in pages
        ]
        for doc in documents:
            if 'metadata' in doc.__dict__ and 'source' in doc.metadata:
                doc.metadata['source'] = uploaded_file.name

        docs = text_splitter.split_documents(documents)
        uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
        clustered_docs.append(docs)
        os.remove(temp_file_path)
    # Use ThreadPoolExecutor to upload documents in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for doc in clustered_docs:
            # Generate a unique ID for each document
            uuids = [str(uuid.uuid4()) for _ in range(len(docs))]
            futures.append(
                executor.submit(vector_store.add_documents, documents=doc, ids=uuids)
            )
        
        print(f'Submitted {len(futures)} documents for upload to Chroma.')
        concurrent.futures.wait(futures)

        # Log any errors that occurred during the uploads
        for future in futures:
            if future.exception() is not None:
                print(f'Error during upload: {future.exception()}')
    return "Documents processed and added to the database!"

# Function to manage and delete documents
def manage_documents():
    # Retrieve all documents' metadata from the Chroma collection
    documents = collection.get(include=["metadatas"])["metadatas"]
    document_names = set(doc["source"] for doc in documents)
    return document_names

def delete_document(doc_name, batch_size=40000):
    # Delete documents from the collection based on metadata source name
    docs_to_delete = collection.get(where={"source": doc_name})
    
    if not docs_to_delete["ids"]:
        return f"No documents found with source name: {doc_name}."
    
    document_ids = docs_to_delete["ids"]
    total_docs = len(document_ids)
    
    # Delete documents in batches
    for i in range(0, total_docs, batch_size):
        batch_ids = document_ids[i:i + batch_size]
        collection.delete(ids=batch_ids)
        print(f"Deleted batch {i // batch_size + 1} of {len(batch_ids)} documents.")

    return f"{doc_name} has been deleted."

# Function to handle chat queries with the LLM
def chat_with_llm(query):
    # Retrieve relevant chunks from Chroma
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    prompt = PromptTemplate(
    template="""
    You are a highly experienced legal assistant specializing in Indian law. When responding, maintain a concise, professional tone. Speak like a seasoned lawyer by focusing only on the legal aspects of the query and avoiding unnecessary explanations. Provide actionable advice based on legal provisions, avoiding phrases like "provided content".
    
    Context: {context}

    Use the given legal document context, give an actionable response, referencing specific clauses, sections, or phrases as necessary. 
    If the context is insufficient, provide general legal advice.
    Ensure your response is clear, professional, and easy to understand. 

    Here’s how you should respond:
    1. Address the client politely and establish credibility.
    2. Offer a legal opinion with references to relevant sections of the Indian Penal Code (IPC) or other applicable laws (stating under the law or code or section,.etc).
    3. Avoid lengthy background explanations. Stick to what the law says and how it applies to the client’s situation.
    4. If necessary, advise the client to document incidents and consult law enforcement or a legal expert for further action.
    5. Do not mention client and do not follow a letter format.

    Always prioritize actionable, legal advice over general information.

    Question: {query}
    Answer:
    """,
    input_variables=["context", "query"]
    )

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    qa_chain = prompt | llm
    # Use RetrievalQA to handle query with context
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, 
    #                                        retriever=retriever, 
    #                                        return_source_documents=True,
    #                                        chain_type_kwargs={"prompt": prompt})
    response = qa_chain.invoke({ "context":context,"query" : query })

    return {
        "result": response.content,
        "source_documents": docs
    }

def stream_response(response_text, chunk_size=15, delay=0.1):
    start = 0
    while start < len(response_text):
        # Get the next chunk of text
        chunk = response_text[start:start + chunk_size]
        yield chunk
        start += chunk_size
        time.sleep(delay)

# hide_streamlit_style = """
# <style>
# #MainMenu {visibility: hidden;}
# header {visibility: hidden;}
# footer {visibility: hidden;}
# </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Injecting custom CSS to reduce padding
reduce_padding_style = """
<style>
.st-emotion-cache-1ibsh2c {
    padding: 1.2rem 1rem 1.2rem !important; /* Adjust these values as needed */
    max-width: initial !important;
    min-width: auto !important;
}

@media (min-width: calc(54rem)) {
    .st-emotion-cache-1ibsh2c {
        padding-left: 1.6rem !important; /* Adjust these values as needed */
        padding-right: 1.2rem !important;
    }
}

.st-emotion-cache-h4xjwg {
    height: 1rem !important; /* Adjust the height as needed */
}
</style>
"""
st.markdown(reduce_padding_style, unsafe_allow_html=True)

# Two-column layout
doc_col, chat_col = st.columns([2.2,4.8],gap='medium')

@st.fragment
def st_upload():
    # Upload Section
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = str(uuid.uuid4())  # Generate a unique key for the uploader

    st.session_state.uploaded_files = st.file_uploader(
        "Upload Legal Documents (PDFs)", 
        accept_multiple_files=True, 
        type=["pdf"],
        key=st.session_state.uploader_key
    )
    
    if st.session_state.uploaded_files:
        message = upload_documents(st.session_state.uploaded_files)
        st.success(message)
        # Detach files by resetting session state
        st.session_state.uploader_key = str(uuid.uuid4())

        st.rerun()
    # View/Delete Section
    st.markdown("##### Uploaded Documents")
    document_names = manage_documents()
    if document_names:
        for i,doc_name in enumerate(document_names):
            col1, col2 = st.columns([3, 1])
            col1.write(f"{i+1}. {doc_name}")
            # Add a delete button
            if col2.button("",icon=":material/delete:", key=doc_name):
                message = delete_document(doc_name)
                st.success(message)
                st.rerun()
    else:
        st.text("No documents uploaded yet.")
# Documents Section (Column 1)
with doc_col:
    st.markdown("#### Documents")
    st_upload()

    

@st.dialog("Source Documents")
def source_dialog(source_docs):
    for doc in source_docs:
        st.write(f"Source: {doc.metadata['source']}")
        st.write(doc.page_content)


# Chat Section (Column 2)
with chat_col:
    st.markdown("### 👨🏻‍⚖️⚖ Legal Helper")
    messages = st.container(height=450,border=False)

    document_names = manage_documents()
    query = st.chat_input("Ask a legal query", disabled=not document_names)

    # Fetch top FAQs
    questions = faq.fetch_top_faqs()
    print("Questions", questions)

    # Display pills only if questions are available
    if len(questions) >= 1:
        selected_pill = st.pills("FAQs", questions, default=None, selection_mode="single")
        st.session_state.selected_question = selected_pill if selected_pill else None
    else:
        st.session_state.selected_question = None  # No FAQs available

    # Dynamically choose the query source (pill or chat input)
    if st.session_state.selected_question and not query:
        # If a pill is selected and no chat input is entered
        query = st.session_state.selected_question
    elif query:
        # If chat input is provided, reset pill selection
        st.session_state.selected_question = None

    # Check if no documents are uploaded
    if not document_names:
        st.warning("No documents uploaded. Please upload legal documents to enable chat functionality.")
        query = None  # Disable query processing

    # Process the query
    if query:
        messages.chat_message("user", avatar=":material/person:").write(query)
        loader_text = messages.empty() 
        qh = Query()
        with loader_text.container():
            with st.spinner("Understanding your query..."):
                # Classify the query
                classification = qh.classify_question(query)

                if classification == "conversational_query":
                    with st.spinner("Loading..."):
                        response = qh.call_conversational_tool(query)
                elif classification == "legal_query":
                    with st.spinner("Searching Legal Precedents for Answers..."):
                        response = chat_with_llm(query)
                        status, message = faq.process_user_question(query)
                        print(status, message)
                else:
                    raise Exception("Unexpected classification result.")

        # Display user query in chat messages
        messages.chat_message("assistant").write_stream(stream_response(response['result']))

        # Display source documents
        if 'source_documents' in response:
            with messages.popover("View Sources"):
                for doc in response["source_documents"]:
                    st.markdown(f"**Source 📄 : {doc.metadata['source']}**")
                    st.markdown(f"**Page Number : {doc.metadata['page']}**")
                    st.write(doc.page_content)
