import os
import shutil
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings("ignore")

PERSIST_DIRECTORY = "chromadb_streamlit"
COLLECTION_NAME = "user_uploaded_pdfs"

os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

PROMPT_TEMPLATE = """
You are a Question Answer expert, tasked with answering questions based on the information contained in the provided documents.
If you don't find the answer or related content in the documents provided, simply say "I don’t know the answer as not 
sufficient information is provided in the PDFs".
Make your answers precise & relevant to the query, try to provide limited information as asked.
Clarify Uncertainty: If the answer to a question is not found in the document, straight away state explicitly that "I don’t know the answer as not 
sufficient information is provided in the PDFs".
NOTE: Do not hallucinate or give out-of-context answers. 
NOTE: Do not frame the answers out of the documents provided.
NOTE: Max length of your response should not exceed the range of 200-250 words.

Context: {context}

Question: {question}
Answer:
"""

MODEL_PATH = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_FUNCTION = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

HF_LLM = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 3,
        "max_length": 900,
        "frequency_penalty": 0.7,
        "presence_penalty": 0.1,
    },
)

def process_pdfs(uploaded_files):
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)

    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    documents = []
    for file_path in file_paths:
        loader = PyMuPDFLoader(file_path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=EMBEDDING_FUNCTION,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )
    vectordb.persist()

    shutil.rmtree(temp_dir)
    return vectordb

def get_answer(query):
    retriever = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_function=EMBEDDING_FUNCTION,
    ).as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=HF_LLM,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

    response = qa_chain({"query": query})
    return response["result"]

if "history" not in st.session_state:
    st.session_state.history = []

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

st.set_page_config(layout='wide', page_title='RAG Based Chat System')
st.sidebar.title("📁 Document Upload Section")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Please wait while I create a Knowledge Database for you ..."):
        st.session_state.vectordb = process_pdfs(uploaded_files)
    st.sidebar.success("Knowledge Database created successfully!")

st.title("💻 RAG Based Chat System")
query = st.text_input("Drop a query here:")

if query:
    if st.session_state.vectordb is None:
        st.warning("Please upload and process PDF files before submitting a query.")
    else:
        with st.spinner("Analysing ..."):
            answer = get_answer(query)
        st.session_state.history.append({"query": query, "answer": answer})
        st.write("Answer:")
        st.write(answer)

if st.session_state.history:
    st.markdown('---')
    for i, interaction in enumerate(st.session_state.history):
        st.write(f"**User:** {interaction['query']}")
        st.write(f"**Bot:** {interaction['answer']}")
        st.markdown('---')

st.write('Developed by - Suryapratap Sunahra')
