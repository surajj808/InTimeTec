# BAsic easy code for Mulitple File Pdf reader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import torch
import chromadb
import os
import re
import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyMuPDFLoader


PROMPT_TEMPLATE = """
You are a Question Answer expert, tasked with answering questions based on the information contained in a provided PDF document.
If you don't find the answer or related content in the documents provided, simply say that "I don’t know the answer as not 
sufficient information is provided in the PDFs".
Make your answers precise & relevant to the query, try to provide limited information as asked.
Clarify Uncertainty: If the answer to a question is not found in the document, straight away state explicitly that "I don’t know the answer as not 
sufficient information is provided in the PDFs".
NOTE: Do not hallucinate or give out of the context answers. 
NOTE: Do not frame the answers out of the documents provided.
NOTE: Max length of your response should not exceed the range of 200-250 words.
Question: "What is the highest sales recorded for the Shampoo ?"

  [/INST]
  Answer: "The highest sales recorded for the Shampoo is 195,000."
  </s>[INST]
  Context: "{context}"
  {question} [/INST]
  Answer is:
  """
persist_directory = "chromadb"
collection_name="dummy"

def check_and_create_collection(client, collection_name):
    # Check if the collection exists
    existing_collections = [col.name for col in client.list_collections()]
    
    if collection_name in existing_collections:
        print(f"Collection '{collection_name}' exists. Deleting the existing collection.")
        client.delete_collection(collection_name)
        
    print(f"Creating collection '{collection_name}' with new embeddings.")


# Ensure the directory exists or create it if necessary
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


## client
client = chromadb.PersistentClient(path=persist_directory)


acc_key = "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"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", token=acc_key)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForSeq2SeqLM.from_pretrained(
                                            "google/flan-t5-small",
                                            torch_dtype=torch.float16,
                                            device_map=None, # Disable automatic device placement
                                            # load_in_4bit=True,
                                            token=acc_key,
)

pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                # truncation=True
            )
### Embedding loading
modelPath = "sentence-transformers/all-mpnet-base-v2"  # Cohere/Cohere-embed-english-v3.0
# model_kwargs={"device": "cpu"},
encode_kwargs = {"normalize_embeddings": False}

embedding_function = HuggingFaceEmbeddings(
                        model_name=modelPath, 
                        model_kwargs={"device": "cpu"}, 
                        encode_kwargs=encode_kwargs
                        )

hf_llm = HuggingFacePipeline(
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


def pdf_embedding(documents):
    loader = PyMuPDFLoader(documents)    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    split_docs2 = texts
    check_and_create_collection(client, collection_name)
    vectordb = Chroma.from_documents(
              documents=split_docs2,
              embedding=embedding_function,
              persist_directory=persist_directory,
              collection_name=collection_name)
    vectordb.persist()  # Ensure the changes are saved
    return vectordb


def FinalRetriever(query): ########
    FinalPrompt = PromptTemplate(template = PROMPT_TEMPLATE, input_variables = ["question"])

    chroma_client = chromadb.PersistentClient(persist_directory)
    collection_objects = chroma_client.list_collections()
    collection_names = [collection.name for collection in collection_objects]
   
  
    for specific_store in collection_names:
        if collection_name == specific_store:
            connect_vectorstore = Chroma(collection_name=specific_store, 
                                         embedding_function=embedding_function, 
                                         persist_directory=persist_directory)
        else:
            pass

    vector_store_retriever_obj = connect_vectorstore.as_retriever(search_kwargs={'k':3})

    # vector_store_retriever_obj = vectordb.as_retriever(search_kwargs={'k':3})
    qa_chain = RetrievalQA.from_chain_type(llm=hf_llm,
                                    chain_type="stuff",
                                    retriever=vector_store_retriever_obj,
                                    chain_type_kwargs={"prompt": FinalPrompt},
                            return_source_documents=True)
    qa_chain_response = qa_chain.invoke(query)
    answer=qa_chain_response["result"].split("Answer is")[-1:][0]
    answer = re.sub(r"[\[\]:]", "", answer)
    answer = re.sub(r'\n', ' ', answer).strip()
    return answer
documents=r'Cpdf_directory\doc.pdf'
# pdf_embedding(documents)
query="what is purpose of this analysis?"
query2='how many indicators were identified to quantify the change in pillar? '
query1="name the 5 dimensions of the global cooperation barometer?"
query4 = "what was launched in 2001?"
answeer=FinalRetriever(query4)
print("Answer is :",answeer)