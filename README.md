# RAG-Based Chat System

## Problem-Statement:

- Build a basic UI (using Gradio, Streamlit, or a simple cURL command) that allows a user to upload a PDF. The user can then ask questions that may or may not have answers in the uploaded PDF(s). For questions that have answers in the PDF(s), the system should return valid responses. For questions without sufficient information in the PDF, it should respond with “I don’t know the answer as not sufficient information is provided in the PDF” (no hallucination).
Backend: Must use Private SLM (pick any Small Language Model that can be hosted locally); use an open-source vector database (in-memory or any choice)

## Environment:
- Python v3.10 or higher
- VS Code environment preferred
- If using Jupyter notebook or Google Colab, mention the below magic command at the start:
-            %%writefile app.py

## Dependencies: 
- Create a virtual environment by using the below command in Powershell terminal:
-            python -m venv venv_name
-        venv_name\Scripts\Activate.ps1
- run the requirements.txt file:
-         !pip install -r requirements.txt
- If GPU is available, make sure to make changes to the below code block:
-         model_kwargs={"device": "cpu"} --> model_kwargs={"device": "auto"}
- To start the Streamlit Server for app, run the below command in terminal:
        streamlit run streamlit_app.py
- In case of any runtime error, re-run the code. 
- In case of any other Embedding / Tokenizer / LLM model from Hugging Face, make sure you have the access to the model on HuggingFace Hub (if needed). If access needed, make sure to pass the **Huggingface_api_key** in Pipeline.
- Streamlit UI run on a **localhost:8501** by default, make sure the port is available. 
