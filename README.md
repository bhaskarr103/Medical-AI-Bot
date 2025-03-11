# Medical Chatbot using LangChain, FAISS & Hugging Face

## Overview
This project implements a **Medical Chatbot** that retrieves information from a knowledge base built using medical PDFs. It uses **LangChain**, **FAISS** for vector storage, and **Mistral-7B-Instruct-v0.3** from Hugging Face for question-answering. The chatbot is deployed with **Streamlit** for an interactive UI.

---

---

## **Project Directory Structure**
```
medical-chatbot/
│── data/                    # Folder containing medical PDFs
│── vectorstore/             # FAISS vector database
│── app.py                   # Streamlit UI
│── model.py                 # Model setup & retrieval
│── requirements.txt         # Dependencies
│── .env                     # Hugging Face API key
```


## Output
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/aaa8fdd6-2564-4b8b-bcf1-3751c2f30c2f" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/638b48c8-5b52-4fff-88df-04a213b5e834" width="300"></td>
  </tr>
</table>




## **Model Flow & Architecture**

### **1. Data Ingestion**
- **PDF Loading:** The chatbot processes **500+ pages** from medical textbooks.
- **Libraries Used:** `PyPDFLoader`, `DirectoryLoader`.
- **Processing Flow:**
  1. Load PDFs from `data/` directory.
  2. Extract textual content from PDF pages.

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()

documents = load_pdf_files("data/")
```

---

### **2. Text Chunking**
- **Why?** Large texts must be divided into chunks to improve embedding performance.
- **Approach:**
  - Use `RecursiveCharacterTextSplitter`.
  - Define `chunk_size=500` and `chunk_overlap=50`.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)

text_chunks = create_chunks(documents)
```

---

### **3. Vectorization (Embeddings & FAISS Storage)**
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Storage:** FAISS (Facebook AI Similarity Search) for fast retrieval.

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local("vectorstore/db_faiss")
```

---

### **4. Model (LLM) Setup**
- **LLM Used:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Temperature:** `0.5` (balanced randomness)
- **Token Management:** Uses `HF_TOKEN` for authentication.

```python
from langchain_huggingface import HuggingFaceEndpoint

def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(repo_id=huggingface_repo_id, temperature=0.5, max_length=512, token=os.environ.get("HF_TOKEN"))
```

---

### **5. Retrieval & Question-Answering**
- **Retrieval:** FAISS searches top `k=5` most relevant chunks.
- **Prompt Template:** Ensures response stays within retrieved context.
- **Chain Type:** `stuff` (merges retrieved chunks before answering).

```python
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

CUSTOM_PROMPT_TEMPLATE = """
Use the provided context to answer the user's question.
If you don't know the answer, just say that you don't know.
Context: {context}
Question: {question}
Start the answer directly.
"""

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm("mistralai/Mistral-7B-Instruct-v0.3"),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)
```

---

### **6. Streamlit UI Integration**
- **Interactive Chat Interface:** Users input queries and get AI-generated medical responses.
- **State Management:** Uses `st.session_state` to maintain chat history.
- **Error Handling:** Displays meaningful error messages when necessary.

```python
import streamlit as st

def main():
    st.title("Ask MedicalBot!")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Enter your medical question here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        response = qa_chain.invoke({'query': prompt})
        result = response["result"]
        st.chat_message('assistant').markdown(result)
        st.session_state.messages.append({'role': 'assistant', 'content': result})

if __name__ == "__main__":
    main()
```



---

## **Key Features & Enhancements**
✅ **Fast retrieval**: FAISS ensures quick search of medical knowledge.
✅ **Customizable LLM**: Easily swap the LLM for different Hugging Face models.
✅ **Secure API access**: Uses `.env` for authentication.
✅ **Interactive UI**: Built with Streamlit for a user-friendly experience.
✅ **Context-aware QA**: Responses are always grounded in the provided medical data.

---

## **Future Improvements**
🚀 **Expand dataset**: Add more medical books/articles for wider coverage.
🚀 **Improve embeddings**: Experiment with better transformer-based embeddings.
🚀 **Fine-tune LLM**: Optimize Mistral-7B for medical-specific queries.

---

## **Installation & Usage**
### **1. Install dependencies**
```bash
pip install -r requirements.txt
```
### **2. Set up API key**
Create a `.env` file:
```
HF_TOKEN=your_huggingface_api_key
```
### **3. Run the chatbot**
```bash
streamlit run app.py
```

---

## **Conclusion**
This **Medical Chatbot** provides AI-powered medical information retrieval. It efficiently processes medical texts, stores them in FAISS, and uses a state-of-the-art LLM for answering queries. Future work can focus on improving accuracy, expanding datasets, and enhancing UI/UX.

