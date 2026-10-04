# Information Retrieval System

## Overview

This project is an AI-powered Information Retrieval System designed to answer questions about uploaded PDF documents.

The application allows a user to upload one or multiple PDF files, process their content, and ask natural-language questions about the information contained in those files.

The system uses text extraction, text chunking, embeddings, FAISS vector search, and a Gemini language model to retrieve relevant information and generate clear answers.

The interface is built with Streamlit.

---

## Project Goal

The main goal of this project is to build a simple Retrieval-Augmented Generation workflow for PDF documents.

Instead of sending an entire PDF directly to a language model, the system:

1. Extracts the text from the PDF.
2. Divides the text into smaller chunks.
3. Converts the chunks into numerical vector representations called embeddings.
4. Stores and indexes these vectors using FAISS.
5. Converts each user question into an embedding.
6. Finds the most relevant document chunks.
7. Sends the retrieved information to Gemini.
8. Generates an answer based on the document content.

This makes the system more efficient and allows users to search documents based on meaning rather than exact keywords.

---

## Main Technologies

- Python
- Streamlit
- LangChain
- Google Gemini API
- Google Generative AI Embeddings
- FAISS
- PyPDF2
- python-dotenv

---

## System Workflow

The application follows this general pipeline:

PDF Upload  
↓  
Text Extraction  
↓  
Text Chunking  
↓  
Embedding Generation  
↓  
FAISS Vector Store  
↓  
User Question  
↓  
Question Embedding  
↓  
Similarity Search  
↓  
Relevant PDF Chunks  
↓  
Gemini Language Model  
↓  
Final Answer

---

## Architecture

### 1. PDF Processing

Uploaded PDF files are processed using PyPDF2.

The `get_pdf_text()` function reads each PDF page and extracts its text content.

---

### 2. Text Chunking

Large documents cannot be efficiently processed as one single block of text.

The `get_text_chunks()` function uses `RecursiveCharacterTextSplitter` to divide the extracted text into smaller overlapping chunks.

The overlap helps preserve context between neighboring chunks.

---

### 3. Embeddings

Each text chunk is converted into an embedding using Google Generative AI embeddings.

An embedding is a numerical representation of the meaning of a piece of text.

Texts with similar meanings produce vectors that are located close to each other in vector space.

---

### 4. FAISS Vector Store

FAISS stands for Facebook AI Similarity Search.

FAISS stores and indexes the document embeddings so that relevant chunks can be retrieved efficiently.

When a user submits a question, the question is also converted into an embedding.

FAISS compares the question vector with the stored document vectors and returns the most similar chunks.

---

### 5. Question Answering

The retrieved chunks are combined into a context.

This context, together with the user's question and recent conversation history, is sent to the Gemini language model.

Gemini then generates a concise answer based only on the information retrieved from the PDF.

---

## Main Functions

### `get_pdf_text(pdf_docs)`

Extracts text from uploaded PDF documents.

Input:
- Uploaded PDF files

Output:
- Combined extracted text

---

### `get_text_chunks(text)`

Splits the extracted text into smaller chunks.

Input:
- Full PDF text

Output:
- List of text chunks

---

### `get_vector_store(text_chunks)`

Converts the text chunks into embeddings and creates a FAISS vector store.

Input:
- Text chunks

Output:
- FAISS vector store

---

### `retrieve_documents(vector_store, question)`

Searches the FAISS vector store for the document chunks that are most relevant to the user's question.

Input:
- Vector store
- User question

Output:
- Relevant document chunks

---

### `answer_question(vector_store, question, chat_history)`

Combines retrieval and generation.

It retrieves the relevant chunks, creates a prompt containing the PDF context and recent conversation history, sends the prompt to Gemini, and returns the final answer.

---

## Conversation History

The application uses Streamlit `session_state` to maintain the visible conversation.

Each user question and assistant answer is stored inside:

`st.session_state.messages`

This allows the interface to display:

User Question 1  
Assistant Answer 1  

User Question 2  
Assistant Answer 2  

User Question 3  
Assistant Answer 3

without losing previous messages when Streamlit reruns the application.

Recent messages are also included in the prompt so the assistant can understand conversational context.

---

## Challenges Encountered

During development, several compatibility and API issues were encountered.

### Conda and Git Bash

Conda environments were initially not activating correctly inside Git Bash.

The Conda shell integration had to be loaded using:

`source /c/Users/User/miniconda3/etc/profile.d/conda.sh`

A Python 3.11 Conda environment was later created to support newer LangChain packages.

---

### LangChain Version Changes

The original tutorial used LangChain APIs from 2024.

Several modules were moved into separate packages in newer LangChain versions.

Examples include:

- `langchain.text_splitter` → `langchain_text_splitters`
- vector stores → `langchain_community`
- Google integrations → `langchain_google_genai`
- older conversational components → `langchain_classic`

The code was updated to work with the newer package structure.

---

### Google PaLM Deprecation

The original project used:

- `GooglePalm`
- `GooglePalmEmbeddings`

These older integrations were replaced with:

- `ChatGoogleGenerativeAI`
- `GoogleGenerativeAIEmbeddings`

The project now uses Gemini models instead of PaLM.

---

### API Rate Limits

Large PDFs can create many text chunks and therefore many embedding requests.

Google may return errors such as:

- `429 RESOURCE_EXHAUSTED`
- `500 INTERNAL`

To reduce these issues, the project uses:

- larger text chunks
- controlled embedding batches
- fewer retrieved chunks
- retry logic for temporary API failures

---

### Second-Question Retrieval Issue

The older `ConversationalRetrievalChain` sometimes caused errors when processing follow-up questions with the newer Google embedding integration.

The final implementation avoids this issue by:

- using the raw user question directly for retrieval
- performing FAISS retrieval manually
- passing conversation history only to the language model

This keeps conversation context while making retrieval more reliable.

---

### Structured Gemini Responses

Gemini may return structured content blocks instead of a simple string.

A helper function extracts only the text portion of the response before displaying it in Streamlit.

This prevents internal metadata from appearing in the user interface.

---

## Final Application Behavior

The final application allows users to:

- Upload one or more PDF documents
- Process document content
- Convert document text into embeddings
- Store and search embeddings with FAISS
- Ask multiple questions
- Maintain visible conversation history
- Retrieve information based on semantic similarity
- Generate natural-language answers using Gemini
- Handle temporary API errors more reliably

---

## Example Questions

What skills does the applicant have?

Where did the applicant work?

What is the applicant’s education?

What tools does the applicant know?

What projects did the applicant complete?

Summarize the CV.

---

## Project Structure

```text
Information-Retrieval-System/
│
├── app.py
├── .env
├── requirements.txt
├── setup.py
│
├── src/
│   ├── __init__.py
│   └── helper.py
│
└── research/
    └── trials.ipynb
```

---

## Environment Variables

The Google API key is stored inside the `.env` file.

Example:

```text
GOOGLE_API_KEY=your_google_api_key
```

The API key should not be committed to a public Git repository.

---

## Running the Application

Activate the Conda environment:

```bash
conda activate genai311
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Summary

This project demonstrates the core concepts behind a modern Retrieval-Augmented Generation system.

It combines document processing, semantic embeddings, vector similarity search, conversational context, and large language models to transform static PDF documents into an interactive question-answering system.