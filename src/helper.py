import os
import time
import certifi

from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["SSL_CERT_FILE"] = certifi.where()


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def get_text_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    return chunks


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )


# --------------------------------------------------
# VECTOR STORE
# --------------------------------------------------

def get_vector_store(text_chunks):

    embeddings = get_embeddings()

    vectors = embeddings.embed_documents(
        text_chunks,
        batch_size=40,
        task_type="RETRIEVAL_DOCUMENT"
    )

    text_embeddings = list(
        zip(text_chunks, vectors)
    )

    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings
    )

    return vector_store


# --------------------------------------------------
# LLM
# --------------------------------------------------

def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0
    )


# --------------------------------------------------
# RETRIEVE DOCUMENTS
# --------------------------------------------------

def retrieve_documents(vector_store, question):

    question = str(question).strip()

    if not question:
        return []

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    for attempt in range(3):

        try:

            return retriever.invoke(
                question
            )

        except Exception as e:

            print(
                f"Retrieval attempt "
                f"{attempt + 1} failed:",
                e
            )

            if attempt == 2:
                raise

            time.sleep(
                2 * (attempt + 1)
            )


# --------------------------------------------------
# CLEAN GEMINI RESPONSE
# --------------------------------------------------

def extract_text_from_response(response):

    content = response.content

    # Normal string response
    if isinstance(content, str):
        return content

    # Structured content blocks
    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":

                    text = block.get(
                        "text",
                        ""
                    )

                    if text:
                        text_parts.append(text)

            elif isinstance(block, str):

                text_parts.append(block)

        return "\n".join(text_parts)

    return str(content)


# --------------------------------------------------
# ANSWER QUESTION
# --------------------------------------------------

def answer_question(
    vector_store,
    question,
    chat_history
):

    # ----------------------------------------------
    # Retrieve relevant PDF chunks
    # ----------------------------------------------

    docs = retrieve_documents(
        vector_store,
        question
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # ----------------------------------------------
    # Previous conversation
    # ----------------------------------------------

    history_text = ""

    # Keep only recent messages so prompt
    # does not become unnecessarily large
    recent_history = chat_history[-6:]

    for message in recent_history:

        role = message["role"]
        content = message["content"]

        history_text += (
            f"{role}: {content}\n"
        )


    # ----------------------------------------------
    # PROMPT
    # ----------------------------------------------

    prompt = f"""
You are an assistant answering questions about an uploaded PDF.

Use only the information contained in the PDF context below.

If the answer cannot be found in the document,
say that the information is not available in the document.

Previous conversation:
{history_text}

PDF context:
{context}

Current question:
{question}

Answer clearly and concisely.
"""


    llm = get_llm()


    # ----------------------------------------------
    # CALL GEMINI WITH RETRIES
    # ----------------------------------------------

    for attempt in range(3):

        try:

            response = llm.invoke(
                prompt
            )

            answer = extract_text_from_response(
                response
            )

            return answer

        except Exception as e:

            print(
                f"LLM attempt "
                f"{attempt + 1} failed:",
                e
            )

            if attempt == 2:
                raise

            time.sleep(
                2 * (attempt + 1)
            )