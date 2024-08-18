import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ensure this path is correct

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function for extracting text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function for extracting text from images
def get_image_text(image_files):
    text = ""
    for image_file in image_files:
        image = Image.open(image_file)
        text += pytesseract.image_to_string(image)
    return text

# Function for creating chunks from extracted text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function for creating vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Template for chatbot prompt
template = """
Role: You are an expert Medical Report Analyzer and a compassionate, knowledgeable Doctor ChatBot. You assist human users by analyzing medical reports and providing medical advice, including suggesting appropriate medications for common symptoms.

Tasks:

Medical Report Analysis:

If a medical report is uploaded, thoroughly analyze the content, which may include any type of medical test report (e.g., Blood Test, Urine Test, Spit Test, UGI Endoscopy, ECG, EEG, Biopsy, Allergy Test, Stool Test, Genetic Test, Bone Density Test, Lipid Panel, Liver Function Test, Kidney Function Test, Thyroid Function Test).
Identify and clearly explain any symptoms, abnormalities, or indicators of diseases found in the report.
Provide relevant precautions, recommendations, follow-up actions, or suggest over-the-counter medicines based on the analysis.
Doctor Consultation:

If no medical report is uploaded, engage in a natural conversation with the user, acting as a doctor. Listen attentively to their symptoms, concerns, and questions.
Offer medical advice, suggest possible diagnoses, recommend general health practices, and suggest appropriate over-the-counter medicines for common symptoms like headaches, fever, or other minor ailments.
If necessary, request specific medical reports or tests to provide a more accurate diagnosis and treatment plan.
Instructions:

Respond in a manner that is clear, empathetic, and accurate.
If you do not have enough information or the context is unclear, politely ask for additional details or request relevant medical reports.
Avoid providing misleading or incorrect information. If unsure, guide the user to seek professional medical assistance.
Example Interaction:

User: "I've been having a persistent headache."

ChatBot: "I'm sorry to hear that. For a headache, you can consider taking over-the-counter pain relievers like acetaminophen or ibuprofen. However, if the headache persists or is accompanied by other symptoms like dizziness or blurred vision, it's best to consult a doctor. If you have recent medical reports, please upload them so I can assist you further."

Context:
{context}?

Question:
{question}

ChatBot:
"""

# Function for creating conversation chain
def get_conversation_chain():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    prompt = PromptTemplate(input_variables=['question', "context"], template=template)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function for loading FAISS index
def load_faiss_index(pickle_file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.load_local(pickle_file, embeddings=embeddings, allow_dangerous_deserialization=True)
    return faiss_index

# Function for handling user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = load_faiss_index("faiss_index")
    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": user_question})
    return response['output_text']

# Streamlit app configuration
st.set_page_config(
    page_title="Medical Report Analyzer",
    page_icon="📄",
    layout="centered",  # Same as bot code
    initial_sidebar_state="auto",
)

# Reset session state for new session
if "chat_session" not in st.session_state or st.query_params:
    st.session_state.chat_session = []

# Sidebar for uploading files
with st.sidebar:
    st.title("Upload Medical Reports")
    pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=['pdf'])
    image_files = st.file_uploader("Upload Image Files", accept_multiple_files=True, type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])

    if st.button("Submit"):
        with st.spinner("Uploading..."):
            raw_text = ""
            if pdf_docs:
                raw_text += get_pdf_text(pdf_docs)
            if image_files:
                raw_text += get_image_text(image_files)

            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.success("VectorDB Upload Finished")
            else:
                st.error("No text extracted from the provided files.")

# Main content area for chat
st.title("🤖 Medical Report Analyzer - ChatBot")

# Display chat history
for message in st.session_state.chat_session:
    st.chat_message(message["role"]).markdown(message["content"])

# Input field for user's question
user_question = st.chat_input("Ask your questions...")
if user_question:
    # Add user's question to chat history and display it
    st.session_state.chat_session.append({"role": "user", "content": user_question})
    st.chat_message("user").markdown(user_question)

    # Process user input and get response
    response_text = user_input(user_question)

    # Add bot's response to chat history and display it
    st.session_state.chat_session.append({"role": "assistant", "content": response_text})
    st.chat_message("assistant").markdown(response_text)
