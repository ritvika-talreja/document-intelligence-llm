import streamlit as st
import pdfplumber
from transformers import pipeline, AutoTokenizer
import nltk
import fitz
import sentencepiece  # Explicit import for SentencePiece
import re  # For text cleaning
from io import BytesIO

# Download NLTK tokenizer
nltk.download('punkt')

# Initialize models with error handling
@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"Error initializing summarizer: {e}")
        return None

@st.cache_resource
def load_qa_pipeline():
    try:
        return pipeline("question-answering", model="deepset/roberta-base-squad2")
    except Exception as e:
        st.error(f"Error initializing QA pipeline: {e}")
        return None

@st.cache_resource
def load_qg_pipeline():
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        model_name = "valhalla/t5-base-qg-hl"
        
        # Use the slow tokenizer explicitly
        tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        return pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error initializing Question Generation pipeline: {e}")
        return None
        
# Load models
summarizer = load_summarizer()
qa_pipeline = load_qa_pipeline()
qg_pipeline = load_qg_pipeline()

# Helper functions
def extract_text_from_pdf_pymupdf(pdf_file):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        pdf_stream = BytesIO(pdf_file.read())
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF with PyMuPDF: {e}")
        return ""

def clean_text(text):
    """Clean and preprocess the extracted text."""
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces into one
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Remove control characters
    return text.strip()

def summarize_long_text(text, chunk_size=1000):
    """Breaks long text into chunks and summarizes each separately."""
    if summarizer:
        summaries = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            try:
                summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False, early_stopping=True)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                st.error(f"Error summarizing chunk: {e}")
        return ' '.join(summaries)
    return "Summarizer not available."

# Streamlit UI
st.title("PDF Summarizer and QA Tool")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    raw_text = extract_text_from_pdf_pymupdf(uploaded_file)
    cleaned_text = clean_text(raw_text)

    st.subheader("Extracted Text")
    st.text_area("Raw Text", raw_text, height=200)

    if st.button("Summarize Text"):
        summary = summarize_long_text(cleaned_text)
        st.subheader("Summary")
        st.write(summary)

    question = st.text_input("Ask a question about the document:")
    if question and qa_pipeline:
        answer = qa_pipeline({'question': question, 'context': cleaned_text})
        st.subheader("Answer")
        st.write(answer['answer'])
