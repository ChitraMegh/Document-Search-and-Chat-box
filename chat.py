import streamlit as st
import pytesseract

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline, GPT2LMHeadModel, GPT2Tokenizer
from pytesseract import image_to_string
from PIL import Image
from datasets import load_dataset

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Load the SQuAD dataset
squad_dataset = load_dataset("squad")
train_dataset = squad_dataset["train"]

# Load DistilBERT model for question-answering
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

# Load GPT-2 model for chat
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to perform OCR on images
def perform_ocr(image_path):
    img = Image.open(image_path)
    text = image_to_string(img)
    return text

# Function to perform Squad question-answering
def perform_squad_query(query):
    # Example Squad context
    squad_context = "In 1858, the Virgin Mary allegedly appeared to Bernadette Soubirous in Lourdes, France."

    # Perform the question-answering task
    answer = qa_pipeline(question=query, context=squad_context)
    return f"Answer: {answer['answer']}"

# Define chat function
def generate_response(prompt, context=None):
    full_prompt = context + " " + prompt if context else prompt
    inputs = gpt_tokenizer.encode(full_prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = gpt_model.generate(inputs, max_length=100, num_beams=2, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit app
def main():
    st.title("Document Search and Chat System")

    # Example SQuAD context
    squad_context = "In 1858, the Virgin Mary allegedly appeared to Bernadette Soubirous in Lourdes, France."

    #  search interface
    search_query = st.text_input("Enter your search query:")
    if st.button("Search"):
        # Perform Squad-based question-answering on the search query
        search_results = perform_squad_query(search_query)
        st.write(search_results)

    # Chat interface
    chat_input = st.text_input("Chat:")
    if st.button("Send"):
        # Use the language model to generate a response based on the user's input
        response = generate_response(chat_input, context=squad_context)
        st.text(response)

    # OCR interface for handling scanned documents
    uploaded_file = st.file_uploader("Upload a scanned document", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        text_from_ocr = perform_ocr(uploaded_file)
        st.text("Text extracted using OCR:")
        st.text(text_from_ocr)

if __name__ == "__main__":
    main()
