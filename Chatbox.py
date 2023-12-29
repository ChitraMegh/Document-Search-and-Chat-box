from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer, pipeline
from datasets import load_dataset
from weaviate import Client
import gradio as gr

# Load the "squad" dataset as an example
dataset = load_dataset('squad')

# Assuming each document is in the "context" field
documents = dataset['train']['context'][:100]  # Use the first 100 documents for this example

# Connect to the Weaviate server
weaviate_url = "http://localhost:8080"
weaviate = Client(weaviate_url)
weaviate.schema.create_class("Document")

# Define the search function
def search_documents(query):
    results = weaviate.query.get("Document", {"text": {"operator": "text2vec", "value": query}})
    document_texts = [result['text'] for result in results['data']['concepts']]
    return document_texts

# Define the chat function
def chat_with_model(question):
    answer = pipeline("question-answering", model=model, tokenizer=tokenizer)(question=question, context=document_texts)
    return answer['answer']

# Load the fine-tuned LLM
model_name = "fine_tuned_model"
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Save the fine-tuned model
model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)

# Index documents in Weaviate
for doc_id, text in enumerate(documents):
    weaviate.data_object.create("Document", {"id": doc_id, "text": text})

# Create Gradio Interface
iface = gr.Interface(
    fn=[search_documents, chat_with_model],
    inputs=["text", "text"],
    outputs=["text", "text"],
    live=True,
)

# Launch the Gradio Interface
iface.launch()
