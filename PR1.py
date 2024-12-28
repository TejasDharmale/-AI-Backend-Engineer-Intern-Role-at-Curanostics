import os
import json
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Flask app
app = Flask(__name__)

# Initialize Elasticsearch
def initialize_elasticsearch():
    try:
        es = Elasticsearch("http://localhost:9200")  # Updated connection syntax
        if not es.ping():
            raise ConnectionError("Elasticsearch connection failed. Ensure Elasticsearch is running.")
        print("Connected to Elasticsearch.")
        return es
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        return None

es = initialize_elasticsearch()

# Load T5 model and tokenizer
def load_model():
    """Load pre-trained T5 model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        print("T5 model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        exit()

tokenizer, model = load_model()

def index_patient_data(file_path, index_name):
    """
    Index patient data into Elasticsearch.
    Args:
        file_path: Path to the patient data JSON file.
        index_name: Name of the Elasticsearch index.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    with open(file_path, 'r') as file:
        data = json.load(file)
        for record in data:
            es.index(index=index_name, body=record)
    print(f"Data indexed successfully into '{index_name}'.")

def generate_summary(text):
    """
    Generate a summary from input text using the T5 model.
    Args:
        text: The input text to summarize.
    Returns:
        A summarized version of the text.
    """
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/retrieve', methods=['POST'])
def retrieve():
    """
    Retrieve patient data from Elasticsearch.
    Request JSON:
        {
            "query": "Search query for patient data"
        }
    """
    if not es:
        return jsonify({"error": "Elasticsearch connection not initialized."}), 500

    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    search_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["name", "conditions", "diagnostics"]
            }
        }
    }

    try:
        results = es.search(index="patients", body=search_body)
        return jsonify(results["hits"]["hits"])
    except Exception as e:
        return jsonify({"error": f"Error retrieving data: {str(e)}"}), 500

@app.route('/generate-summary', methods=['POST'])
def generate_summary_api():
    """
    Generate a summary for retrieved patient data.
    Request JSON:
        {
            "content": "Raw medical test data to summarize"
        }
    """
    content = request.json.get("content", "")
    if not content:
        return jsonify({"error": "Content is required"}), 400

    try:
        summary = generate_summary(content)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"Error generating summary: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({"status": "OK", "message": "The server is running."}), 200

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint to provide information about the API.
    """
    return jsonify({
        "message": "Welcome to the RAG Model API for patient data summarization.",
        "endpoints": {
            "/retrieve": "POST - Retrieve patient data from Elasticsearch.",
            "/generate-summary": "POST - Generate a summary for retrieved patient data.",
            "/health": "GET - Check the health of the server."
        }
    }), 200

if __name__ == "__main__":
    print("Registered routes:")
    print(app.url_map)

    # Optionally index data (uncomment and replace file path if needed)
    # index_patient_data("path_to_patient_data.json", "patients")

    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as e:
        print(f"Error starting Flask application: {e}")
