# Patient Data Summarization and Retrieval API

## Overview
This repository implements a Flask-based API for efficient search and summarization of patient medical data. It combines the power of Elasticsearch for data retrieval and Hugging Face's T5 model for text summarization.

---

## Features
- **Data Retrieval**: Search patient data with Elasticsearch using multi-field matching.
- **Summarization**: Generate concise summaries of medical records using the T5 model.
- **Health Check**: Verify the server status with a dedicated endpoint.
- **API Documentation**: Endpoints for all operations, including retrieval and summarization.

---

## Endpoints
- `/retrieve` (POST): Search and retrieve patient data from Elasticsearch.
- `/generate-summary` (POST): Generate summaries for medical data.
- `/health` (GET): Check server health.
- `/` (GET): API documentation and details.

---

## Technologies Used
- **Flask**: Web framework for building the API.
- **Elasticsearch**: Data storage and retrieval engine.
- **Hugging Face Transformers**: Text summarization with the T5 model.
- **Python**: Primary programming language.

---

## Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
