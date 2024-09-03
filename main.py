from flask import Flask, jsonify
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from talmud_query.talmud_query import talmud_query_v1, talmud_query_v2
from talmud_query.feedback import feedback_to_langsmith
import uuid
import asyncio
from functools import wraps

# Get environment variables
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

REACT_APP_URL = os.getenv("REACT_APP_URL")
BACKEND_JS_URL = os.getenv("BACKEND_JS_URL")
EXPECTED_ORIGIN = BACKEND_JS_URL

# Initialize Flask app
app = Flask(__name__)
# Configure CORS
cors_config = {
    "origins": [EXPECTED_ORIGIN],  # Use a list here
    "methods": ["GET", "POST"],
    "allow_headers": ["Content-Type", "Authorization"],
    "supports_credentials": True
}
CORS(app, **cors_config)  # Use the unpacking operator to pass the config

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.getenv('API_KEY'):
            return jsonify({"error": "Invalid API key"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/feedback', methods=['GET'])
@require_api_key
def query_feedback():
    score, comment, run_id = request.args.get("score"), request.args.get("comment"), request.args.get("run_id")
    
    if not score:
        return jsonify({"error": "Score or comment is required"}), 400
    if not run_id:
        return jsonify({"error": "Run ID is required"}), 400
    
    feedback_to_langsmith(run_id, score, comment)
    
    return jsonify({
        "success": True
    })

@app.route('/query', methods=['GET'])
@require_api_key
def query_talmud():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    response = talmud_query_v2(query)

    answer = response[0]["answer"] if response and response[0] else None
    relevant_passage_ids = response[0]["relevant_passage_ids"] if response and response[0] else None
    run_id = response[1] if response and response[1] else str(uuid.uuid4())

    return jsonify({
        "answer": answer,
        "relevant_passage_ids": relevant_passage_ids,
        "run_id": run_id
    })

@app.before_request
def before_request():
    if request.method == 'OPTIONS':
        return '', 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 5001)))