from flask import Flask, jsonify
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from talmud_query.talmud_query import talmud_query_v1, talmud_query_v2
from talmud_query.feedback import feedback_to_langsmith
import uuid
import asyncio

REACT_APP_URL = os.getenv("REACT_APP_URL")
EXPECTED_ORIGIN = REACT_APP_URL if REACT_APP_URL else "http://localhost:3000"

# Initialize Flask app
app = Flask(__name__)
# Configure CORS
cors_config = {
    "origins": EXPECTED_ORIGIN,
    "methods": ["GET", "POST"],
    "allow_headers": ["Content-Type", "Authorization"],
    "supports_credentials": True
}
CORS(app, origins=cors_config["origins"], methods=cors_config["methods"], allow_headers=cors_config["allow_headers"], supports_credentials=cors_config["supports_credentials"])


@app.route('/feedback', methods=['GET'])
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
def query_talmud():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    response = talmud_query_v1(query)

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

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 5001)))