from flask import Flask, jsonify
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from talmud_query import from_query_to_answer, feedback_to_langsmith
import uuid

REACT_APP_URL = os.getenv("REACT_APP_URL")
expected_origin = REACT_APP_URL if REACT_APP_URL else "http://localhost:3000"

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
cors_config = {
    "origins": expected_origin,
    "methods": ["GET", "POST"],
    "allow_headers": ["Content-Type", "Authorization"],
    "supports_credentials": True
}
CORS(app, origins=cors_config["origins"], methods=cors_config["methods"], allow_headers=cors_config["allow_headers"], supports_credentials=cors_config["supports_credentials"])

@app.route('/feedback', methods=['GET'])
def query_feedback():
    score = request.args.get("score")
    comment = request.args.get("comment")
    run_id = request.args.get("run_id")
    
    if not score:
        return jsonify({"error": "Score or comment is required"}), 400
    if not run_id:
        return jsonify({"error": "Run ID is required"}), 400
    
    # Save feedback to database
    x = feedback_to_langsmith(run_id, score, comment)
    print(x)
    return jsonify({
        "success": True
    })

@app.route('/query', methods=['GET'])
def query_talmud():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    start_time = time.time()
    response = from_query_to_answer(query)
    elapsed_time = time.time() - start_time
    print(f"Query took {elapsed_time} seconds")

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