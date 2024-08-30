from flask import Flask, jsonify
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from talmud_query import from_query_to_answer

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


@app.route('/query', methods=['GET'])
def query_talmud():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    start_time = time.time()
    response = from_query_to_answer(query)
    elapsed_time = time.time() - start_time

    return jsonify({"response": response, "time_taken": elapsed_time})


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