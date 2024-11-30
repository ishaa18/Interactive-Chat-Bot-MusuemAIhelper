from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
from transformers import pipeline

# Initialize Flask app and allow CORS
app = Flask(__name__)
CORS(app)

# Load FAQ data
with open('faqs1.json', 'r') as f:
    faq_data = json.load(f)

# Initialize the Hugging Face QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Define a function to get the answer
def answer_question(user_question):
    # Step 1: Check for exact match in FAQ
    for faq in faq_data["faqs"]:
        if faq["question"].strip().lower() == user_question.strip().lower():
            return faq["answer"]

    # Step 2: Use QA model with combined FAQ answers as context
    combined_context = " ".join([faq["answer"] for faq in faq_data["faqs"]])
    response = qa_pipeline(question=user_question, context=combined_context)

    # Step 3: Filter based on a confidence threshold
    if response['score'] > 0.7:  # Confidence threshold
        return response['answer']
    
    # Step 4: Return fallback response
    return "I'm sorry, I couldn't find a precise answer to your question. Can you try rephrasing it?"

# Flask endpoint for chatbot
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    answer = answer_question(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
