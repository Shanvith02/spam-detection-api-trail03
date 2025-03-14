from flask import Flask, request, jsonify
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask App
app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return "Spam Detection API is Running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Check if 'message' key exists in request
        if "message" not in data:
            return jsonify({"error": "Missing 'message' key in request"}), 400
        
        message = [data["message"]]
        message_vectorized = vectorizer.transform(message)
        prediction = model.predict(message_vectorized)
        
        result = "Spam" if prediction[0] == 1 else "Ham"
        
        return jsonify({"message": message[0], "prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
