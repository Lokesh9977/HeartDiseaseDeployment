import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = joblib.load("model.pkl")
featureNames = joblib.load("feature_names.pkl")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        inputData = request.get_json(force=True)

        missingFields = [f for f in featureNames if f not in inputData]
        if missingFields:
            return jsonify({
                "error": f"Missing fields: {missingFields}",
                "required_fields": featureNames
            }), 400

        inputDf = pd.DataFrame([[inputData[f] for f in featureNames]], columns=featureNames)

        prediction = model.predict(inputDf)[0]
        probability = model.predict_proba(inputDf)[0][1]

        resultText = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": resultText,
            "probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
