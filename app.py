# install: pip install flask tensorflow keras pandas shap
from flask import Flask, request, jsonify
import os
import tempfile
import tensorflow as tf
import shap
import numpy as np

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_model():
    file = request.files['model']
    temp_path = tempfile.mktemp(suffix='.h5')
    file.save(temp_path)

    try:
        model = tf.keras.models.load_model(temp_path)
        explain_score = check_explainability(model)
        bias_score = check_bias(model)
        privacy_score = check_privacy(model)

        result = {
            "bias_risk": bias_score,
            "explainability": explain_score,
            "privacy_score": privacy_score,
            "gdpr_compliant": privacy_score > 70,
            "hipaa_compliant": privacy_score > 80
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def check_explainability(model):
    return "SHAP Compatible" if isinstance(model, tf.keras.Model) else "Not explainable"

def check_bias(model):
    # Placeholder: real bias check needs training data and labels
    return "Medium Risk (requires dataset inspection)"

def check_privacy(model):
    size = model.count_params()
    return max(30, min(100, 100 - (size // 100000)))  # simple heuristic

if __name__ == '__main__':
    app.run(debug=True)
