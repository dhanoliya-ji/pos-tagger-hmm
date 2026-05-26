from flask import Flask, render_template, request
import pickle
import sys
import re
import os

# Add notebooks folder to Python path
sys.path.append("notebooks")

from hmm import HMMPOSTagger

# Initialize Flask app
app = Flask(__name__)

# =========================
# LOAD TRAINED MODEL
# =========================

# Try to load the large model first, fall back to small model
model_files = [
    "models/hmm_tagger_large.pkl",
    "models/hmm_tagger.pkl"
]

tagger = None
model_used = None

for model_file in model_files:
    if os.path.exists(model_file):
        try:
            with open(model_file, "rb") as f:
                tagger = pickle.load(f)
            model_used = model_file
            print(f"✓ Loaded model: {model_file}")
            break
        except Exception as e:
            print(f"✗ Failed to load {model_file}: {e}")
            continue

if tagger is None:
    print("ERROR: No trained model found!")
    print("Please run train.bat or train_large.py first")
    exit(1)

# =========================
# HOME ROUTE
# =========================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    sentence = ""

    if request.method == "POST":

        # Get input sentence
        sentence = request.form["sentence"]

        # Remove extra spaces
        sentence = sentence.strip()

        # Tokenize sentence (preserve case for proper noun detection)
        # Split on whitespace but preserve punctuation context
        words = sentence.split()

        # Predict tags only if words exist
        if words:
            result = tagger.viterbi(words)

    return render_template(
        "index.html",
        result=result,
        sentence=sentence,
        model_info=model_used
    )

# =========================
# MODEL INFO ROUTE
# =========================

@app.route("/info")
def info():
    """Return model information"""
    return {
        "model": model_used,
        "tags": len(tagger.tags),
        "vocabulary": len(tagger.vocab),
        "status": "ready"
    }

# =========================
# RUN APPLICATION
# =========================

if __name__ == "__main__":

    app.run(
        debug=True
    )