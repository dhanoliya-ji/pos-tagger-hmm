#!/usr/bin/env python
"""Evaluate HMM POS Tagger on test data with detailed metrics"""

import nltk
import pickle
from tqdm import tqdm

# Download datasets
nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)
nltk.download('treebank', quiet=True)

from nltk.corpus import brown, treebank

print("\n" + "=" * 60)
print("LOADING DATASETS")
print("=" * 60)

# Load datasets
brown_sentences = brown.tagged_sents(tagset='universal')
treebank_sentences = treebank.tagged_sents(tagset='universal')
all_sentences = brown_sentences + treebank_sentences

total_sentences = len(all_sentences)
split = int(0.8 * total_sentences)

test_data = all_sentences[split:]

print(f"\nTest set size: {len(test_data):,} sentences")

# Load trained model
print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)

model_path = "../models/hmm_tagger_large.pkl"
print(f"\nLoading model from {model_path}...")

try:
    with open(model_path, "rb") as f:
        tagger = pickle.load(f)
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model not found at {model_path}")
    print("Please run train_large.py first.")
    exit(1)

# Evaluate model
print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

correct = 0
total = 0
tag_correct = {}
tag_total = {}

print("\nEvaluating on test set...\n")

for sentence in tqdm(test_data, desc="Evaluating", unit="sent"):

    words = [word for word, tag in sentence]

    predicted = tagger.viterbi(words)

    for (_, pred_tag), (_, true_tag) in zip(predicted, sentence):

        if pred_tag == true_tag:
            correct += 1
        
        # Per-tag statistics
        if true_tag not in tag_total:
            tag_total[true_tag] = 0
            tag_correct[true_tag] = 0
        
        tag_total[true_tag] += 1
        if pred_tag == true_tag:
            tag_correct[true_tag] += 1

        total += 1

accuracy = (correct / total) * 100

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"\nOverall Accuracy: {accuracy:.2f}%")
print(f"Correct: {correct:,} / {total:,}")

print("\n" + "-" * 60)
print("Per-Tag Accuracy:")
print("-" * 60)

# Sort by accuracy (descending)
sorted_tags = sorted(
    tag_total.items(),
    key=lambda x: (tag_correct[x[0]] / x[1]) if x[1] > 0 else 0,
    reverse=True
)

for tag, total_count in sorted_tags:
    correct_count = tag_correct[tag]
    tag_accuracy = (correct_count / total_count) * 100
    print(f"{tag:8} {tag_accuracy:6.2f}%  ({correct_count:,}/{total_count:,})")

print("\n✓ Evaluation completed!")
