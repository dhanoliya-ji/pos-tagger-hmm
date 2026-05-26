#!/usr/bin/env python
"""Train HMM POS Tagger on large datasets with progress tracking and memory optimization"""

import nltk
import pickle
import sys
from tqdm import tqdm

from hmm import HMMPOSTagger

# Download required datasets
print("Downloading NLTK datasets...\n")
nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)
nltk.download('treebank', quiet=True)

from nltk.corpus import brown, treebank

print("\n" + "=" * 60)
print("LOADING DATASETS")
print("=" * 60)

# Load Brown corpus (large dataset)
print("\nLoading Brown corpus...")
brown_sentences = brown.tagged_sents(tagset='universal')
print(f"✓ Brown corpus: {len(brown_sentences):,} sentences")

# Load Penn Treebank (additional dataset)
print("Loading Penn Treebank...")
treebank_sentences = treebank.tagged_sents(tagset='universal')
print(f"✓ Penn Treebank: {len(treebank_sentences):,} sentences")

# Combine datasets
all_sentences = brown_sentences + treebank_sentences
total_sentences = len(all_sentences)

print(f"\n✓ Total sentences: {total_sentences:,}")

# Calculate vocabulary statistics
print("\nCalculating vocabulary statistics...")
all_words = []
total_words = 0

for sentence in tqdm(all_sentences, desc="Scanning sentences", unit="sent"):
    for word, tag in sentence:
        all_words.append(word.lower())
        total_words += 1

unique_words = set(all_words)
print(f"✓ Total words: {total_words:,}")
print(f"✓ Unique words: {len(unique_words):,}")

# Train-test split (80-20)
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

split = int(0.8 * total_sentences)
train_data = all_sentences[:split]

print(f"\nTrain set: {len(train_data):,} sentences")
print(f"Test set: {total_sentences - len(train_data):,} sentences")

# Create and train tagger
print("\nTraining HMM POS Tagger (batch mode)...")
tagger = HMMPOSTagger()

# Train in batches to optimize memory usage
batch_size = 1000
print(f"Training in batches of {batch_size:,} sentences...\n")

for i in tqdm(range(0, len(train_data), batch_size), desc="Training", unit="batch"):
    batch = train_data[i:i + batch_size]
    # Accumulate counts without computing probs each time
    tagger.train(batch, compute_probs=False)

# Compute probabilities once at the end
print("\nFinalizing model (computing probabilities)...")
tagger.finalize_training()

print("✓ Training completed!")

# Statistics
print("\n" + "=" * 60)
print("MODEL STATISTICS")
print("=" * 60)

print(f"✓ Total tags: {len(tagger.tags)}")
print(f"✓ Vocabulary size: {len(tagger.vocab):,}")
print(f"✓ Emission probabilities computed: {sum(len(v) for v in tagger.emission_probs.values()):,}")

# Save trained model
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

model_path = "../models/hmm_tagger_large.pkl"
print(f"\nSaving model to {model_path}...")

with open(model_path, "wb") as f:
    pickle.dump(tagger, f)

print("✓ Model saved successfully!")

# Test the model
print("\n" + "=" * 60)
print("TESTING MODEL")
print("=" * 60)

test_sentences_examples = [
    "The cat is running quickly",
    "I am very happy today",
    "He walks to the store",
    "She plays tennis on weekends",
    "Dogs like to play fetch",
]

print("\nSample predictions:\n")
for sentence in test_sentences_examples:
    words = sentence.split()
    result = tagger.viterbi(words)
    print(f"Sentence: {sentence}")
    print("Tags:")
    for word, tag in result:
        print(f"  {word:15} --> {tag}")
    print()

print("✓ Testing completed!")
print("\nTraining script finished successfully!")
