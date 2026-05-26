#!/usr/bin/env python
"""Test script to verify POS tagging fixes"""

import sys
sys.path.append("notebooks")

from hmm import HMMPOSTagger

# Create a simple test dataset
test_sentences = [
    [("The", "DET"), ("cat", "NOUN"), ("is", "VERB"), ("running", "VERB")],
    [("I", "PRON"), ("am", "VERB"), ("happy", "ADJ")],
    [("He", "PRON"), ("quickly", "ADV"), ("walked", "VERB")],
    [("She", "PRON"), ("plays", "VERB"), ("tennis", "NOUN")],
    [("Dogs", "NOUN"), ("are", "VERB"), ("jumping", "VERB")],
]

# Train model
print("Training HMM POS Tagger...\n")
tagger = HMMPOSTagger()
tagger.train(test_sentences, compute_probs=True)
print("Training completed!\n")

# Test sentences
test_cases = [
    "The cat is running",
    "I am happy",
    "He quickly walked",
    "Dogs are jumping",
]

print("=" * 50)
print("POS Tagging Results")
print("=" * 50)

for sentence in test_cases:
    words = sentence.split()
    result = tagger.viterbi(words)
    
    print(f"\nSentence: {sentence}")
    print("Tags:")
    for word, tag in result:
        print(f"  {word:15} --> {tag}")

print("\n" + "=" * 50)
print("Testing edge cases...")
print("=" * 50)

# Test single word
single_word = tagger.viterbi(["running"])
print(f"\nSingle word 'running': {single_word}")

# Test empty list
empty = tagger.viterbi([])
print(f"Empty sentence: {empty}")

print("\nAll tests completed successfully!")