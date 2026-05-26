import pickle

# Load trained model
with open("../models/hmm_tagger.pkl", "rb") as f:
    tagger = pickle.load(f)

# Test sentence
sentence = "The dog is barking loudly"

# Tokenize sentence
words = sentence.split()

# Predict tags
result = tagger.viterbi(words)

print("\nPredicted Tags:\n")

for word, tag in result:
    print(f"{word} --> {tag}")