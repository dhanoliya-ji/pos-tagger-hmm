import nltk
import pickle

from nltk.corpus import treebank

from hmm import HMMPOSTagger

# Download dataset
nltk.download('treebank')
nltk.download('universal_tagset')

# Load tagged sentences
sentences = treebank.tagged_sents(tagset='universal')

# Train-test split
split = int(0.8 * len(sentences))

train_data = sentences[:split]

# Create tagger model
tagger = HMMPOSTagger()

print("Training started...\n")

# Train model with probability computation
tagger.train(train_data, compute_probs=True)

print("Training completed!\n")

# Save trained model
with open("../models/hmm_tagger.pkl", "wb") as f:
    pickle.dump(tagger, f)

print("Model saved successfully!")