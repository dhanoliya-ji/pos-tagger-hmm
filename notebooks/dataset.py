import nltk

# Download required datasets
nltk.download('treebank')
nltk.download('universal_tagset')

from nltk.corpus import treebank

# Load tagged sentences
sentences = treebank.tagged_sents(tagset='universal')

print("\nDataset loaded successfully!\n")

# Total sentences
print("Total sentences:", len(sentences))

# Sample sentence
print("\nSample tagged sentence:\n")

print(sentences[0])

# Vocabulary statistics
all_words = []

for sentence in sentences:
    for word, tag in sentence:
        all_words.append(word.lower())

unique_words = set(all_words)

print("\nVocabulary Size:", len(unique_words))