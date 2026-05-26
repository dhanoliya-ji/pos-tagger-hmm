# HMM POS Tagger

A Hidden Markov Model-based Part-of-Speech tagger trained on large datasets (Brown Corpus + Penn Treebank).

## Features

✅ **Trained on 40k+ sentences** (1.1M words)
✅ **High accuracy** (~87% on test set)
✅ **Universal POS tags** (12 tags)
✅ **Flask web interface**
✅ **Viterbi algorithm** for sequence prediction
✅ **Batch training** for memory efficiency
✅ **Fast inference** on new sentences

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Web App
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

### Train on Large Dataset
```bash
train.bat
```
Or manually:
```bash
cd notebooks
python train_large.py
```

### Evaluate Model
```bash
evaluate.bat
```
Or manually:
```bash
cd notebooks
python evaluate_large.py
```

## Model Performance

- **Overall Accuracy:** ~87%
- **Vocabulary Size:** 30,000+ unique words
- **POS Tags:** 12 universal tags (NOUN, VERB, ADJ, ADV, etc.)
- **Model Size:** ~18 MB
- **Training Time:** 5-15 minutes on large dataset

### Per-Tag Accuracy
```
DET   95%
NOUN  92%
CCONJ 89%
ADP   88%
PRON  90%
VERB  85%
ADJ   83%
ADV   81%
```

## Architecture

### HMM Components
- **Emission Probabilities:** P(word | tag)
- **Transition Probabilities:** P(tag_i | tag_j)
- **Viterbi Algorithm:** Dynamic programming for optimal path

### Training
- **Datasets:** Brown Corpus (37k) + Penn Treebank (3k)
- **Smoothing:** Laplace smoothing for unseen words
- **Heuristics:** Suffix patterns, capitalization detection

## API Usage

### Web Interface
```
GET / - Main tagging interface
POST / - Submit sentence for tagging
GET /info - Model information
```

### Python
```python
import pickle
from notebooks.hmm import HMMPOSTagger

with open("models/hmm_tagger_large.pkl", "rb") as f:
    tagger = pickle.load(f)

sentence = "The cat is running"
words = sentence.split()
result = tagger.viterbi(words)

for word, tag in result:
    print(f"{word} -> {tag}")
```

## Directory Structure

```
pos-tagger-hmm/
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
├── train.bat             # Quick training script
├── evaluate.bat          # Quick evaluation script
│
├── notebooks/
│   ├── hmm.py            # HMM tagger implementation
│   ├── train_large.py    # Large dataset training
│   ├── evaluate_large.py # Model evaluation
│   └── training.py       # Original training script
│
├── models/
│   └── hmm_tagger_large.pkl  # Trained model
│
├── static/
│   └── style.css         # Web UI styling
│
└── templates/
    └── index.html        # Web UI template
```

## Bug Fixes Applied

✅ **Fixed transition probability smoothing** - Proper Laplace smoothing
✅ **Fixed emission probability heuristics** - Better unknown word handling
✅ **Fixed Viterbi algorithm** - Proper error handling for edge cases
✅ **Optimized tokenization** - Preserves capitalization for proper nouns

## Future Improvements

- [ ] Support for domain-specific training data
- [ ] Bidirectional tagger for higher accuracy
- [ ] GPU acceleration for faster training
- [ ] REST API for integration
- [ ] More language support

## References

- Brown Corpus: https://www.nltk.org/howto/collocations.html
- Penn Treebank: https://www.nltk.org/api/nltk.corpus.html#nltk.corpus.treebank
- Viterbi Algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm

## License

MIT

## Author

Gajendra Dhanoliya
