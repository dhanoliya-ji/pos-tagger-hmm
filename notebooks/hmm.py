import math
from collections import defaultdict


class HMMPOSTagger:

    def __init__(self):

        # Probabilities
        self.transition_probs = {}
        self.emission_probs = {}

        # Counts
        self.transition_counts = {}
        self.emission_counts = {}

        self.tag_counts = defaultdict(int)

        self.vocab = set()
        self.tags = set()

    # ====================================
    # TRAIN MODEL
    # ====================================

    def train(self, tagged_sentences, compute_probs=False):
        """
        Train HMM POS Tagger on tagged sentences.
        
        Args:
            tagged_sentences: List of list of (word, tag) tuples
            compute_probs: If True, compute probabilities after training.
                          Set to False for batch training, then call 
                          finalize_training() once at the end.
        """

        for sentence in tagged_sentences:

            prev_tag = "<START>"

            for word, tag in sentence:

                word = word.lower()

                self.vocab.add(word)
                self.tags.add(tag)

                # Initialize dictionaries

                if tag not in self.emission_counts:
                    self.emission_counts[tag] = {}

                if prev_tag not in self.transition_counts:
                    self.transition_counts[prev_tag] = {}

                if word not in self.emission_counts[tag]:
                    self.emission_counts[tag][word] = 0

                if tag not in self.transition_counts[prev_tag]:
                    self.transition_counts[prev_tag][tag] = 0

                # Increment counts

                self.emission_counts[tag][word] += 1
                self.transition_counts[prev_tag][tag] += 1

                self.tag_counts[tag] += 1

                prev_tag = tag

            # END state

            if prev_tag not in self.transition_counts:
                self.transition_counts[prev_tag] = {}

            if "<END>" not in self.transition_counts[prev_tag]:
                self.transition_counts[prev_tag]["<END>"] = 0

            self.transition_counts[prev_tag]["<END>"] += 1

        # Compute probabilities only if requested
        if compute_probs:
            self.finalize_training()

    # ====================================
    # FINALIZE TRAINING
    # ====================================

    def finalize_training(self):
        """Compute transition and emission probabilities from counts."""

        # ====================================
        # TRANSITION PROBABILITIES
        # ====================================

        for prev_tag in self.transition_counts:

            self.transition_probs[prev_tag] = {}

            total = sum(
                self.transition_counts[prev_tag].values()
            )

            num_tags = len(self.tags)

            for tag in self.tags:

                count = self.transition_counts[
                    prev_tag
                ].get(tag, 0)

                # Only apply smoothing for tags that actually exist
                if tag in self.transition_counts[prev_tag] or count > 0:
                    self.transition_probs[prev_tag][tag] = (
                        (count + 1) / (total + num_tags)
                    )
                else:
                    self.transition_probs[prev_tag][tag] = (
                        1 / (total + num_tags)
                    )

        # ====================================
        # EMISSION PROBABILITIES
        # ====================================

        for tag in self.emission_counts:

            self.emission_probs[tag] = {}

            total = sum(
                self.emission_counts[tag].values()
            )

            vocab_size = len(self.vocab)

            for word in self.emission_counts[tag]:

                count = self.emission_counts[tag][word]

                self.emission_probs[tag][word] = (
                    (count + 1) / (total + vocab_size)
                )

    # ====================================
    # EMISSION PROBABILITY
    # ====================================

    def emission_prob(self, tag, word):

        original_word = word

        word = word.lower()

        # Known words

        if (
            tag in self.emission_probs
            and word in self.emission_probs[tag]
        ):

            return self.emission_probs[tag][word]

        # Unknown word heuristics - prioritize more specific patterns

        # Check digits first
        if word.isdigit() and tag == "NUM":
            return 0.01

        # Verb patterns
        if tag == "VERB":
            if word.endswith("ing") or word.endswith("ed"):
                return 0.008
            return 1e-8

        # Adverb patterns
        if tag == "ADV":
            if word.endswith("ly"):
                return 0.008
            return 1e-8

        # Noun patterns - proper nouns
        if tag == "NOUN":
            if original_word and original_word[0].isupper():
                return 0.008
            return 1e-8

        # Default low probability for all other tags
        return 1e-8

    # ====================================
    # TRANSITION PROBABILITY
    # ====================================

    def transition_prob(self, prev_tag, tag):

        if (
            prev_tag in self.transition_probs
            and tag in self.transition_probs[prev_tag]
        ):

            return self.transition_probs[prev_tag][tag]

        return 1e-8

    # ====================================
    # VITERBI ALGORITHM
    # ====================================

    def viterbi(self, sentence):

        if not sentence or not self.tags:
            return []

        V = [{}]
        path = {}

        # Initialization

        for tag in self.tags:

            transition = self.transition_prob(
                "<START>",
                tag
            )

            emission = self.emission_prob(
                tag,
                sentence[0]
            )

            V[0][tag] = (
                math.log(transition)
                + math.log(emission)
            )

            path[tag] = [tag]

        # Handle single word case
        if len(sentence) == 1:
            best_tag = max(
                self.tags,
                key=lambda tag: V[0].get(tag, float('-inf'))
            )
            return [(sentence[0], best_tag)]

        # Recursion

        for t in range(1, len(sentence)):

            V.append({})
            new_path = {}

            for curr_tag in self.tags:

                best_prob = float('-inf')
                best_state = None

                for prev_tag in self.tags:

                    transition_prob = self.transition_prob(
                        prev_tag,
                        curr_tag
                    )

                    emission_prob = self.emission_prob(
                        curr_tag,
                        sentence[t]
                    )

                    # Avoid log(0) errors
                    if transition_prob > 0 and emission_prob > 0:
                        prob = (
                            V[t - 1][prev_tag]
                            + math.log(transition_prob)
                            + math.log(emission_prob)
                        )

                        if prob > best_prob:
                            best_prob = prob
                            best_state = prev_tag

                if best_state is not None:
                    V[t][curr_tag] = best_prob
                    new_path[curr_tag] = (
                        path[best_state] + [curr_tag]
                    )

            path = new_path

        # Termination

        best_prob = float('-inf')
        best_state = None

        for tag in self.tags:

            if tag in V[len(sentence) - 1]:
                transition_prob = self.transition_prob(
                    tag,
                    "<END>"
                )

                if transition_prob > 0:
                    prob = (
                        V[len(sentence) - 1][tag]
                        + math.log(transition_prob)
                    )

                    if prob > best_prob:
                        best_prob = prob
                        best_state = tag

        if best_state is None:
            # Fallback: pick tag with highest probability
            best_state = max(
                path.keys(),
                key=lambda tag: V[len(sentence) - 1].get(tag, float('-inf'))
            )

        return list(zip(sentence, path[best_state]))
