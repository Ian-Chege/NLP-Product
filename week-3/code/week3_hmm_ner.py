# =============================================================================
# Week 3 – Hidden Markov Models and Sequence Labeling
# =============================================================================
#
# This script covers two connected NLP concepts:
#
#   1. Hidden Markov Models (HMMs)
#      An HMM models sequences of words as outputs from hidden POS tag states.
#      It learns:
#        - transition probabilities: how likely one tag follows another
#        - emission probabilities: how likely a word belongs to a given tag
#
#      NOTE: NLTK's built-in HMM tagger has a known numerical overflow bug on
#      Python 3.9+ that causes it to default every word to a single tag.
#      We implement our own HMM instead — it is numerically stable and shows
#      exactly how the algorithm works.
#
#      Training data: MACULA Greek NT — the most domain-appropriate available
#      dataset for biblical text. Expert POS tags for every Greek word + gloss.
#
#   2. Named Entity Recognition (NER)
#      Identifies real-world entities (people, places, organisations) in text.
#
# =============================================================================

import os
import csv
import re
import math
import urllib.request
from collections import Counter

import nltk
nltk.download("punkt",                      quiet=True)
nltk.download("punkt_tab",                  quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("maxent_ne_chunker",          quiet=True)
nltk.download("maxent_ne_chunker_tab",      quiet=True)
nltk.download("words",                      quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk


# =============================================================================
# PART A – OUR OWN HMM IMPLEMENTATION
# =============================================================================
# We build a simple but fully working HMM tagger from scratch.
# All probability calculations are done in log space (using math.log) to
# avoid the underflow/overflow problems that break NLTK's implementation.
#
# Three tables are learned from training data:
#   start       P(first tag in a sentence)
#   transitions P(current tag | previous tag)
#   emissions   P(word | tag)
# =============================================================================

class BiblicalHMM:
    """A simple, numerically stable Hidden Markov Model POS tagger."""

    def train(self, sentences):
        """
        Learn probabilities from a list of labeled sentences.
        Each sentence is a list of (word, tag) tuples.
        """
        self.start       = Counter()   # how often each tag starts a sentence
        self.transitions = {}          # {prev_tag: Counter({next_tag: count})}
        self.emissions   = {}          # {tag: Counter({word: count})}
        self.tag_totals  = Counter()   # total words per tag

        for sent in sentences:
            if not sent:
                continue
            # Count the starting tag
            self.start[sent[0][1]] += 1

            prev_tag = None
            for word, tag in sent:
                word = word.lower()

                # Emission counts
                if tag not in self.emissions:
                    self.emissions[tag] = Counter()
                self.emissions[tag][word] += 1
                self.tag_totals[tag] += 1

                # Transition counts
                if prev_tag is not None:
                    if prev_tag not in self.transitions:
                        self.transitions[prev_tag] = Counter()
                    self.transitions[prev_tag][tag] += 1

                prev_tag = tag

        self.tags = list(self.tag_totals.keys())

    def _log_emission(self, tag, word):
        """Log probability of a word being emitted by a tag. Uses add-1 smoothing."""
        tag_total  = self.tag_totals[tag]
        word_count = self.emissions.get(tag, {}).get(word, 0)
        vocab_size = len(set(w for c in self.emissions.values() for w in c))
        # Add-1 (Laplace) smoothing so unseen words get a small probability
        return math.log((word_count + 1) / (tag_total + vocab_size))

    def _log_transition(self, prev_tag, tag):
        """Log probability of tag following prev_tag."""
        counts = self.transitions.get(prev_tag, {})
        total  = sum(counts.values()) or 1
        count  = counts.get(tag, 0)
        return math.log(max(count / total, 1e-10))

    def _log_start(self, tag):
        """Log probability of a tag starting a sentence."""
        total = sum(self.start.values()) or 1
        return math.log(max(self.start.get(tag, 0) / total, 1e-10))

    def tag(self, tokens):
        """
        Run the Viterbi algorithm to find the most likely tag sequence.
        Viterbi works by building the best path one word at a time,
        keeping only the highest-probability path to each tag at each step.
        """
        if not tokens:
            return []

        # viterbi[tag] = best log-probability to reach this tag at current word
        # backpointer[tag] = the tag sequence that led here
        viterbi     = {tag: self._log_start(tag) + self._log_emission(tag, tokens[0].lower())
                       for tag in self.tags}
        backpointer = {tag: [tag] for tag in self.tags}

        for token in tokens[1:]:
            word         = token.lower()
            new_viterbi  = {}
            new_backptr  = {}

            for tag in self.tags:
                emit = self._log_emission(tag, word)
                # Find the best previous tag to transition from
                best_prev, best_score = max(
                    ((prev, viterbi[prev] + self._log_transition(prev, tag))
                     for prev in self.tags),
                    key=lambda x: x[1]
                )
                new_viterbi[tag] = best_score + emit
                new_backptr[tag] = backpointer[best_prev] + [tag]

            viterbi     = new_viterbi
            backpointer = new_backptr

        # Pick the tag sequence with the highest final probability
        best_tag = max(self.tags, key=lambda t: viterbi[t])
        return list(zip(tokens, backpointer[best_tag]))


# =============================================================================
# PART B – DOWNLOAD AND PARSE MACULA GREEK NT TRAINING DATA
# =============================================================================

MACULA_URL = (
    "https://raw.githubusercontent.com/Clear-Bible/macula-greek"
    "/main/Nestle1904/tsv/macula-greek-Nestle1904.tsv"
)
CACHE_PATH = os.path.join(os.path.dirname(__file__), "macula-greek-NT.tsv")

if not os.path.exists(CACHE_PATH):
    print("Downloading MACULA Greek NT (one-time, ~19 MB)... ", end="", flush=True)
    urllib.request.urlretrieve(MACULA_URL, CACHE_PATH)
    print("done.")
else:
    print("Using cached MACULA Greek NT data.")


def macula_class_to_pos(word_class, word_type):
    """Map MACULA class/type fields to Penn Treebank-style POS tags."""
    if word_class == "noun":
        return "NNP" if word_type == "proper" else "NN"
    return {"verb": "VB", "adj": "JJ", "adv": "RB", "conj": "CC",
            "prep": "IN", "det": "DT", "pron": "PRP", "ptcl": "RP",
            "num": "CD", "intj": "UH"}.get(word_class, "NN")


def clean_gloss(gloss):
    """Extract the core English word from a MACULA gloss phrase."""
    gloss = re.sub(r'\[.*?\]', '', gloss).strip()   # remove [optional] words
    words = [w for w in gloss.split() if w.isalpha()]
    return words[-1].lower() if words else None      # take last meaningful word


print("\n" + "=" * 60)
print("Fig 5 – DATASET USED FOR TRAINING AND TESTING")
print("=" * 60)
print("Training : MACULA Greek NT — expert-annotated New Testament")
print("           (all NT books except Jude, trained word-for-word)")
print("           Source: github.com/Clear-Bible/macula-greek")
print("Testing  : Book of Jude, verses 1–7 (KJV)\n")

train_sentences, jude_sentences = [], []
current_verse, current_sent, current_book = None, [], None

with open(CACHE_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        ref        = row["ref"]
        book_verse = ref.rsplit("!", 1)[0]
        book       = ref.split()[0]
        word_class = row["class"].strip()
        word_type  = row["type"].strip()
        gloss      = row["gloss"].strip()

        if not word_class or not gloss:
            continue
        word = clean_gloss(gloss)
        if not word:
            continue

        if book_verse != current_verse:
            if current_sent:
                (jude_sentences if current_book == "JUD" else train_sentences).append(current_sent)
            current_sent, current_verse, current_book = [], book_verse, book

        current_sent.append((word, macula_class_to_pos(word_class, word_type)))

if current_sent:
    (jude_sentences if current_book == "JUD" else train_sentences).append(current_sent)

print(f"Training sentences (NT minus Jude) : {len(train_sentences)}")
print(f"Jude verses (held-out test)        : {len(jude_sentences)}")
print(f"\nSample training sentences:")
print("-" * 60)
for sent in train_sentences[-3:]:
    print("  " + " ".join(f"{w}/{t}" for w, t in sent[:9]) + " ...")


# =============================================================================
# TRAIN AND TEST
# =============================================================================

print("\n" + "=" * 60)
print("Fig 1 – SEQUENCE LABELING CODE")
print("=" * 60)
print("""
  # Our own HMM — trains transition + emission probabilities from MACULA data
  tagger = BiblicalHMM()
  tagger.train(train_sentences)

  # Each sentence is a list of (english_gloss, pos_tag) tuples
  # e.g. [('jude', 'NNP'), ('servant', 'NN'), ('christ', 'NNP'), ...]

  # Viterbi decoding finds the most likely tag sequence for new text
  tokens = word_tokenize(sentence)
  tagged = tagger.tag(tokens)
""")

print("Training HMM on MACULA Greek NT... ", end="", flush=True)
tagger = BiblicalHMM()
tagger.train(train_sentences)
print("done.")

jude_text = (
    "Jude, the servant of Jesus Christ, and brother of James, to them that are "
    "sanctified by God the Father, and preserved in Jesus Christ, and called: "
    "Mercy unto you, and peace, and love, be multiplied. "
    "Beloved, when I gave all diligence to write unto you of the common salvation, "
    "it was needful for me to write unto you, and exhort you that ye should earnestly "
    "contend for the faith which was once delivered unto the saints. "
    "For there are certain men crept in unawares, who were before of old ordained to "
    "this condemnation, ungodly men, turning the grace of our God into lasciviousness, "
    "and denying the only Lord God, and our Lord Jesus Christ."
)

READABLE = {
    "NN": "Noun",      "NNP": "Proper Noun", "NNS": "Noun",  "NNPS": "Proper Noun",
    "VB": "Verb",      "VBD": "Verb",        "VBG": "Verb",  "VBN": "Verb",
    "VBP": "Verb",     "VBZ": "Verb",
    "JJ": "Adjective", "RB": "Adverb",       "CC": "Conjunction",
    "IN": "Preposition","DT": "Determiner",  "PRP": "Pronoun",
    "RP": "Particle",  "CD": "Number",       "UH": "Interjection",
}

sentences      = sent_tokenize(jude_text)
first_sentence = sentences[0]
tokens         = word_tokenize(first_sentence)
hmm_tagged     = tagger.tag(tokens)

print("\n" + "=" * 60)
print("Fig 2 – HIDDEN MARKOV MODEL OUTPUT")
print("=" * 60)
print(f"Input:\n  \"{first_sentence}\"\n")
print(f"{'Word':<18} {'HMM Tag':<10} {'Label'}")
print("-" * 45)
for word, tag in hmm_tagged:
    print(f"  {word:<16} {tag:<10} {READABLE.get(tag, 'Other')}")


# =============================================================================
# NAMED ENTITY RECOGNITION
# =============================================================================

print("\n" + "=" * 60)
print("Fig 3 – NAMED ENTITY RECOGNITION EXAMPLE")
print("=" * 60)
print(f"Input: \"{first_sentence}\"\n")

ner_tree       = ne_chunk(pos_tag(word_tokenize(first_sentence)))
entities_found = []
for chunk in ner_tree:
    if hasattr(chunk, "label"):
        name = " ".join(w for w, _ in chunk)
        entities_found.append((name, chunk.label()))
        print(f"  [{chunk.label()}]  →  '{name}'")
if not entities_found:
    print("  (No named entities found)")


print("\n" + "=" * 60)
print("Fig 4 – SENTENCE LABELING RESULTS (full passage)")
print("=" * 60)

all_entities = []
for i, sent in enumerate(sentences, 1):
    tree          = ne_chunk(pos_tag(word_tokenize(sent)))
    sent_entities = [((" ".join(w for w, _ in c)), c.label())
                     for c in tree if hasattr(c, "label")]
    all_entities += sent_entities
    status = ", ".join(f"{n} [{t}]" for n, t in sent_entities) or "no entities"
    print(f"  Sentence {i}: {status}")

print(f"\n  Total entities found: {len(all_entities)}")
seen = set()
for name, label in all_entities:
    if (name.lower(), label) not in seen:
        seen.add((name.lower(), label))
        print(f"    {name:<22} [{label}]")


# =============================================================================
# IMPROVED NER – spaCy (en_core_web_sm)
# =============================================================================
# NLTK's ne_chunk was trained on news text, causing biblical titles like
# "Lord Jesus Christ" to be misclassified as ORGANIZATION.
# spaCy's en_core_web_sm model is trained on a much larger, more diverse
# corpus (OntoNotes 5) and handles proper names significantly better.
#
# This section shows the same NER task run through spaCy for comparison.
# =============================================================================

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")

    print("\n" + "=" * 60)
    print("BONUS – IMPROVED NER WITH spaCy (en_core_web_sm)")
    print("=" * 60)
    print("Comparison: same passage run through spaCy instead of NLTK ne_chunk\n")
    print(f"Input: \"{first_sentence}\"\n")

    doc            = nlp(first_sentence)
    spacy_entities = [(ent.text, ent.label_) for ent in doc.ents]

    print("spaCy NER results (first sentence):")
    for name, label in spacy_entities:
        print(f"  [{label}]  →  '{name}'")

    print("\nspaCy NER results (full passage):")
    all_spacy = []
    for i, sent in enumerate(sentences, 1):
        doc           = nlp(sent)
        sent_ents     = [(ent.text, ent.label_) for ent in doc.ents]
        all_spacy    += sent_ents
        status        = ", ".join(f"{n} [{l}]" for n, l in sent_ents) or "no entities"
        print(f"  Sentence {i}: {status}")

    print("\nComparison summary:")
    print(f"  {'Entity':<25} {'NLTK ne_chunk':<20} {'spaCy'}")
    print("  " + "-" * 60)

    # Build a lookup of what NLTK found
    nltk_lookup = {n.lower(): l for n, l in
                   [((" ".join(w for w, _ in c)), c.label())
                    for sent in sentences
                    for c in ne_chunk(pos_tag(word_tokenize(sent)))
                    if hasattr(c, "label")]}
    spacy_lookup = {n.lower(): l for n, l in all_spacy}

    all_names = sorted(set(list(nltk_lookup.keys()) + list(spacy_lookup.keys())))
    for name in all_names:
        nltk_label  = nltk_lookup.get(name, "—")
        spacy_label = spacy_lookup.get(name, "—")
        print(f"  {name.title():<25} {nltk_label:<20} {spacy_label}")

except ImportError:
    print("\n  spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")

