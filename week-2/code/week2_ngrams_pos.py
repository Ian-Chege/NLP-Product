# =============================================================================
# Week 2: N-gram Models and POS Tagging
# =============================================================================
#
# This script covers two key NLP concepts:
#
#   1. N-grams  – sequences of N consecutive words used to model language patterns
#                 Bigram = 2 words, Trigram = 3 words
#
#   2. POS Tagging – labelling each word with its grammatical role
#                    (Noun, Verb, Adjective, etc.)
#
# Sample text: selected verses from the Book of Jude (KJV), the corpus our
# main project (AI Bible Study Companion) is built on.
# =============================================================================

import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk import pos_tag
from collections import Counter, defaultdict

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

# -----------------------------------------------------------------------------
# Sample text: Jude 1:3-4 (KJV)
# -----------------------------------------------------------------------------
sample_text = (
    "Beloved, when I gave all diligence to write unto you of the common salvation, "
    "it was needful for me to write unto you, and exhort you that ye should earnestly "
    "contend for the faith which was once delivered unto the saints. "
    "For there are certain men crept in unawares, who were before of old ordained to "
    "this condemnation, ungodly men, turning the grace of our God into lasciviousness, "
    "and denying the only Lord God, and our Lord Jesus Christ."
)

# Tokenize and keep only real words (no punctuation)
tokens = [t for t in word_tokenize(sample_text) if t.isalpha()]

print("=" * 60)
print("SAMPLE TEXT")
print("=" * 60)
print(sample_text)


# =============================================================================
# STEP 1: BIGRAMS AND TRIGRAMS
# =============================================================================
# An N-gram is a sliding window of N consecutive words.
# Bigrams capture two-word patterns ("the faith", "unto you").
# Trigrams capture three-word patterns ("contend for the").
#
# These patterns help a model predict what word is likely to come next
# based on what it has already seen.
# =============================================================================

bigrams  = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

# Count how often each bigram and trigram appears
bigram_freq  = Counter(bigrams)
trigram_freq = Counter(trigrams)

print("\n" + "=" * 60)
print("Fig 1 – BIGRAM OUTPUT (top 10 most frequent)")
print("=" * 60)
for pair, count in bigram_freq.most_common(10):
    print(f"  {pair[0]:15} {pair[1]:15}  →  {count}x")

print("\nFig 1 – TRIGRAM OUTPUT (top 10 most frequent)")
print("-" * 60)
for trio, count in trigram_freq.most_common(10):
    print(f"  {trio[0]:12} {trio[1]:12} {trio[2]:12}  →  {count}x")


# =============================================================================
# STEP 2: POS TAGGING
# =============================================================================
# Part-of-Speech tagging assigns each word a grammatical label.
# NLTK uses the Penn Treebank tag set; we translate them to plain English.
#
# Common tags:
#   NN  = Noun       VB  = Verb       JJ  = Adjective
#   NNP = Proper Noun  RB  = Adverb    IN  = Preposition
# =============================================================================

READABLE = {
    "NN": "Noun",      "NNS": "Noun",     "NNP": "Proper Noun", "NNPS": "Proper Noun",
    "VB": "Verb",      "VBD": "Verb",     "VBG": "Verb",        "VBN": "Verb",
    "VBP": "Verb",     "VBZ": "Verb",
    "JJ": "Adjective", "JJR": "Adjective","JJS": "Adjective",
    "RB": "Adverb",    "RBR": "Adverb",   "RBS": "Adverb",
    "IN": "Preposition","DT": "Determiner","CC": "Conjunction",
    "PRP": "Pronoun",  "PRP$": "Pronoun",
}

tagged = pos_tag(tokens)

print("\n" + "=" * 60)
print("Fig 2 – POS TAGGING EXAMPLE (first 20 words)")
print("=" * 60)
print(f"{'Word':<18} {'Tag':<8} {'Label'}")
print("-" * 40)
for word, tag in tagged[:20]:
    label = READABLE.get(tag, "Other")
    print(f"  {word:<16} {tag:<8} {label}")

print("\n" + "=" * 60)
print("Fig 3 – PYTHON CODE USED FOR POS TAGGING")
print("=" * 60)
print("  tagged = pos_tag(tokens)")
print("  # tokens is the list of words from word_tokenize()")
print("  # pos_tag returns a list of (word, tag) tuples")
print("  # e.g. [('Beloved', 'NNP'), ('gave', 'VBD'), ...]")


# =============================================================================
# STEP 3: LANGUAGE PREDICTION USING BIGRAMS
# =============================================================================
# Given a word, we can look at all bigrams that start with that word
# and predict which word is most likely to follow it.
# This is a simple version of how language models work.
# =============================================================================

# Build a dictionary: word → list of words that follow it
next_word_map = defaultdict(list)
for w1, w2 in bigrams:
    next_word_map[w1.lower()].append(w2.lower())

def predict_next(word):
    candidates = next_word_map.get(word.lower(), [])
    if not candidates:
        return f"(no prediction — '{word}' not seen in text)"
    most_common = Counter(candidates).most_common(3)
    return [(w, c) for w, c in most_common]

print("\n" + "=" * 60)
print("Fig 4 – LANGUAGE PREDICTION EXAMPLE")
print("=" * 60)
test_words = ["unto", "the", "faith", "God"]
for word in test_words:
    predictions = predict_next(word)
    print(f"\n  After '{word}', most likely next words:")
    for next_w, count in predictions:
        print(f"    → '{next_w}'  ({count}x in text)")


# =============================================================================
# STEP 4: FULL SENTENCE ANALYSIS
# =============================================================================
# Combining tokenization, POS tagging, and N-grams to give a complete
# structural summary of the text.
# =============================================================================

pos_counts = Counter(READABLE.get(tag, "Other") for _, tag in tagged)

print("\n" + "=" * 60)
print("Fig 5 – SENTENCE ANALYSIS OUTPUT")
print("=" * 60)
print(f"  Total words (tokens)  : {len(tokens)}")
print(f"  Unique words          : {len(set(t.lower() for t in tokens))}")
print(f"  Total bigrams         : {len(bigrams)}")
print(f"  Total trigrams        : {len(trigrams)}")
print(f"\n  Grammatical breakdown:")
for label, count in pos_counts.most_common():
    print(f"    {label:<15} : {count}")


# =============================================================================
# TASK 3 – POS TAGGING ON DIFFERENT TEXT TYPES
# =============================================================================
# POS tagging behaves differently depending on the type of text.
# Here we compare three domains: news, social media, and student project titles.
# =============================================================================

text_samples = {
    "News Article": (
        "The government announced a new policy on renewable energy funding today."
    ),
    "Social Media": (
        "omg just saw the new AI tool everyone's talking about, it's literally insane lol"
    ),
    "Student Project Title": (
        "AI-Powered Bible Study Companion: Scripture Analysis Using NLP and Deep Learning"
    ),
}

print("\n" + "=" * 60)
print("TASK 3 – POS TAGGING ON DIFFERENT TEXT TYPES")
print("=" * 60)

for domain, text in text_samples.items():
    tokens_d = word_tokenize(text)
    tagged_d = pos_tag(tokens_d)

    print(f"\n  [{domain}]")
    print(f"  Text: \"{text}\"")
    print(f"  {'Word':<22} {'Tag':<8} {'Label'}")
    print("  " + "-" * 42)
    for word, tag in tagged_d:
        label = READABLE.get(tag, "Other")
        print(f"    {word:<20} {tag:<8} {label}")

