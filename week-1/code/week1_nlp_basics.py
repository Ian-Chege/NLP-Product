# =============================================================================
# Week 1 – Introduction to NLP: Tokenization & Stopwords Removal
# =============================================================================
#
# This script demonstrates two fundamental NLP preprocessing steps:
#   1. Tokenization  – splitting a sentence into individual words (tokens)
#   2. Stopword removal – filtering out common words that carry little meaning
#
# We use a verse from the Book of Jude as our sample text, since our main
# project (AI Bible Study Companion) is built around biblical scripture.
# =============================================================================

import nltk

# Download the NLTK data files we need (only runs once; skips if already done)
nltk.download("punkt")           # tokenizer models
nltk.download("punkt_tab")       # updated tokenizer tables
nltk.download("stopwords")       # list of common English stopwords

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# -----------------------------------------------------------------------------
# Sample text – Jude 1:3 (KJV)
# -----------------------------------------------------------------------------
sample_text = (
    "Beloved, when I gave all diligence to write unto you of the common salvation, "
    "it was needful for me to write unto you, and exhort you that ye should earnestly "
    "contend for the faith which was once delivered unto the saints."
)

print("=" * 60)
print("ORIGINAL TEXT")
print("=" * 60)
print(sample_text)


# =============================================================================
# STEP 1 – TOKENIZATION
# =============================================================================
# word_tokenize() splits the sentence into a list of individual tokens.
# Punctuation marks are treated as separate tokens.
# =============================================================================

tokens = word_tokenize(sample_text)

print("\n" + "=" * 60)
print("Fig 4 – TOKENIZATION OUTPUT")
print("=" * 60)
print(f"Total tokens found: {len(tokens)}\n")
print(tokens)


# =============================================================================
# STEP 2 – STOPWORDS REMOVAL
# =============================================================================
# Stopwords are very common words (e.g. "the", "is", "and") that appear
# frequently but don't add much meaning to our analysis.
# Removing them leaves us with the words that actually matter.
#
# We also include a few KJV-specific words (thee, thou, ye, unto) that
# behave like stopwords in biblical text.
# =============================================================================

english_stopwords = set(stopwords.words("english"))

# Extra stopwords specific to old King James English
kjv_stopwords = {"thee", "thou", "thy", "thine", "hath", "ye", "unto", "wherefore"}

all_stopwords = english_stopwords | kjv_stopwords

# Keep a token only if:
#   - it is a real word (isalpha skips punctuation and numbers)
#   - it is NOT in our stopwords list
filtered_tokens = [
    token for token in tokens
    if token.isalpha() and token.lower() not in all_stopwords
]

print("\n" + "=" * 60)
print("Fig 5 – STOPWORDS REMOVAL OUTPUT")
print("=" * 60)
print(f"Tokens before removal : {len([t for t in tokens if t.isalpha()])}")
print(f"Tokens after removal  : {len(filtered_tokens)}")
print(f"Stopwords removed     : {len([t for t in tokens if t.isalpha()]) - len(filtered_tokens)}\n")
print(filtered_tokens)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("These filtered tokens are the meaningful words in the verse.")
print("They are what our NLP pipeline uses for keyword extraction,")
print("frequency analysis, and AI-powered study generation.")
print("=" * 60)
