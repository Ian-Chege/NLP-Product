# =============================================================================
# Week 4 – Syntactic and Semantic Analysis
# =============================================================================
#
# This script covers two connected NLP concepts:
#
#   1. Syntax / Dependency Parsing
#      spaCy reads a sentence and maps out how each word depends on another.
#      This gives us: subject, verb, object, and all the connecting roles.
#
#   2. Semantics / Semantic Similarity
#      spaCy can compare two sentences and return a score (0–1) showing how
#      similar they are in meaning, even if they use different words.
#
# Sample text: sentences from the Book of Jude and our AI Bible Study project,
# alongside everyday examples to show how the tools generalise.
# =============================================================================

import spacy

# Load the English model (run once to download: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    raise


# =============================================================================
# STEP 1 – DEPENDENCY PARSING (Practical Task 1)
# =============================================================================
# Dependency parsing finds the grammatical relationships between words.
#
# Every sentence has a ROOT (the main verb). All other words depend on
# something — they may be a subject (nsubj), object (dobj), modifier (amod),
# etc. spaCy labels each word with its dependency role and the word it points to.
#
# Tag glossary used below:
#   ROOT   = main verb of the sentence
#   nsubj  = nominal subject (who/what does the action)
#   dobj   = direct object (who/what receives the action)
#   attr   = attribute (describes the subject after a linking verb)
#   amod   = adjectival modifier (adjective modifying a noun)
#   compound = noun–noun compound ("language processing")
#   det    = determiner ("the", "a")
#   prep   = prepositional modifier
#   pobj   = object of a preposition
# =============================================================================

DEMO_SENTENCE = "The lecturer teaches Natural Language Processing."

print("=" * 60)
print("Fig 1 – DEPENDENCY PARSING EXAMPLE")
print("=" * 60)
print(f"Input: \"{DEMO_SENTENCE}\"\n")
print(f"  {'Word':<20} {'Dep Role':<14} {'Head Word'}")
print("  " + "-" * 48)

doc = nlp(DEMO_SENTENCE)
for token in doc:
    print(f"  {token.text:<20} {token.dep_:<14} {token.head.text}")

print("\n  Key relationships:")
for token in doc:
    if token.dep_ in ("nsubj", "ROOT", "dobj", "attr"):
        print(f"    {token.dep_:<8} → '{token.text}'  (head: '{token.head.text}')")


# =============================================================================
# STEP 2 – BIBLICAL TEXT DEPENDENCY PARSING
# =============================================================================
# We apply the same parser to a verse from Jude to see how it handles
# archaic language — the main text our project is built on.
# =============================================================================

JUDE_SENTENCE = "Jude, the servant of Jesus Christ, contend for the faith."

print("\n" + "=" * 60)
print("Fig 2 – DEPENDENCY PARSING ON BIBLICAL TEXT (Jude 1:3)")
print("=" * 60)
print(f"Input: \"{JUDE_SENTENCE}\"\n")
print(f"  {'Word':<20} {'Dep Role':<14} {'Head Word'}")
print("  " + "-" * 48)

doc_jude = nlp(JUDE_SENTENCE)
for token in doc_jude:
    print(f"  {token.text:<20} {token.dep_:<14} {token.head.text}")


# =============================================================================
# STEP 3 – ASSIGNMENT 1: DEPENDENCY ANALYSIS (two given sentences)
# =============================================================================

assignment_sentences = [
    "The administrator updated student records.",
    "Machine learning improves language processing.",
]

print("\n" + "=" * 60)
print("Fig 3 – ASSIGNMENT 1: DEPENDENCY ANALYSIS")
print("=" * 60)

for sentence in assignment_sentences:
    doc_a = nlp(sentence)
    print(f"\n  Sentence: \"{sentence}\"")
    print(f"  {'Word':<20} {'Dep Role':<14} {'Head Word'}")
    print("  " + "-" * 48)
    for token in doc_a:
        print(f"  {token.text:<20} {token.dep_:<14} {token.head.text}")

    # Highlight the main grammatical roles
    subject = next((t.text for t in doc_a if t.dep_ == "nsubj"), "—")
    root    = next((t.text for t in doc_a if t.dep_ == "ROOT"), "—")
    obj     = next((t.text for t in doc_a if t.dep_ in ("dobj", "attr")), "—")
    print(f"\n  Summary → Subject: '{subject}'  |  Verb: '{root}'  |  Object: '{obj}'")


# =============================================================================
# STEP 4 – SEMANTIC SIMILARITY (Practical Task 2)
# =============================================================================
# spaCy computes semantic similarity using word vectors.
# The score ranges from 0.0 (completely different) to 1.0 (identical).
#
# en_core_web_sm uses averaged word vectors, so sentences with synonyms or
# related vocabulary will score higher than sentences on unrelated topics.
# =============================================================================

print("\n" + "=" * 60)
print("Fig 4 – SEMANTIC SIMILARITY ANALYSIS")
print("=" * 60)

pairs = [
    ("The student passed the examination.",
     "The learner succeeded in the exam."),
]

for s1, s2 in pairs:
    doc1  = nlp(s1)
    doc2  = nlp(s2)
    score = doc1.similarity(doc2)
    print(f"\n  Sentence A: \"{s1}\"")
    print(f"  Sentence B: \"{s2}\"")
    print(f"  Similarity Score: {score:.4f}  ({'high – similar meaning' if score > 0.75 else 'moderate – some overlap' if score > 0.5 else 'low – different meaning'})")


# =============================================================================
# STEP 5 – ASSIGNMENT 2: SIMILAR VS UNRELATED SENTENCES
# =============================================================================

print("\n" + "=" * 60)
print("Fig 5 – ASSIGNMENT 2: SIMILAR VS UNRELATED SENTENCE PAIRS")
print("=" * 60)

comparison_pairs = [
    # Similar pair
    ("Jude urges believers to contend for the faith.",
     "Jude encourages Christians to defend their belief."),
    # Unrelated pair
    ("The market price of wheat fell sharply.",
     "The disciples gathered in the upper room."),
]

labels = ["Similar pair", "Unrelated pair"]

for label, (s1, s2) in zip(labels, comparison_pairs):
    doc1  = nlp(s1)
    doc2  = nlp(s2)
    score = doc1.similarity(doc2)
    print(f"\n  [{label}]")
    print(f"  Sentence A: \"{s1}\"")
    print(f"  Sentence B: \"{s2}\"")
    print(f"  Similarity Score: {score:.4f}")

print("\n  Observation:")
print("  The similar pair scores higher because both sentences share")
print("  related vocabulary (faith/belief, urge/encourage, Jude/Jude).")
print("  The unrelated pair scores lower because wheat markets and")
print("  biblical gatherings share no common vocabulary or context.")
