# =============================================================================
# Week 6 – Word Embeddings and Distributed Representations
# =============================================================================
#
# This single script consolidates every Week 6 practical task:
#
#   Practical Task 1  → Train a Word2Vec model, display vectors, find similar words
#   Practical Task 2  → Similarity analysis on 10 words (cosine similarity)
#   Practical Task 3  → One-Hot Encoding vs Word Embeddings comparison report
#   Advanced Task     → Train on a larger dataset + visualise word clusters (PCA)
#
# It also adds ONE NEW CONCEPT not covered in the class notes:
#
#   NEW → Embedding-powered SEMANTIC SEARCH ENGINE
#         Represent whole documents as the average of their word vectors,
#         then rank them against a query by cosine similarity. This is the
#         retrieval idea behind search engines and the AI Bible Study
#         Companion project in this repository.
#
# =============================================================================

import numpy as np
import nltk
from gensim.models import Word2Vec

nltk.download("brown",  quiet=True)
nltk.download("punkt",  quiet=True)
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

# Matplotlib is only needed for the visualisation step. If it is missing the
# script still runs everything else and just skips the plot.
try:
    import matplotlib
    matplotlib.use("Agg")          # render to a file, no display window needed
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# -----------------------------------------------------------------------------
# Small helper: cosine similarity between two vectors
#     cos(θ) = (A · B) / (|A| |B|)        →  1 = identical, 0 = unrelated
# -----------------------------------------------------------------------------
def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


# =============================================================================
# PRACTICAL TASK 1 – TRAIN A WORD2VEC MODEL
# =============================================================================
# We start exactly like the class demo: a tiny hand-written corpus so the
# Word2Vec API and its output are easy to see.
# =============================================================================

print("=" * 65)
print("PRACTICAL TASK 1 – TRAIN A WORD2VEC MODEL")
print("=" * 65)

mini_corpus = [
    ["i", "love", "nlp"],
    ["nlp", "is", "interesting"],
    ["machine", "learning", "is", "powerful"],
    ["i", "love", "machine", "learning"],
    ["deep", "learning", "powers", "modern", "nlp"],
]

mini_model = Word2Vec(
    mini_corpus,
    vector_size=50,    # each word becomes a 50-dimensional dense vector
    window=2,          # context window: 2 words on each side
    min_count=1,       # keep every word, even ones seen only once
    sg=0,              # sg=0 → CBOW architecture (sg=1 would be Skip-Gram)
)

print(f"\n  Vocabulary size : {len(mini_model.wv.index_to_key)} words")
print(f"  Vector size     : {mini_model.wv.vector_size} dimensions")
print(f"\n  Vector for 'nlp' (first 8 of 50 values):")
print(f"    {np.round(mini_model.wv['nlp'][:8], 3)} ...")
print(f"\n  Words most similar to 'nlp':")
for word, score in mini_model.wv.most_similar("nlp", topn=3):
    print(f"    {word:<12} → {score:.4f}")


# =============================================================================
# LARGER MODEL – TRAIN ON THE BROWN CORPUS  (Advanced Task, part 1)
# =============================================================================
# A 5-sentence corpus is too small to learn real meaning. Here we train on the
# Brown corpus (~1 million words of real English) so that the similarity,
# analogy, and search sections below produce genuinely meaningful results.
# =============================================================================

print("\n" + "=" * 65)
print("LARGER MODEL – TRAINING WORD2VEC ON THE BROWN CORPUS")
print("=" * 65)
print("  Loading and lower-casing the Brown corpus...")

brown_sents = [[w.lower() for w in sent if w.isalpha()] for sent in brown.sents()]

print(f"  Sentences       : {len(brown_sents):,}")
print(f"  Training Word2Vec (Skip-Gram, 100 dims)... this takes a moment.")

model = Word2Vec(
    brown_sents,
    vector_size=100,
    window=5,
    min_count=5,       # ignore very rare words
    sg=1,              # Skip-Gram → better quality embeddings for rarer words
    workers=4,
    epochs=5,
)
wv = model.wv
print(f"  Done. Vocabulary: {len(wv.index_to_key):,} words")


# =============================================================================
# PRACTICAL TASK 2 – SIMILARITY ANALYSIS ON 10 WORDS
# =============================================================================
# For each of 10 words we print the closest neighbour the model learned, plus
# the similarity score. Words must exist in the vocabulary to be analysed.
# =============================================================================

print("\n" + "=" * 65)
print("PRACTICAL TASK 2 – SIMILARITY ANALYSIS (10 WORDS)")
print("=" * 65)

target_words = ["man", "woman", "school", "money", "city",
                "water", "music", "war", "science", "government"]

print(f"\n  {'Word':<12} {'Nearest neighbour':<20} {'Cosine score'}")
print("  " + "-" * 48)
for word in target_words:
    if word in wv:
        neighbour, score = wv.most_similar(word, topn=1)[0]
        print(f"  {word:<12} {neighbour:<20} {score:.4f}")
    else:
        print(f"  {word:<12} {'(not in vocabulary)':<20} —")

# A direct pairwise score, mirroring model.wv.similarity() from the notes
pair_a, pair_b = "man", "woman"
if pair_a in wv and pair_b in wv:
    print(f"\n  Direct similarity('{pair_a}', '{pair_b}') = "
          f"{wv.similarity(pair_a, pair_b):.4f}")


# =============================================================================
# WORD ANALOGY – VECTOR ARITHMETIC
# =============================================================================
# The famous demonstration that embeddings capture relationships:
#     king - man + woman  ≈  queen
# =============================================================================

print("\n" + "=" * 65)
print("WORD ANALOGY – king - man + woman ≈ ?")
print("=" * 65)

try:
    result = wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
    print("  king - man + woman is closest to:")
    for word, score in result:
        print(f"    {word:<12} → {score:.4f}")
except KeyError as e:
    print(f"  Could not run analogy — word missing from vocabulary: {e}")


# =============================================================================
# PRACTICAL TASK 3 – ONE-HOT ENCODING vs WORD EMBEDDINGS
# =============================================================================
# We build a One-Hot vector for a 3-word vocabulary, then contrast it with the
# dense embedding the model learned, and print the comparison report table.
# =============================================================================

print("\n" + "=" * 65)
print("PRACTICAL TASK 3 – ONE-HOT ENCODING vs WORD EMBEDDINGS")
print("=" * 65)

vocab = ["cat", "dog", "animal"]
print("\n  One-Hot Encoding (each word is an isolated 1):")
for i, word in enumerate(vocab):
    onehot = [1 if j == i else 0 for j in range(len(vocab))]
    print(f"    {word:<8} → {onehot}")

print("\n  Word Embedding (dense, learned from context — first 6 dims):")
for word in vocab:
    if word in wv:
        print(f"    {word:<8} → {np.round(wv[word][:6], 3)} ...")
    else:
        print(f"    {word:<8} → (not in vocabulary)")

print("\n  COMPARISON REPORT")
print("  " + "-" * 60)
print(f"  {'Feature':<26}{'One-Hot Encoding':<18}{'Word Embeddings'}")
print("  " + "-" * 60)
rows = [
    ("Size",                  "Vocab length",     "Fixed & compact"),
    ("Meaning representation", "None",             "Captures meaning"),
    ("Efficiency",            "Sparse / wasteful", "Dense / efficient"),
    ("Semantic understanding", "No",               "Yes"),
]
for feature, oh, emb in rows:
    print(f"  {feature:<26}{oh:<18}{emb}")
print("  " + "-" * 60)
print("  Conclusion: One-Hot vectors grow with the vocabulary and treat every")
print("  word as unrelated. Embeddings are compact, fixed-size, and place words")
print("  with similar meaning close together in vector space.")


# =============================================================================
# ADVANCED TASK – VISUALISE WORD CLUSTERS WITH PCA
# =============================================================================
# 100-dimensional vectors can't be seen directly, so we project a handful of
# words down to 2D using PCA (computed here with NumPy's SVD — no extra
# library needed) and plot them. Semantically related words should cluster.
# =============================================================================

print("\n" + "=" * 65)
print("ADVANCED TASK – PCA VISUALISATION OF WORD CLUSTERS")
print("=" * 65)

plot_words = ["man", "woman", "king", "queen", "boy", "girl",
              "school", "student", "teacher", "book",
              "water", "river", "sea", "rain",
              "car", "road", "engine", "train"]
plot_words = [w for w in plot_words if w in wv]

vectors = np.array([wv[w] for w in plot_words])

# PCA via SVD: centre the data, then take the first 2 principal components.
centred = vectors - vectors.mean(axis=0)
_, _, Vt = np.linalg.svd(centred, full_matrices=False)
coords = centred @ Vt[:2].T        # project onto the top 2 components

if HAVE_MPL:
    plt.figure(figsize=(11, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=60, color="#2563eb")
    for word, (x, y) in zip(plot_words, coords):
        plt.annotate(word, (x, y), fontsize=11,
                     xytext=(5, 4), textcoords="offset points")
    plt.title("Word Embeddings projected to 2D with PCA (Brown corpus)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    out_path = "week-6/screenshots/pca_word_clusters.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"  Saved plot → {out_path}")
    print("  Open it and screenshot for the logbook (Fig 1).")
else:
    print("  matplotlib not installed — skipping the plot.")
    print("  Install it with:  pip install matplotlib")
print(f"  Plotted {len(plot_words)} words. Expect family/water/transport groups.")


# =============================================================================
# NEW CONCEPT (not in the notes) – EMBEDDING-POWERED SEMANTIC SEARCH
# =============================================================================
# One-Hot and keyword search only match exact words. With embeddings we can
# search by MEANING: we represent each document as the average of its word
# vectors (a simple "document embedding"), then rank documents by cosine
# similarity to the query — even when they share no words with it.
#
# This is the same retrieval idea behind search engines and the AI Bible
# Study Companion in this repository.
# =============================================================================

print("\n" + "=" * 65)
print("NEW CONCEPT – EMBEDDING-POWERED SEMANTIC SEARCH ENGINE")
print("=" * 65)

documents = [
    "the government raised taxes to fund public schools",
    "scientists study the ocean and its deep water currents",
    "the army prepared its soldiers for the coming war",
    "students read books and learn new ideas at school",
    "the city built new roads for cars and trains",
    "music and art bring joy to people everywhere",
]


def embed_text(text):
    """Average the word vectors of every known word → one document vector."""
    words = [w for w in word_tokenize(text.lower()) if w in wv]
    if not words:
        return np.zeros(wv.vector_size)
    return np.mean([wv[w] for w in words], axis=0)


# Pre-compute an embedding for every document once.
doc_vectors = [embed_text(doc) for doc in documents]


def search(query, top_k=3):
    """Return the top_k documents ranked by cosine similarity to the query."""
    q_vec = embed_text(query)
    scored = [(cosine(q_vec, d_vec), doc)
              for d_vec, doc in zip(doc_vectors, documents)]
    scored.sort(reverse=True)
    return scored[:top_k]


# Note: the query words ("education", "naval", "vehicles") do NOT appear in the
# documents — keyword search would return nothing. Semantic search still works.
demo_queries = ["education and learning", "naval and sea exploration", "vehicles and transport"]

for query in demo_queries:
    print(f"\n  Query: \"{query}\"")
    print("  " + "-" * 58)
    for score, doc in search(query):
        print(f"    {score:.3f}  |  {doc}")
