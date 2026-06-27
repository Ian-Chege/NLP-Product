from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import nltk
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# Download NLTK corpora at startup
for resource in (
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "stopwords",
):
    nltk.download(resource, quiet=True)

from jude_corpus import JUDE_CORPUS  # noqa: E402 — must follow nltk.download
from nlp_pipeline import JudePipeline  # noqa: E402
import neural_lm  # noqa: E402
import sentiment as sentiment_module  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=neural_lm._build, daemon=True).start()
    threading.Thread(target=sentiment_module._get_pipe, daemon=True).start()
    yield


app = FastAPI(title="Jude Companion API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = JudePipeline()


# ── Verse routes ──────────────────────────────────────────────────────────────

@app.get("/verses")
def get_verses():
    return JUDE_CORPUS


@app.get("/verse/{n}")
def get_verse(n: int):
    verse = next((v for v in JUDE_CORPUS if v["verse_number"] == n), None)
    if not verse:
        raise HTTPException(status_code=404, detail="Verse not found")
    return verse


# ── NLP routes ────────────────────────────────────────────────────────────────

@app.get("/analyze/{n}")
def analyze_verse(n: int):
    result = pipeline.analyze_verse(n)
    if not result:
        raise HTTPException(status_code=404, detail="Verse not found")
    return result


@app.get("/frequency")
def get_frequency(verse: Optional[int] = None):
    return pipeline.get_frequency(verse_number=verse)


# ── Week 2: N-grams ──────────────────────────────────────────────────────────

@app.get("/ngrams")
def get_ngrams(verse: Optional[int] = None, top_n: int = 10):
    """
    Returns the most frequent bigrams and trigrams.
    - No params: patterns across the whole book (reveals Jude's recurring themes)
    - ?verse=N: patterns within a single verse
    """
    return pipeline.get_ngrams(verse_number=verse, top_n=top_n)


# ── Week 4: Dependency parsing ────────────────────────────────────────────────

@app.get("/parse/{n}")
def parse_verse(n: int):
    """
    Returns the grammatical structure of a verse:
    subject, main verb, object, and a full token-level dependency tree.
    """
    result = pipeline.parse_verse(n)
    if not result:
        raise HTTPException(status_code=404, detail="Verse not found")
    return result


# ── Week 4: Semantic similarity ───────────────────────────────────────────────

@app.get("/related/{n}")
def find_related(n: int, top_n: int = 3):
    """
    Finds the most semantically similar verses to verse N.
    This is the automatic cross-reference finder — no manual concordance needed.
    """
    result = pipeline.find_related(verse_number=n, top_n=top_n)
    if result is None:
        raise HTTPException(status_code=404, detail="Verse not found")
    return result


# ── Week 5: Book-wide themes ──────────────────────────────────────────────────

@app.get("/themes")
def get_themes(top_n: int = 8):
    """
    Returns the most recurring phrases across the entire book of Jude.
    These are the dominant themes Jude returns to again and again.
    """
    return pipeline.get_themes(top_n=top_n)


# ── Week 6: Embedding-powered semantic search ─────────────────────────────────

@app.get("/search")
def semantic_search(q: str, top_n: int = 5):
    """
    Search the book of Jude by MEANING, not keywords.

    Each verse is represented as a document embedding (the average of its word
    vectors); the query is embedded the same way and verses are ranked by
    cosine similarity. Finds relevant verses even when they share no words
    with the query.
    """
    return pipeline.semantic_search(query=q, top_n=top_n)


# ── Week 7: Neural Language Model ────────────────────────────────────────────

@app.get("/model-status")
def model_status():
    """Check whether the neural language model has finished training."""
    return {"ready": neural_lm.is_ready()}


@app.get("/predict")
def predict_next_word(text: str, top_n: int = 5):
    """
    Predict the most likely next word(s) given a sentence fragment.
    Returns words ranked by confidence (softmax probability).
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    return {"input": text, "predictions": neural_lm.predict_next(text, top_n=top_n)}


class TokenizeRequest(BaseModel):
    text: str


@app.post("/tokenize-text")
def tokenize_text(request: TokenizeRequest):
    """
    Tokenize arbitrary text using the model's fitted Tokenizer.
    Returns word → index mapping and the integer sequence.
    Demonstrates Keras Tokenizer (Week 7 Class Demonstrations 3 & 4).
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    return neural_lm.tokenize_text(request.text)


# ── Week 8: Transformer sentiment analysis ────────────────────────────────────

@app.get("/sentiment-status")
def sentiment_status():
    return {"ready": sentiment_module.is_ready()}


@app.get("/sentiment/{n}")
def get_verse_sentiment(n: int):
    """Sentiment analysis for a single verse using DistilBERT."""
    verse = next((v for v in JUDE_CORPUS if v["verse_number"] == n), None)
    if not verse:
        raise HTTPException(status_code=404, detail="Verse not found")
    result = sentiment_module.analyze_verse(verse["text"])
    return {"verse_number": n, "text": verse["text"], **result}


@app.get("/sentiment")
def get_book_sentiment():
    """Sentiment analysis across all 25 verses — returns per-verse results + summary."""
    return sentiment_module.analyze_book(JUDE_CORPUS)


# ── AI study route ────────────────────────────────────────────────────────────

class StudyRequest(BaseModel):
    verse_number: int


@app.post("/study")
async def get_study(request: StudyRequest):
    verse = next(
        (v for v in JUDE_CORPUS if v["verse_number"] == request.verse_number), None
    )
    if not verse:
        raise HTTPException(status_code=404, detail="Verse not found")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    client = openai.OpenAI(api_key=api_key)

    prompt = f"""You are a biblical scholar and study guide author for a university NLP course.
Analyze this verse from the book of Jude (KJV):

Jude 1:{verse['verse_number']} — "{verse['text']}"

Return ONLY valid JSON — no markdown fences, no extra text — matching this exact schema:
{{
  "study_questions": [
    "question 1",
    "question 2",
    "question 3",
    "question 4"
  ],
  "cross_references": [
    {{"reference": "Book Chapter:Verse", "explanation": "one sentence explanation"}},
    {{"reference": "Book Chapter:Verse", "explanation": "one sentence explanation"}},
    {{"reference": "Book Chapter:Verse", "explanation": "one sentence explanation"}}
  ],
  "devotional": "A 2–3 sentence devotional reflection that connects this verse to daily life."
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )

    return json.loads(response.choices[0].message.content)
