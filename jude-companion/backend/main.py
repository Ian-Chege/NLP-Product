from __future__ import annotations

import json
import os
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

app = FastAPI(title="Jude Companion API", version="1.0.0")

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
