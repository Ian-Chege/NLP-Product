from __future__ import annotations

import threading

_pipe = None
_lock = threading.Lock()


def _get_pipe():
    global _pipe
    if _pipe is None:
        with _lock:
            if _pipe is None:
                print("[Sentiment] Loading DistilBERT sentiment pipeline…")
                from transformers import pipeline
                _pipe = pipeline("sentiment-analysis", framework="pt")
                print("[Sentiment] Pipeline ready.")
    return _pipe


def analyze_verse(text: str) -> dict:
    result = _get_pipe()(text[:512])[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 4),
    }


def analyze_book(verses: list[dict]) -> dict:
    pipe = _get_pipe()
    texts = [v["text"][:512] for v in verses]
    raw = pipe(texts)

    results = [
        {
            "verse_number": v["verse_number"],
            "text": v["text"],
            "label": r["label"],
            "score": round(r["score"], 4),
        }
        for v, r in zip(verses, raw)
    ]

    positive = [r for r in results if r["label"] == "POSITIVE"]
    negative = [r for r in results if r["label"] == "NEGATIVE"]

    return {
        "verses": results,
        "summary": {
            "total": len(results),
            "positive": len(positive),
            "negative": len(negative),
            "most_positive": max(results, key=lambda x: x["score"] if x["label"] == "POSITIVE" else 0),
            "most_negative": max(results, key=lambda x: x["score"] if x["label"] == "NEGATIVE" else 0),
        },
    }


def is_ready() -> bool:
    return _pipe is not None
