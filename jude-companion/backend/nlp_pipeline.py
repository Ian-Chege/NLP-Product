from __future__ import annotations

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.util import ngrams
from collections import Counter
from typing import Optional

from jude_corpus import JUDE_CORPUS

# spaCy is loaded once and reused — loading it per-request is too slow
_spacy_nlp = None

def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp

BIBLICAL_STOPWORDS = {
    "thee", "thou", "thy", "thine", "hath", "doth", "ye", "unto",
    "wherefore", "also", "yet", "even", "hast", "art", "wilt", "shalt"
}

PENN_TO_READABLE = {
    "NN": "Noun",   "NNS": "Noun",   "NNP": "Noun",  "NNPS": "Noun",
    "VB": "Verb",   "VBD": "Verb",   "VBG": "Verb",
    "VBN": "Verb",  "VBP": "Verb",   "VBZ": "Verb",
    "JJ": "Adjective", "JJR": "Adjective", "JJS": "Adjective",
    "RB": "Adverb", "RBR": "Adverb", "RBS": "Adverb",
}


class JudePipeline:
    def __init__(self):
        self.stop_words = set(stopwords.words("english")) | BIBLICAL_STOPWORDS
        self.verses = JUDE_CORPUS

    def tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [
            t for t in tokens
            if t.isalpha() and t.lower() not in self.stop_words
        ]

    def get_pos_tags(self, tokens: list[str]) -> list[tuple]:
        word_tokens = [t for t in tokens if t.isalpha()]
        tagged = pos_tag(word_tokens)
        return [
            (word, tag, PENN_TO_READABLE.get(tag, "Other"))
            for word, tag in tagged
        ]

    def get_keywords(self, text: str, top_n: int = 6) -> list[tuple]:
        tokens = self.tokenize(text)
        filtered = self.remove_stopwords(tokens)
        counter = Counter(t.lower() for t in filtered)
        return counter.most_common(top_n)

    def get_frequency(self, verse_number: int | None = None) -> list[dict]:
        if verse_number is not None:
            verse = next(
                (v for v in self.verses if v["verse_number"] == verse_number), None
            )
            if not verse:
                return []
            tokens = self.remove_stopwords(self.tokenize(verse["text"]))
        else:
            tokens = []
            for v in self.verses:
                tokens.extend(self.remove_stopwords(self.tokenize(v["text"])))

        counter = Counter(t.lower() for t in tokens)
        return [
            {"word": word, "count": count}
            for word, count in counter.most_common(15)
        ]

    def analyze_verse(self, verse_number: int) -> dict | None:
        verse = next(
            (v for v in self.verses if v["verse_number"] == verse_number), None
        )
        if not verse:
            return None

        tokens = self.tokenize(verse["text"])
        filtered = self.remove_stopwords(tokens)
        pos_tags = self.get_pos_tags(tokens)
        keywords = self.get_keywords(verse["text"], top_n=6)

        return {
            "verse_number": verse_number,
            "text": verse["text"],
            "tokens": tokens,
            "filtered_tokens": filtered,
            "pos_tags": [
                {"word": w, "tag": t, "label": lbl}
                for w, t, lbl in pos_tags
            ],
            "keywords": [{"word": w, "count": c} for w, c in keywords],
            "token_count": len(tokens),
            "unique_count": len({t.lower() for t in tokens if t.isalpha()}),
        }

    # ── Week 2: N-grams ───────────────────────────────────────────────────────

    def get_ngrams(self, verse_number: int | None = None, top_n: int = 10) -> dict:
        """Return the most frequent bigrams and trigrams for a verse or the whole book."""
        if verse_number is not None:
            verse = next((v for v in self.verses if v["verse_number"] == verse_number), None)
            texts = [verse["text"]] if verse else []
        else:
            texts = [v["text"] for v in self.verses]

        all_tokens = []
        for text in texts:
            tokens = [t.lower() for t in self.tokenize(text) if t.isalpha()
                      and t.lower() not in self.stop_words]
            all_tokens.extend(tokens)

        bigram_counts  = Counter(ngrams(all_tokens, 2))
        trigram_counts = Counter(ngrams(all_tokens, 3))

        return {
            "scope": f"verse {verse_number}" if verse_number else "full book",
            "bigrams": [
                {"phrase": " ".join(p), "count": c}
                for p, c in bigram_counts.most_common(top_n)
            ],
            "trigrams": [
                {"phrase": " ".join(p), "count": c}
                for p, c in trigram_counts.most_common(top_n)
            ],
        }

    # ── Week 4: Dependency parsing ────────────────────────────────────────────

    def parse_verse(self, verse_number: int) -> dict | None:
        """Return the dependency parse of a verse — subject, verb, object, and full token tree."""
        verse = next((v for v in self.verses if v["verse_number"] == verse_number), None)
        if not verse:
            return None

        nlp = _get_spacy()
        doc = nlp(verse["text"])

        tokens = [
            {
                "word": token.text,
                "dep": token.dep_,
                "head": token.head.text,
                "pos": token.pos_,
            }
            for token in doc
        ]

        subject = next((t.text for t in doc if t.dep_ == "nsubj"), None)
        root    = next((t.text for t in doc if t.dep_ == "ROOT"),  None)
        obj     = next((t.text for t in doc if t.dep_ in ("dobj", "attr", "pobj")), None)

        return {
            "verse_number": verse_number,
            "text": verse["text"],
            "subject": subject,
            "verb": root,
            "object": obj,
            "tokens": tokens,
        }

    # ── Week 4: Semantic similarity ───────────────────────────────────────────

    def _verse_vector(self, text: str):
        """Return the spaCy doc object for a verse text."""
        return _get_spacy()(text)

    def find_related(self, verse_number: int, top_n: int = 3) -> list[dict]:
        """
        Find the most semantically similar verses to a given verse.
        This is the automatic cross-reference finder — surfaces connected passages
        without the student needing to search manually.
        """
        target_verse = next((v for v in self.verses if v["verse_number"] == verse_number), None)
        if not target_verse:
            return []

        target_doc = self._verse_vector(target_verse["text"])

        scores = []
        for verse in self.verses:
            if verse["verse_number"] == verse_number:
                continue
            other_doc = self._verse_vector(verse["text"])
            score = target_doc.similarity(other_doc)
            scores.append({
                "verse_number": verse["verse_number"],
                "text": verse["text"],
                "similarity": round(score, 4),
            })

        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_n]

    # ── Book-wide themes ──────────────────────────────────────────────────────

    def get_themes(self, top_n: int = 8) -> dict:
        """
        Return the most recurring phrases (bigrams + trigrams) across the entire book.
        These represent the dominant themes Jude keeps returning to.
        """
        return self.get_ngrams(verse_number=None, top_n=top_n)
