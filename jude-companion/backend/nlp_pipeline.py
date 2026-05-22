from __future__ import annotations

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from collections import Counter
from typing import Optional

from jude_corpus import JUDE_CORPUS

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
