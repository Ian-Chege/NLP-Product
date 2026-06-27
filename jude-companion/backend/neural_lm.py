from __future__ import annotations

import numpy as np

_tokenizer = None
_model = None
_index_word: dict[int, str] = {}
_context_length = 3


def _build() -> None:
    """Train the neural language model on the Jude corpus (lazy, called once)."""
    global _tokenizer, _model, _index_word

    if _model is not None:
        return

    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Dense, Flatten, Dropout
    from tensorflow.keras.utils import to_categorical

    from jude_corpus import JUDE_CORPUS

    corpus = [v["text"] for v in JUDE_CORPUS]

    tok = Tokenizer(oov_token="<OOV>")
    tok.fit_on_texts(corpus)
    _tokenizer = tok
    _index_word = {v: k for k, v in tok.word_index.items()}

    vocab_size = len(tok.word_index) + 1

    X_list, y_list = [], []
    for text in corpus:
        seq = tok.texts_to_sequences([text])[0]
        for i in range(_context_length, len(seq)):
            X_list.append(seq[i - _context_length : i])
            y_list.append(seq[i])

    if not X_list:
        return

    X = np.array(X_list)
    Y = to_categorical(y_list, num_classes=vocab_size)

    print(f"[NLM] Vocabulary size : {vocab_size} tokens")
    print(f"[NLM] Training samples: {len(X_list)}")
    print(f"[NLM] Context window  : {_context_length} words → predict next word")
    print("[NLM] Training neural language model on the Book of Jude…")

    model = Sequential(
        [
            Embedding(vocab_size, 32),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(vocab_size, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, Y, epochs=80, verbose=0)
    _model = model

    print("[NLM] Training complete. Model ready for next-word prediction.")


def predict_next(text: str, top_n: int = 5) -> list[dict]:
    """Return the top-N predicted next words with confidence scores."""
    _build()
    assert _tokenizer and _model

    seq = _tokenizer.texts_to_sequences([text.lower()])[0]
    if not seq:
        return []

    # Take the last `_context_length` tokens; left-pad with 0 if needed
    context = seq[-_context_length:]
    context = [0] * (_context_length - len(context)) + context
    probs = _model.predict(np.array([context]), verbose=0)[0]

    # Collect top candidates, skipping the padding token (0) and OOV (1)
    results = []
    for idx in np.argsort(probs)[::-1]:
        if idx <= 1:
            continue
        word = _index_word.get(int(idx))
        if not word:
            continue
        results.append({"word": word, "confidence": round(float(probs[idx]), 4)})
        if len(results) >= top_n:
            break

    return results


def tokenize_text(text: str) -> dict:
    """
    Tokenize arbitrary text and return the word-index mapping and sequence.
    Demonstrates Class Demonstrations 3 & 4 from Week 7.
    """
    _build()
    assert _tokenizer

    sequences = _tokenizer.texts_to_sequences([text])
    words = text.lower().split()
    word_index = {w: _tokenizer.word_index.get(w, _tokenizer.word_index.get("<OOV>")) for w in words}

    return {
        "text": text,
        "word_index": word_index,
        "sequences": sequences,
        "vocabulary_size": len(_tokenizer.word_index),
    }


def is_ready() -> bool:
    return _model is not None
