# Week 4: Syntactic and Semantic Analysis


## Task: Dependency Parsing and Semantic Similarity

## Running the Code

```bash
pip install spacy
python -m spacy download en_core_web_sm
python3 week-4/code/week4_syntax_semantics.py
```

> **Note on the similarity warning:** `en_core_web_sm` prints a `[W007]` warning when computing similarity, because the small model uses context tensors instead of full word vectors. The scores are still meaningful for comparison purposes. To suppress the warning, use `en_core_web_md` (medium model with full vectors) by running `python -m spacy download en_core_web_md`.
