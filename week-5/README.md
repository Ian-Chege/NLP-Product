# Week 5: CAT 1 Preparation and Mini NLP Projects

## NLP Workflow Covered This Week

| Step | Technique | Week Introduced |
|---|---|---|
| Text input | Raw sentence | – |
| Tokenization | `word_tokenize()` | Week 1 |
| Stopword removal | NLTK stopwords list | Week 1 |
| POS tagging | `pos_tag()` | Week 2 |
| N-grams | `nltk.util.ngrams` | Week 2 |
| Sequence labeling | HMM concepts | Week 3 |
| Dependency parsing | spaCy `dep_` | Week 4 |
| Semantic similarity | spaCy `.similarity()` | Week 4 |


## Running the Code

```bash
pip install nltk spacy
python -m spacy download en_core_web_sm
python3 week-5/code/week5_pipeline_chatbot.py
```

The script runs the full pipeline and chatbot demo automatically, then drops into interactive mode so you can type your own messages. Type `bye` to exit.
