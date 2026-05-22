# Jude Companion — AI Bible Study App

**BIT4133 Natural Language Processing**

An interactive web application for exploring the book of Jude (KJV, 25 verses) through NLTK-powered NLP analysis and GPT-4o-mini AI study generation.

---

## Project structure

```
jude-companion/
├── backend/
│   ├── main.py            FastAPI app + all routes
│   ├── jude_corpus.py     All 25 Jude verses (KJV)
│   ├── nlp_pipeline.py    JudePipeline NLP class
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── VerseExplorer.jsx    Sidebar verse list
    │   │   ├── PipelineView.jsx     4-step NLP breakdown
    │   │   ├── FrequencyChart.jsx   Word frequency bar chart
    │   │   └── StudyCompanion.jsx   AI-generated study content
    │   ├── App.jsx
    │   ├── index.css       Hallmark design tokens + Tailwind
    │   └── main.jsx
    ├── package.json
    └── vite.config.js      Includes /api → :8000 proxy
```

---

## Backend setup

```bash
cd jude-companion/backend

# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file
cp .env.example .env
# Edit .env and add your OpenAI API key:
#   OPENAI_API_KEY=sk-...

# 4. Run the server
uvicorn main:app --reload --port 8000
```

NLTK corpora (`punkt`, `punkt_tab`, `averaged_perceptron_tagger`, `averaged_perceptron_tagger_eng`, `stopwords`) are downloaded automatically on first startup.

The API will be available at `http://localhost:8000`. You can explore the interactive docs at `http://localhost:8000/docs`.

---

## Frontend setup

```bash
cd jude-companion/frontend

# 1. Install dependencies
npm install

# 2. Run the dev server
npm run dev
```

Open `http://localhost:5173` in your browser.

The Vite dev server proxies all `/api/*` requests to the FastAPI backend at `http://localhost:8000`.

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/verses` | All 25 verses `[{verse_number, text}]` |
| GET | `/verse/{n}` | Single verse |
| GET | `/analyze/{n}` | Full NLP analysis for verse n |
| GET | `/frequency?verse={n}` | Word frequency — verse or whole book |
| POST | `/study` | AI study content for a verse |

### POST `/study` request body
```json
{ "verse_number": 3 }
```

### POST `/study` response schema
```json
{
  "study_questions": ["string", "string", "string", "string"],
  "cross_references": [
    { "reference": "Book Chapter:Verse", "explanation": "string" }
  ],
  "devotional": "string"
}
```

---

## NLP Pipeline — `JudePipeline`

| Method | Description |
|--------|-------------|
| `tokenize(text)` | NLTK `word_tokenize` |
| `remove_stopwords(tokens)` | NLTK stopwords + 16 biblical stopwords |
| `get_pos_tags(tokens)` | Penn Treebank tags → readable labels |
| `get_keywords(text, top_n=6)` | Top N content words by frequency |
| `get_frequency(verse_number=None)` | Top 15 words; whole book if no verse |
| `analyze_verse(verse_number)` | Full analysis dict |

**Biblical stopwords added:** thee, thou, thy, thine, hath, doth, ye, unto, wherefore, also, yet, even, hast, art, wilt, shalt

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Used only by `POST /study` |

Place your `.env` file inside `backend/` (alongside `main.py`).
