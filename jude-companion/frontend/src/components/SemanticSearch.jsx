import { useState } from "react"
import { Search, Loader2 } from "lucide-react"

// Example queries that deliberately share few/no words with the verses they
// match — they demonstrate that the search works by meaning, not keywords.
const SUGGESTIONS = [
  "false teachers sneaking in",
  "God keep me from falling",
  "angels who left their home",
  "glory and majesty to God forever",
]

export default function SemanticSearch({ onSelectVerse }) {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState([])
  const [searched, setSearched] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function runSearch(q) {
    const term = (q ?? query).trim()
    if (!term) return
    setQuery(term)
    setLoading(true)
    setError(null)
    try {
      const r = await fetch(`/api/search?q=${encodeURIComponent(term)}&top_n=6`)
      if (!r.ok) throw new Error("Search failed")
      const data = await r.json()
      setResults(data.results)
      setSearched(true)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="section-card">
        <div className="section-card-title">
          <Search size={15} /> Semantic Search — find verses by meaning
        </div>

        <form
          className="chat-input-bar"
          style={{ paddingTop: 0, borderTop: "none" }}
          onSubmit={(e) => {
            e.preventDefault()
            runSearch()
          }}
        >
          <input
            className="chat-input"
            placeholder="Describe an idea, e.g. 'false teachers sneaking in'…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button
            className="chat-send-btn"
            type="submit"
            disabled={loading || !query.trim()}
            aria-label="Search"
          >
            {loading ? (
              <Loader2 size={16} style={{ animation: "spin 0.8s linear infinite" }} />
            ) : (
              <Search size={16} />
            )}
          </button>
        </form>

        <div className="search-suggestions">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              className="search-chip"
              onClick={() => runSearch(s)}
              disabled={loading}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="error-state">
          <div className="error-state-title">Search failed</div>
          <div>{error}</div>
        </div>
      )}

      {!error && searched && results.length === 0 && (
        <div className="empty-state">No matching verses found.</div>
      )}

      {!error && results.length > 0 && (
        <div className="search-results">
          <div className="search-results-caption">
            Ranked by meaning · cosine similarity of word-embedding vectors
          </div>
          {results.map((r) => (
            <button
              key={r.verse_number}
              className="search-result"
              onClick={() => onSelectVerse?.(r.verse_number)}
              title={`Go to Jude 1:${r.verse_number}`}
            >
              <div className="search-result-head">
                <span className="search-result-ref">Jude 1:{r.verse_number}</span>
                <span className="search-result-score">{r.score.toFixed(3)}</span>
              </div>
              <div className="search-score-track">
                <div
                  className="search-score-fill"
                  style={{ width: `${Math.max(0, Math.min(1, r.score)) * 100}%` }}
                />
              </div>
              <div className="search-result-text">{r.text}</div>
            </button>
          ))}
        </div>
      )}

      {!searched && !loading && !error && (
        <div className="empty-state">
          Search the book of Jude by meaning. Try a suggestion above — the
          matches share few or no words with your query, yet land on the right
          verses.
        </div>
      )}
    </div>
  )
}
