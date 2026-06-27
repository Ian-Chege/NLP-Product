import { useState, useEffect } from "react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"

const POS_COLOR = "#4ade80"
const NEG_COLOR = "#f87171"
const POS_DIM   = "#bbf7d0"
const NEG_DIM   = "#fecaca"

function SentimentBadge({ label, score }) {
  const positive = label === "POSITIVE"
  return (
    <div style={{
      display: "inline-flex", flexDirection: "column", alignItems: "center",
      gap: "0.35rem", padding: "0.75rem 1.5rem",
      background: positive ? "#f0fdf4" : "#fef2f2",
      border: `2px solid ${positive ? POS_COLOR : NEG_COLOR}`,
      borderRadius: 12,
    }}>
      <span style={{ fontSize: "1.5rem" }}>{positive ? "😊" : "⚠️"}</span>
      <span style={{
        fontWeight: 700, fontSize: "0.9rem",
        color: positive ? "#16a34a" : "#dc2626",
        letterSpacing: "0.08em",
      }}>
        {label}
      </span>
      <span style={{ fontSize: "0.78rem", color: "var(--muted)" }}>
        {(score * 100).toFixed(1)}% confidence
      </span>
    </div>
  )
}

function ConfidenceBar({ score, label }) {
  const positive = label === "POSITIVE"
  return (
    <div style={{ marginTop: "0.5rem" }}>
      <div style={{
        height: 8, background: "var(--border)", borderRadius: 4, overflow: "hidden",
      }}>
        <div style={{
          width: `${score * 100}%`, height: "100%",
          background: positive ? POS_COLOR : NEG_COLOR,
          borderRadius: 4, transition: "width 0.5s ease",
        }} />
      </div>
    </div>
  )
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 8, padding: "0.5rem 0.75rem", fontSize: "0.8rem",
      maxWidth: 260,
    }}>
      <div style={{ fontWeight: 600, marginBottom: 2 }}>Jude 1:{d.verse_number}</div>
      <div style={{ color: d.label === "POSITIVE" ? "#16a34a" : "#dc2626", marginBottom: 4 }}>
        {d.label} · {(d.score * 100).toFixed(1)}%
      </div>
      <div style={{ color: "var(--muted)", fontSize: "0.75rem", lineHeight: 1.4 }}>
        {d.text.slice(0, 80)}…
      </div>
    </div>
  )
}

export default function SentimentAnalysis({ verseNumber }) {
  const [verse, setVerse]           = useState(null)
  const [book, setBook]             = useState(null)
  const [view, setView]             = useState("verse")
  const [loading, setLoading]       = useState(false)
  const [bookLoading, setBookLoading] = useState(false)
  const [ready, setReady]           = useState(false)
  const [checking, setChecking]     = useState(true)

  // Poll until DistilBERT pipeline is loaded
  useEffect(() => {
    let timer
    const check = () => {
      fetch("/api/sentiment-status")
        .then(r => r.json())
        .then(({ ready }) => {
          if (ready) { setReady(true); setChecking(false) }
          else timer = setTimeout(check, 2500)
        })
        .catch(() => { timer = setTimeout(check, 3000) })
    }
    check()
    return () => clearTimeout(timer)
  }, [])

  // Fetch single verse sentiment whenever verse or readiness changes
  useEffect(() => {
    if (!ready || !verseNumber) return
    setLoading(true)
    fetch(`/api/sentiment/${verseNumber}`)
      .then(r => r.json())
      .then(setVerse)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [verseNumber, ready])

  const loadBook = () => {
    if (book) { setView("book"); return }
    setBookLoading(true)
    fetch("/api/sentiment")
      .then(r => r.json())
      .then(data => { setBook(data); setView("book") })
      .catch(() => {})
      .finally(() => setBookLoading(false))
  }

  const chartData = book?.verses?.map(v => ({
    ...v,
    value: v.score,
  }))

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>

      {/* Loading banner */}
      {!ready && (
        <div style={{
          background: "var(--surface)", border: "1px solid var(--border)",
          borderRadius: 8, padding: "0.75rem 1rem",
          display: "flex", alignItems: "center", gap: "0.5rem",
          color: "var(--muted)", fontSize: "0.85rem",
        }}>
          <span style={{ animation: "spin 1.2s linear infinite", display: "inline-block" }}>⟳</span>
          Loading DistilBERT sentiment model…
        </div>
      )}

      {/* View toggle */}
      {ready && (
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button
            className={`tab-trigger${view === "verse" ? " active" : ""}`}
            onClick={() => setView("verse")}
          >
            This verse
          </button>
          <button
            className={`tab-trigger${view === "book" ? " active" : ""}`}
            onClick={loadBook}
            disabled={bookLoading}
          >
            {bookLoading ? "Analysing…" : "Whole book"}
          </button>
        </div>
      )}

      {/* ── Single verse view ── */}
      {view === "verse" && ready && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Verse Sentiment — Jude 1:{verseNumber}</h2>
            <p className="card-description">
              Classified by DistilBERT fine-tuned on SST-2 (Stanford Sentiment Treebank).
            </p>
          </div>
          <div className="card-content">
            {loading && (
              <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>Analysing…</div>
            )}
            {!loading && verse && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                <p style={{ fontSize: "0.9rem", color: "var(--muted)", lineHeight: 1.6 }}>
                  "{verse.text}"
                </p>
                <div style={{ display: "flex", alignItems: "flex-start", gap: "1.5rem", flexWrap: "wrap" }}>
                  <SentimentBadge label={verse.label} score={verse.score} />
                  <div style={{ flex: 1, minWidth: 180 }}>
                    <p style={{ fontSize: "0.78rem", color: "var(--muted)", marginBottom: "0.4rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      Confidence
                    </p>
                    <ConfidenceBar score={verse.score} label={verse.label} />
                    <p style={{ fontSize: "0.78rem", color: "var(--muted)", marginTop: "0.5rem" }}>
                      {verse.label === "POSITIVE"
                        ? "The Transformer model reads this verse as carrying a positive, affirming tone."
                        : "The Transformer model reads this verse as carrying a negative, warning tone."}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Book-wide view ── */}
      {view === "book" && book && (
        <>
          {/* Summary cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem" }}>
            {[
              { label: "Total verses", value: book.summary.total, color: "var(--foreground)" },
              { label: "Positive", value: book.summary.positive, color: "#16a34a" },
              { label: "Negative", value: book.summary.negative, color: "#dc2626" },
            ].map(({ label, value, color }) => (
              <div key={label} className="card" style={{ textAlign: "center" }}>
                <div className="card-content" style={{ paddingTop: "0.75rem" }}>
                  <div style={{ fontSize: "2rem", fontWeight: 700, color }}>{value}</div>
                  <div style={{ fontSize: "0.78rem", color: "var(--muted)", marginTop: 2 }}>{label}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Bar chart */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Sentiment Across All 25 Verses</h2>
              <p className="card-description">Each bar is one verse — height = confidence score. Green = POSITIVE, red = NEGATIVE.</p>
            </div>
            <div className="card-content">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
                  <XAxis dataKey="verse_number" tick={{ fontSize: 10 }} label={{ value: "Verse", position: "insideBottom", offset: -2, fontSize: 10 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                    {chartData.map((d, i) => (
                      <Cell key={i} fill={d.label === "POSITIVE" ? POS_COLOR : NEG_COLOR} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Highlights */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
            {[
              { title: "Most Positive Verse", verse: book.summary.most_positive, color: "#16a34a", bg: "#f0fdf4", border: POS_COLOR },
              { title: "Most Negative Verse", verse: book.summary.most_negative, color: "#dc2626", bg: "#fef2f2", border: NEG_COLOR },
            ].map(({ title, verse, color, bg, border }) => (
              <div key={title} className="card" style={{ background: bg, borderColor: border }}>
                <div className="card-header">
                  <h2 className="card-title" style={{ color, fontSize: "0.85rem" }}>{title}</h2>
                  <p className="card-description" style={{ fontWeight: 600 }}>Jude 1:{verse.verse_number} · {(verse.score * 100).toFixed(1)}%</p>
                </div>
                <div className="card-content">
                  <p style={{ fontSize: "0.8rem", color: "var(--muted)", lineHeight: 1.5 }}>
                    "{verse.text.slice(0, 120)}…"
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Full verse list */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">All Verses</h2>
            </div>
            <div className="card-content" style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
              {book.verses.map(v => (
                <div key={v.verse_number} style={{
                  display: "flex", alignItems: "center", gap: "0.75rem",
                  padding: "0.4rem 0.6rem", borderRadius: 6,
                  background: v.label === "POSITIVE" ? POS_DIM : NEG_DIM,
                }}>
                  <span style={{ minWidth: 32, fontWeight: 600, fontSize: "0.8rem", color: "var(--muted)" }}>
                    {v.verse_number}
                  </span>
                  <div style={{ flex: 1, height: 6, background: "white", borderRadius: 3, overflow: "hidden" }}>
                    <div style={{
                      width: `${v.score * 100}%`, height: "100%",
                      background: v.label === "POSITIVE" ? POS_COLOR : NEG_COLOR,
                      borderRadius: 3,
                    }} />
                  </div>
                  <span style={{
                    minWidth: 70, fontSize: "0.75rem", fontWeight: 600,
                    color: v.label === "POSITIVE" ? "#16a34a" : "#dc2626",
                  }}>
                    {v.label} {(v.score * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
