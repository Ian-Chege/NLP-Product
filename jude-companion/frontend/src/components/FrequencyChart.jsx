import { useEffect, useState } from "react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"

const ACCENT = "oklch(45% 0.22 270)"
const ACCENT_LIGHT = "oklch(88% 0.08 270)"

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const { word, count } = payload[0].payload
  return (
    <div
      style={{
        background: "var(--color-paper)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-md)",
        padding: "0.5rem 0.75rem",
        fontSize: "0.8125rem",
        boxShadow: "0 2px 8px oklch(0% 0 0 / 0.08)",
      }}
    >
      <span style={{ fontWeight: 600, color: "var(--color-ink)" }}>{word}</span>
      <span style={{ color: "var(--color-ink-3)", marginLeft: "0.5rem" }}>
        {count} {count === 1 ? "occurrence" : "occurrences"}
      </span>
    </div>
  )
}

export default function FrequencyChart({ verseNumber }) {
  const [mode, setMode] = useState("verse")
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    const url =
      mode === "verse"
        ? `/api/frequency?verse=${verseNumber}`
        : "/api/frequency"

    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error("Failed to fetch frequency data")
        return r.json()
      })
      .then((d) => {
        setData(d)
        setLoading(false)
      })
      .catch((e) => {
        setError(e.message)
        setLoading(false)
      })
  }, [mode, verseNumber])

  return (
    <div>
      {/* Toggle */}
      <div className="freq-toggle" role="group" aria-label="Frequency scope">
        <button
          className={`freq-toggle-btn${mode === "verse" ? " active" : ""}`}
          onClick={() => setMode("verse")}
        >
          This verse
        </button>
        <button
          className={`freq-toggle-btn${mode === "book" ? " active" : ""}`}
          onClick={() => setMode("book")}
        >
          Whole book
        </button>
      </div>

      {error && (
        <div className="error-state">
          <div className="error-state-title">Failed to load</div>
          <div>{error}</div>
        </div>
      )}

      {loading && !error && (
        <div>
          {[...Array(8)].map((_, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
                marginBottom: "0.625rem",
              }}
            >
              <div
                className="skeleton"
                style={{ height: "0.75rem", width: "4.5rem", flexShrink: 0 }}
              />
              <div
                className="skeleton"
                style={{
                  height: "1.5rem",
                  width: `${25 + (i % 5) * 12}%`,
                  borderRadius: "var(--radius-sm)",
                }}
              />
            </div>
          ))}
        </div>
      )}

      {!loading && !error && data.length === 0 && (
        <div className="empty-state">No words found after stopword removal.</div>
      )}

      {!loading && !error && data.length > 0 && (
        <>
          <div
            style={{
              fontSize: "0.75rem",
              color: "var(--color-ink-3)",
              marginBottom: "1rem",
              fontWeight: 500,
            }}
          >
            Top {data.length} words ·{" "}
            {mode === "verse" ? `Jude 1:${verseNumber}` : "Full book (Jude 1:1–25)"}
            {" · "}stopwords removed
          </div>

          <ResponsiveContainer width="100%" height={Math.max(260, data.length * 32)}>
            <BarChart
              data={[...data].reverse()}
              layout="vertical"
              margin={{ top: 0, right: 20, left: 4, bottom: 0 }}
            >
              <CartesianGrid
                horizontal={false}
                strokeDasharray="3 3"
                stroke="var(--color-border)"
              />
              <XAxis
                type="number"
                allowDecimals={false}
                tick={{ fontSize: 11, fill: "var(--color-ink-3)" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="word"
                width={88}
                tick={{ fontSize: 12, fill: "var(--color-ink-2)", fontWeight: 500 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: "var(--color-accent-light)" }} />
              <Bar dataKey="count" radius={[0, 4, 4, 0]} maxBarSize={24}>
                {data.map((_, idx) => (
                  <Cell
                    key={idx}
                    fill={idx >= data.length - 3 ? ACCENT : ACCENT_LIGHT}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  )
}
