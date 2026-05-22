import { useEffect, useState } from "react"

const POS_CLASS = {
  Noun:      "pos-noun",
  Verb:      "pos-verb",
  Adjective: "pos-adjective",
  Adverb:    "pos-adverb",
  Other:     "pos-other",
}

function StepHeader({ num, label, count }) {
  return (
    <div className="step-header">
      <span className="step-number">{num}</span>
      <span className="step-label">{label}</span>
      {count != null && (
        <span className="step-count">{count} token{count !== 1 ? "s" : ""}</span>
      )}
    </div>
  )
}

function SkeletonStep() {
  return (
    <div className="pipeline-step">
      <div className="skeleton" style={{ height: "1rem", width: "30%", marginBottom: "0.625rem" }} />
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.375rem" }}>
        {Array.from({ length: 12 }, (_, i) => (
          <div
            key={i}
            className="skeleton"
            style={{
              height: "1.625rem",
              width: `${3 + (i % 4)}rem`,
              borderRadius: "9999px",
            }}
          />
        ))}
      </div>
    </div>
  )
}

export default function PipelineView({ verseNumber }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetch(`/api/analyze/${verseNumber}`)
      .then((r) => {
        if (!r.ok) throw new Error("Failed to fetch analysis")
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
  }, [verseNumber])

  if (error) {
    return (
      <div className="error-state">
        <div className="error-state-title">Analysis failed</div>
        <div>{error}</div>
      </div>
    )
  }

  if (loading) {
    return (
      <div>
        <div className="stat-row">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="stat-card">
              <div className="skeleton" style={{ height: "0.75rem", width: "60%" }} />
              <div className="skeleton" style={{ height: "2rem", width: "40%", marginTop: "0.5rem" }} />
            </div>
          ))}
        </div>
        <SkeletonStep />
        <SkeletonStep />
        <SkeletonStep />
        <SkeletonStep />
      </div>
    )
  }

  return (
    <div>
      {/* Stat cards */}
      <div className="stat-row">
        <div className="stat-card">
          <div className="stat-label">Total tokens</div>
          <div className="stat-value">{data.token_count}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Unique tokens</div>
          <div className="stat-value">{data.unique_count}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Keywords</div>
          <div className="stat-value">{data.keywords.length}</div>
        </div>
      </div>

      {/* Step 1 — All tokens */}
      <div className="pipeline-step">
        <StepHeader num={1} label="All tokens" count={data.tokens.length} />
        <div className="token-wrap">
          {data.tokens.map((tok, i) => (
            <span key={i} className="token token-raw">
              {tok}
            </span>
          ))}
        </div>
      </div>

      {/* Step 2 — After stopword removal */}
      <div className="pipeline-step">
        <StepHeader
          num={2}
          label="After stopword removal"
          count={data.filtered_tokens.length}
        />
        <div className="token-wrap">
          {data.filtered_tokens.map((tok, i) => (
            <span key={i} className="token token-filtered">
              {tok}
            </span>
          ))}
        </div>
      </div>

      {/* Step 3 — POS tags */}
      <div className="pipeline-step">
        <StepHeader num={3} label="Part-of-speech tags" count={data.pos_tags.length} />
        <div className="token-wrap">
          {data.pos_tags.map((item, i) => (
            <span key={i} className={`pos-token ${POS_CLASS[item.label] ?? "pos-other"}`}>
              {item.word}
              <span className="pos-label">{item.label}</span>
            </span>
          ))}
        </div>

        {/* POS legend */}
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "0.5rem",
            marginTop: "0.75rem",
          }}
        >
          {Object.entries(POS_CLASS).map(([label, cls]) => (
            <span
              key={label}
              className={`token ${cls.replace("pos-", "pos-")}`}
              style={{ fontSize: "0.6875rem", padding: "0.125rem 0.5rem" }}
            >
              {label}
            </span>
          ))}
        </div>
      </div>

      {/* Step 4 — Keywords */}
      <div className="pipeline-step">
        <StepHeader num={4} label="Top keywords" count={data.keywords.length} />
        <div className="token-wrap">
          {data.keywords.map((kw, i) => (
            <span key={i} className="token token-keyword">
              {kw.word}
              <span
                style={{
                  marginLeft: "0.375rem",
                  opacity: 0.75,
                  fontSize: "0.6875rem",
                  fontWeight: 400,
                }}
              >
                ×{kw.count}
              </span>
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
