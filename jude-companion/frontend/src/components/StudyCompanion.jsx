import { useState } from "react"
import { BookOpen, HelpCircle, Link2, Feather, Loader2 } from "lucide-react"

function SectionTitle({ icon: Icon, children }) {
  return (
    <div className="section-card-title">
      <Icon size={14} style={{ color: "var(--color-accent)", flexShrink: 0 }} />
      {children}
    </div>
  )
}

export default function StudyCompanion({ verseNumber }) {
  const [study, setStudy] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [lastVerse, setLastVerse] = useState(null)

  async function generate() {
    setLoading(true)
    setError(null)
    setStudy(null)

    try {
      const res = await fetch("/api/study", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verse_number: verseNumber }),
      })
      if (!res.ok) {
        const msg = await res.json().catch(() => ({ detail: "Request failed" }))
        throw new Error(msg.detail ?? "Request failed")
      }
      const data = await res.json()
      setStudy(data)
      setLastVerse(verseNumber)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const isStale = lastVerse !== null && lastVerse !== verseNumber

  return (
    <div>
      <button
        className="generate-btn"
        onClick={generate}
        disabled={loading}
        aria-busy={loading}
      >
        {loading ? (
          <>
            <Loader2 size={15} style={{ animation: "spin 1s linear infinite" }} />
            Generating…
          </>
        ) : (
          <>
            <BookOpen size={15} />
            {study && isStale
              ? "Regenerate for this verse"
              : study
              ? "Regenerate study"
              : "Generate Study"}
          </>
        )}
      </button>

      {isStale && study && !loading && (
        <div
          style={{
            fontSize: "0.75rem",
            color: "var(--color-ink-3)",
            marginTop: "-0.875rem",
            marginBottom: "1rem",
          }}
        >
          Showing results for Jude 1:{lastVerse} — click to regenerate for 1:{verseNumber}
        </div>
      )}

      {error && (
        <div className="error-state" style={{ padding: "1rem", textAlign: "left", alignItems: "flex-start" }}>
          <div className="error-state-title">Generation failed</div>
          <div>{error}</div>
        </div>
      )}

      {loading && (
        <div>
          {/* Skeleton for study questions */}
          <div className="section-card">
            <div className="skeleton" style={{ height: "0.875rem", width: "40%", marginBottom: "0.875rem" }} />
            {[...Array(4)].map((_, i) => (
              <div key={i} style={{ display: "flex", gap: "0.75rem", marginBottom: "0.75rem" }}>
                <div className="skeleton" style={{ height: "0.75rem", width: "1rem", flexShrink: 0 }} />
                <div className="skeleton" style={{ height: "0.75rem", width: `${50 + (i % 3) * 15}%` }} />
              </div>
            ))}
          </div>
          {/* Skeleton for cross-refs */}
          <div className="section-card">
            <div className="skeleton" style={{ height: "0.875rem", width: "40%", marginBottom: "0.875rem" }} />
            {[...Array(3)].map((_, i) => (
              <div key={i} className="cross-ref-card" style={{ marginBottom: "0.5rem" }}>
                <div className="skeleton" style={{ height: "0.75rem", width: "30%", marginBottom: "0.375rem" }} />
                <div className="skeleton" style={{ height: "0.75rem", width: "80%" }} />
              </div>
            ))}
          </div>
          {/* Skeleton for devotional */}
          <div className="section-card">
            <div className="skeleton" style={{ height: "0.875rem", width: "35%", marginBottom: "0.875rem" }} />
            <div className="skeleton" style={{ height: "3.5rem", width: "100%" }} />
          </div>
        </div>
      )}

      {study && !loading && (
        <div>
          {/* Study questions */}
          <div className="section-card">
            <SectionTitle icon={HelpCircle}>Study Questions</SectionTitle>
            {study.study_questions?.map((q, i) => (
              <div key={i} className="study-question">
                <span className="question-num">{i + 1}.</span>
                <span>{q}</span>
              </div>
            ))}
          </div>

          {/* Cross references */}
          <div className="section-card">
            <SectionTitle icon={Link2}>Cross References</SectionTitle>
            <div className="cross-ref-grid">
              {study.cross_references?.map((ref, i) => (
                <div key={i} className="cross-ref-card">
                  <div className="cross-ref-ref">{ref.reference}</div>
                  <div className="cross-ref-expl">{ref.explanation}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Devotional */}
          <div className="section-card">
            <SectionTitle icon={Feather}>Devotional</SectionTitle>
            <blockquote className="devotional">{study.devotional}</blockquote>
          </div>
        </div>
      )}

      {!study && !loading && !error && (
        <div className="empty-state">
          <BookOpen size={24} style={{ color: "var(--color-ink-4)", marginBottom: "0.25rem" }} />
          <div style={{ fontWeight: 500, color: "var(--color-ink-2)" }}>
            No study generated yet
          </div>
          <div>
            Click <strong>Generate Study</strong> to get AI-powered questions,
            cross-references, and a devotional for this verse.
          </div>
        </div>
      )}
    </div>
  )
}
