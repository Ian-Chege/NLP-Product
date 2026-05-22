import { useEffect, useState } from "react"

function preview(text) {
  return text.split(/\s+/).slice(0, 5).join(" ") + "…"
}

export default function VerseExplorer({ selectedVerse, onSelectVerse }) {
  const [verses, setVerses] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch("/api/verses")
      .then((r) => r.json())
      .then((data) => {
        setVerses(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  return (
    <>
      <div className="sidebar-header">
        <div className="sidebar-title">Book of Jude</div>
        <div className="sidebar-subtitle">
          {loading ? "Loading…" : `${verses.length} verses · KJV`}
        </div>
      </div>

      <div className="verse-list" role="listbox" aria-label="Jude verses">
        {loading
          ? Array.from({ length: 10 }, (_, i) => (
              <div key={i} style={{ padding: "0.5rem 1rem" }}>
                <div
                  className="skeleton"
                  style={{ height: "0.875rem", width: i % 3 === 0 ? "65%" : "80%" }}
                />
              </div>
            ))
          : verses.map((v) => (
              <button
                key={v.verse_number}
                className={`verse-item${v.verse_number === selectedVerse ? " active" : ""}`}
                role="option"
                aria-selected={v.verse_number === selectedVerse}
                onClick={() => onSelectVerse(v.verse_number)}
              >
                <span className="verse-num">{v.verse_number}</span>
                <span className="verse-preview">{preview(v.text)}</span>
              </button>
            ))}
      </div>
    </>
  )
}
