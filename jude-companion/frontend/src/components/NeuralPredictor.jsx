import { useState, useEffect } from "react"

export default function NeuralPredictor() {
  const [input, setInput] = useState("")
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(false)
  const [modelReady, setModelReady] = useState(false)
  const [checking, setChecking] = useState(true)

  // Tokenizer demo state
  const [tokInput, setTokInput] = useState("")
  const [tokResult, setTokResult] = useState(null)
  const [tokLoading, setTokLoading] = useState(false)

  // Poll until the neural model finishes training
  useEffect(() => {
    let timer
    const check = () => {
      fetch("/api/model-status")
        .then((r) => r.json())
        .then(({ ready }) => {
          if (ready) {
            setModelReady(true)
            setChecking(false)
          } else {
            timer = setTimeout(check, 2000)
          }
        })
        .catch(() => {
          timer = setTimeout(check, 3000)
        })
    }
    check()
    return () => clearTimeout(timer)
  }, [])

  const handlePredict = (e) => {
    e.preventDefault()
    if (!input.trim() || !modelReady) return
    setLoading(true)
    fetch(`/api/predict?text=${encodeURIComponent(input)}&top_n=6`)
      .then((r) => r.json())
      .then((data) => setPredictions(data.predictions || []))
      .catch(() => setPredictions([]))
      .finally(() => setLoading(false))
  }

  const handleTokenize = (e) => {
    e.preventDefault()
    if (!tokInput.trim() || !modelReady) return
    setTokLoading(true)
    fetch("/api/tokenize-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: tokInput }),
    })
      .then((r) => r.json())
      .then(setTokResult)
      .catch(() => setTokResult(null))
      .finally(() => setTokLoading(false))
  }

  const maxConf = predictions[0]?.confidence || 1

  return (
    <div className="pipeline-grid" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>

      {/* Model status banner */}
      {!modelReady && (
        <div style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 8,
          padding: "0.75rem 1rem",
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          color: "var(--muted)",
          fontSize: "0.85rem",
        }}>
          <span style={{ animation: "spin 1.2s linear infinite", display: "inline-block" }}>⟳</span>
          Training neural language model on the book of Jude… this takes about 30 seconds.
        </div>
      )}

      {/* Prediction card */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Next-Word Predictor</h2>
          <p className="card-description">
            Type a sentence fragment from Jude and the neural network predicts the most likely next word.
          </p>
        </div>
        <div className="card-content">
          <form onSubmit={handlePredict} style={{ display: "flex", gap: "0.5rem", marginBottom: "1.25rem" }}>
            <input
              className="search-input"
              style={{ flex: 1 }}
              type="text"
              placeholder={modelReady ? "e.g. Jude the servant of" : "Waiting for model…"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={!modelReady}
            />
            <button
              className="search-button"
              type="submit"
              disabled={!modelReady || loading || !input.trim()}
            >
              {loading ? "…" : "Predict"}
            </button>
          </form>

          {predictions.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              {predictions.map(({ word, confidence }, i) => (
                <div key={word} style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                  <span style={{
                    minWidth: 90,
                    fontWeight: i === 0 ? 600 : 400,
                    fontSize: "0.9rem",
                    color: i === 0 ? "var(--primary)" : "var(--foreground)",
                  }}>
                    {word}
                  </span>
                  <div style={{
                    flex: 1,
                    height: 10,
                    background: "var(--border)",
                    borderRadius: 5,
                    overflow: "hidden",
                  }}>
                    <div style={{
                      width: `${(confidence / maxConf) * 100}%`,
                      height: "100%",
                      background: i === 0 ? "var(--primary)" : "var(--muted-foreground)",
                      borderRadius: 5,
                      transition: "width 0.4s ease",
                    }} />
                  </div>
                  <span style={{ minWidth: 46, textAlign: "right", fontSize: "0.8rem", color: "var(--muted)" }}>
                    {(confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Tokenizer demo card */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Tokenizer Explorer</h2>
          <p className="card-description">
            See how the Keras Tokenizer assigns integer indices to words and converts text to sequences.
          </p>
        </div>
        <div className="card-content">
          <form onSubmit={handleTokenize} style={{ display: "flex", gap: "0.5rem", marginBottom: "1.25rem" }}>
            <input
              className="search-input"
              style={{ flex: 1 }}
              type="text"
              placeholder={modelReady ? "Enter any text to tokenize" : "Waiting for model…"}
              value={tokInput}
              onChange={(e) => setTokInput(e.target.value)}
              disabled={!modelReady}
            />
            <button
              className="search-button"
              type="submit"
              disabled={!modelReady || tokLoading || !tokInput.trim()}
            >
              {tokLoading ? "…" : "Tokenize"}
            </button>
          </form>

          {tokResult && (
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              {/* Word index table */}
              <div>
                <p style={{ fontSize: "0.75rem", color: "var(--muted)", marginBottom: "0.4rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Word Index
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                  {Object.entries(tokResult.word_index).map(([word, idx]) => (
                    <span key={word} style={{
                      display: "inline-flex", alignItems: "center", gap: 4,
                      background: "var(--surface)", border: "1px solid var(--border)",
                      borderRadius: 6, padding: "2px 8px", fontSize: "0.82rem",
                    }}>
                      <span style={{ color: "var(--foreground)" }}>{word}</span>
                      <span style={{ color: "var(--primary)", fontWeight: 600 }}>→ {idx}</span>
                    </span>
                  ))}
                </div>
              </div>

              {/* Integer sequence */}
              <div>
                <p style={{ fontSize: "0.75rem", color: "var(--muted)", marginBottom: "0.4rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Integer Sequence
                </p>
                <code style={{
                  display: "block",
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  padding: "0.5rem 0.75rem",
                  fontSize: "0.85rem",
                  color: "var(--primary)",
                  wordBreak: "break-all",
                }}>
                  [{tokResult.sequences[0]?.join(", ")}]
                </code>
              </div>

              <p style={{ fontSize: "0.78rem", color: "var(--muted)" }}>
                Vocabulary size: <strong>{tokResult.vocabulary_size}</strong> unique tokens in the book of Jude.
              </p>
            </div>
          )}
        </div>
      </div>

    </div>
  )
}
