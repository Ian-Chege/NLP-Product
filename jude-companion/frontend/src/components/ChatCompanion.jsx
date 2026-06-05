import { useState, useRef, useEffect } from "react"
import { Send, Loader2 } from "lucide-react"

function intro(n) {
  return `Hello! I'm looking at Jude 1:${n}.\n\nAsk me anything about this verse:\n• "What are the key words?"\n• "How is the sentence structured?"\n• "Which verses are most similar?"\n• "What themes run through the book?"\n• "Generate study questions"`
}

async function getResponse(message, verseNumber) {
  const lower = message.toLowerCase()
  const words = new Set((lower.match(/\b\w+\b/g) || []))

  try {
    // Keywords / topics
    if (words.has("keyword") || words.has("keywords") || words.has("key") || lower.includes("important word") || lower.includes("main word")) {
      const d = await fetch(`/api/analyze/${verseNumber}`).then(r => r.json())
      const list = d.keywords.map(k => `${k.word} (${k.count}x)`).join(", ")
      return `Key words in Jude 1:${verseNumber}:\n${list}\n\nThese are the most meaningful terms after filtering out common words like "the", "and", and "unto".`
    }

    // Sentence structure / dependency parsing
    if (words.has("structure") || words.has("grammar") || words.has("subject") || words.has("verb") || words.has("object") || words.has("parse") || lower.includes("sentence structure") || lower.includes("how is this")) {
      const d = await fetch(`/api/parse/${verseNumber}`).then(r => r.json())
      return `Sentence structure of Jude 1:${verseNumber}:\n\nSubject: ${d.subject || "unclear (archaic KJV sentence)"}\nMain verb: ${d.verb || "unclear"}\nObject: ${d.object || "unclear"}\n\nspaCy maps how every word connects to the main verb — this is dependency parsing from Week 4.`
    }

    // Related / similar verses
    if (words.has("similar") || words.has("related") || words.has("connected") || lower.includes("cross reference") || lower.includes("other verse") || lower.includes("like this")) {
      const d = await fetch(`/api/related/${verseNumber}`).then(r => r.json())
      const lines = d.map(r => `Jude 1:${r.verse_number} (score: ${r.similarity})\n"${r.text.slice(0, 80)}..."`).join("\n\n")
      return `Verses most similar in meaning to Jude 1:${verseNumber}:\n\n${lines}\n\nSimilarity scores come from spaCy's semantic vectors — higher means more related in meaning.`
    }

    // Book themes / n-grams
    if (words.has("theme") || words.has("themes") || words.has("pattern") || words.has("patterns") || words.has("recurring") || lower.includes("whole book") || lower.includes("all of jude")) {
      const d = await fetch(`/api/themes`).then(r => r.json())
      const top = d.bigrams.slice(0, 6).map(b => `"${b.phrase}" — ${b.count}x`).join("\n")
      return `Most recurring phrases across the whole Book of Jude:\n\n${top}\n\nThese n-grams reveal the themes Jude keeps returning to. This is the Week 2 technique applied to the full corpus.`
    }

    // Study questions / devotional (calls OpenAI)
    if (words.has("study") || words.has("question") || words.has("questions") || words.has("devotional") || words.has("reflect")) {
      const d = await fetch(`/api/study`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verse_number: verseNumber }),
      }).then(r => {
        if (!r.ok) throw new Error("Study generation failed — check your OpenAI API key")
        return r.json()
      })
      const qs = d.study_questions?.map((q, i) => `${i + 1}. ${q}`).join("\n") || ""
      return `Study questions for Jude 1:${verseNumber}:\n\n${qs}\n\nDevotional:\n${d.devotional || ""}`
    }

    // Token / word count
    if (words.has("token") || words.has("tokens") || lower.includes("word count") || lower.includes("how many words")) {
      const d = await fetch(`/api/analyze/${verseNumber}`).then(r => r.json())
      const sample = d.filtered_tokens.slice(0, 8).join(", ")
      return `Jude 1:${verseNumber} has ${d.token_count} tokens and ${d.unique_count} unique words.\n\nAfter removing stopwords, ${d.filtered_tokens.length} meaningful words remain:\n${sample}${d.filtered_tokens.length > 8 ? "..." : ""}`
    }

    // General summary / explain
    if (lower.includes("tell me") || lower.includes("explain") || lower.includes("about") || lower.includes("analyse") || lower.includes("analyze") || lower.includes("summarize")) {
      const d = await fetch(`/api/analyze/${verseNumber}`).then(r => r.json())
      const kws = d.keywords.slice(0, 3).map(k => k.word).join(", ")
      return `Jude 1:${verseNumber}\n"${d.text}"\n\n${d.token_count} tokens, ${d.unique_count} unique words.\nKey words: ${kws}\n\nAsk about sentence structure, related verses, or book themes to explore further.`
    }

    // Greeting
    if (words.has("hello") || words.has("hi") || words.has("hey")) {
      return intro(verseNumber)
    }

    // Default fallback
    return `I can help you explore Jude 1:${verseNumber}. Try asking:\n\n• "What are the key words?"\n• "How is this sentence structured?"\n• "Which verses are most similar?"\n• "What themes run through the book?"\n• "Generate study questions"`

  } catch (e) {
    return `Something went wrong: ${e.message}\n\nMake sure the backend is running on port 8000.`
  }
}

export default function ChatCompanion({ verseNumber }) {
  const [messages, setMessages] = useState([{ role: "bot", text: intro(verseNumber) }])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)
  const prevVerse = useRef(verseNumber)

  // Reset chat when verse changes
  useEffect(() => {
    if (prevVerse.current !== verseNumber) {
      prevVerse.current = verseNumber
      setMessages([{ role: "bot", text: intro(verseNumber) }])
    }
  }, [verseNumber])

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  async function send() {
    const text = input.trim()
    if (!text || loading) return
    setInput("")
    setMessages(m => [...m, { role: "user", text }])
    setLoading(true)
    const response = await getResponse(text, verseNumber)
    setMessages(m => [...m, { role: "bot", text: response }])
    setLoading(false)
  }

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-row chat-row-${msg.role}`}>
            <div className={`chat-bubble chat-bubble-${msg.role}`}>
              {msg.text}
            </div>
          </div>
        ))}
        {loading && (
          <div className="chat-row chat-row-bot">
            <div className="chat-bubble chat-bubble-bot chat-thinking">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-bar">
        <input
          className="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder={`Ask about Jude 1:${verseNumber}…`}
          disabled={loading}
          autoFocus
        />
        <button
          className="chat-send-btn"
          onClick={send}
          disabled={loading || !input.trim()}
          aria-label="Send"
        >
          {loading
            ? <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} />
            : <Send size={16} />
          }
        </button>
      </div>
    </div>
  )
}
