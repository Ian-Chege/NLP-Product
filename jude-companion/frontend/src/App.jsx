import { useEffect, useState } from "react"
import "./App.css"
import VerseExplorer from "./components/VerseExplorer"
import PipelineView from "./components/PipelineView"
import FrequencyChart from "./components/FrequencyChart"
import StudyCompanion from "./components/StudyCompanion"
import { BookMarked, BarChart2, Sparkles } from "lucide-react"

const TABS = [
  { id: "nlp",       label: "NLP Analysis",   icon: BookMarked },
  { id: "frequency", label: "Word Frequency",  icon: BarChart2 },
  { id: "study",     label: "Study Companion", icon: Sparkles },
]

export default function App() {
  const [selectedVerse, setSelectedVerse] = useState(1)
  const [activeTab, setActiveTab] = useState("nlp")
  const [verseText, setVerseText] = useState("")

  useEffect(() => {
    fetch(`/api/verse/${selectedVerse}`)
      .then((r) => r.json())
      .then((v) => setVerseText(v.text))
      .catch(() => {})
  }, [selectedVerse])

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <VerseExplorer
          selectedVerse={selectedVerse}
          onSelectVerse={setSelectedVerse}
        />
      </aside>

      {/* Main */}
      <main className="main-content">
        {/* Verse header */}
        <div className="content-header">
          <h1 className="content-heading">
            Jude 1:{selectedVerse}
          </h1>
          {verseText && (
            <span className="content-verse-text" title={verseText}>
              {verseText}
            </span>
          )}
        </div>

        {/* Tab bar */}
        <div className="tab-bar" role="tablist">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`tab-trigger${activeTab === id ? " active" : ""}`}
              role="tab"
              aria-selected={activeTab === id}
              aria-controls={`panel-${id}`}
              onClick={() => setActiveTab(id)}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Tab panels */}
        <div className="tab-panel" id={`panel-${activeTab}`} role="tabpanel">
          {activeTab === "nlp" && (
            <PipelineView verseNumber={selectedVerse} />
          )}
          {activeTab === "frequency" && (
            <FrequencyChart verseNumber={selectedVerse} />
          )}
          {activeTab === "study" && (
            <StudyCompanion verseNumber={selectedVerse} />
          )}
        </div>
      </main>
    </div>
  )
}
