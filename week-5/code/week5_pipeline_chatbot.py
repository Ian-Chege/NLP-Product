# =============================================================================
# Week 5 – CAT 1 Preparation: Complete NLP Pipeline and Mini Chatbot
# =============================================================================
#
# This script consolidates all NLP techniques from Weeks 1–4:
#
#   Week 1  → Tokenization and stopword removal
#   Week 2  → N-gram models and POS tagging
#   Week 3  → HMM concepts (sequence labeling)
#   Week 4  → Dependency parsing and semantic similarity
#
# It also introduces a rule-based chatbot to demonstrate how NLP
# powers conversational systems.
#
# =============================================================================

import nltk
import spacy

nltk.download("punkt",                          quiet=True)
nltk.download("punkt_tab",                      quiet=True)
nltk.download("stopwords",                      quiet=True)
nltk.download("averaged_perceptron_tagger",     quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.util import ngrams
from collections import Counter

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    raise


# =============================================================================
# PRACTICAL TASK 1 – COMPLETE NLP PIPELINE
# =============================================================================
# We run the full preprocessing pipeline on a sentence, showing each step
# and how the text changes from raw input to analysed output.
#
# This is the same pipeline that powers the AI Bible Study Companion backend.
# =============================================================================

text = (
    "Natural Language Processing helps computers understand human language "
    "by breaking sentences into words and analysing their grammatical roles."
)

print("=" * 65)
print("PRACTICAL TASK 1 – COMPLETE NLP PIPELINE")
print("=" * 65)
print(f"\nInput text:\n  \"{text}\"\n")


# ── Step 1: Tokenization ──────────────────────────────────────────────────────
tokens = word_tokenize(text)

print("─" * 65)
print("Step 1 – TOKENIZATION")
print("─" * 65)
print(f"  Total tokens  : {len(tokens)}")
print(f"  Token list    : {tokens}\n")


# ── Step 2: Stopword Removal ──────────────────────────────────────────────────
stop_words = set(stopwords.words("english"))
filtered = [w for w in tokens if w.isalpha() and w.lower() not in stop_words]

print("─" * 65)
print("Step 2 – STOPWORD REMOVAL")
print("─" * 65)
print(f"  Before removal : {len([t for t in tokens if t.isalpha()])} words")
print(f"  After removal  : {len(filtered)} words")
print(f"  Filtered list  : {filtered}\n")


# ── Step 3: POS Tagging ───────────────────────────────────────────────────────
READABLE = {
    "NN": "Noun",       "NNS": "Noun",      "NNP": "Proper Noun", "NNPS": "Proper Noun",
    "VB": "Verb",       "VBD": "Verb",      "VBG": "Verb",        "VBN": "Verb",
    "VBP": "Verb",      "VBZ": "Verb",
    "JJ": "Adjective",  "RB": "Adverb",
    "IN": "Preposition","DT": "Determiner", "CC": "Conjunction",  "PRP": "Pronoun",
}

tagged = pos_tag(filtered)

print("─" * 65)
print("Step 3 – POS TAGGING")
print("─" * 65)
print(f"  {'Word':<22} {'Tag':<8} {'Label'}")
print("  " + "-" * 40)
for word, tag in tagged:
    label = READABLE.get(tag, "Other")
    print(f"  {word:<22} {tag:<8} {label}")

pos_counts = Counter(READABLE.get(tag, "Other") for _, tag in tagged)
print(f"\n  Grammatical breakdown:")
for label, count in pos_counts.most_common():
    print(f"    {label:<15} : {count}")


# ── Step 4: N-grams ───────────────────────────────────────────────────────────
bigrams  = list(ngrams(filtered, 2))
trigrams = list(ngrams(filtered, 3))

print(f"\n─" + "─" * 64)
print("Step 4 – N-GRAMS")
print("─" * 65)
print(f"  Total bigrams  : {len(bigrams)}")
print(f"  Total trigrams : {len(trigrams)}")
print(f"  Sample bigrams : {bigrams[:5]}")
print(f"  Sample trigrams: {trigrams[:3]}")


# ── Step 5: Dependency Parsing ────────────────────────────────────────────────
doc = nlp(text)

print(f"\n─" + "─" * 64)
print("Step 5 – DEPENDENCY PARSING")
print("─" * 65)
print(f"  {'Word':<22} {'Dep Role':<14} {'Head Word'}")
print("  " + "-" * 52)
for token in doc:
    print(f"  {token.text:<22} {token.dep_:<14} {token.head.text}")

subject = next((t.text for t in doc if t.dep_ == "nsubj"), "—")
root    = next((t.text for t in doc if t.dep_ == "ROOT"),  "—")
obj     = next((t.text for t in doc if t.dep_ in ("dobj", "attr")), "—")
print(f"\n  Key roles → Subject: '{subject}'  |  Verb: '{root}'  |  Object: '{obj}'")


# ── Step 6: Semantic Similarity ───────────────────────────────────────────────
sentence_a = nlp("NLP helps computers understand language.")
sentence_b = nlp("Artificial intelligence enables machines to process text.")
score = sentence_a.similarity(sentence_b)

print(f"\n─" + "─" * 64)
print("Step 6 – SEMANTIC SIMILARITY")
print("─" * 65)
print(f"  Sentence A : \"{sentence_a}\"")
print(f"  Sentence B : \"{sentence_b}\"")
print(f"  Score      : {score:.4f}  ({'high similarity' if score > 0.75 else 'moderate similarity'})")

print(f"\n{'=' * 65}")
print("PIPELINE COMPLETE — 6 steps processed one sentence end to end")
print("  Raw text → tokens → filtered → POS tagged → n-grams → parsed → similarity")
print("=" * 65)


# =============================================================================
# PRACTICAL TASK 2 – MINI CHATBOT SIMULATION
# =============================================================================
# Theme: Self-Driving Cars
#
# A rule-based chatbot works by checking whether certain keywords appear
# in the user's message and returning a matching response.
#
# This is the simplest form of NLP-powered conversation:
#   user input → tokenize → keyword match → natural response
#
# We first run a demo with preset inputs so you can screenshot the output,
# then offer an interactive mode so you can try it live.
# =============================================================================

def chatbot_response(user_input):
    """Return a response based on keywords in the user's message."""
    # Tokenize into individual words to avoid substring false matches
    words_in_input = set(word_tokenize(user_input.lower()))

    if words_in_input & {"hello", "hi", "hey"} or "good morning" in user_input.lower():
        return (
            "Hello, I'm AutoBot, your self-driving car guide. "
            "Ask me anything about autonomous vehicles — how they work, "
            "safety, or when they'll be on every road."
        )
    elif words_in_input & {"work", "how", "sensors", "technology", "detect"}:
        return (
            "Self-driving cars use a combination of cameras, radar, LiDAR, "
            "and ultrasonic sensors to build a 360-degree picture of the road. "
            "An onboard AI processes this data in real time to steer, brake, "
            "and make decisions — all without a human touching the wheel."
        )
    elif words_in_input & {"safe", "safety", "accident", "crash", "dangerous"}:
        return (
            "Safety is the core promise. Autonomous vehicles don't get tired, "
            "distracted, or drunk. Studies show that over 90% of road accidents "
            "are caused by human error — self-driving technology directly targets "
            "that statistic. That said, the technology is still maturing and "
            "edge cases like bad weather remain active research challenges."
        )
    elif words_in_input & {"tesla", "waymo", "company", "companies", "who", "making", "built"}:
        return (
            "The biggest names right now are Waymo (Google's spin-off), Tesla, "
            "Cruise (GM), and Mobileye. Waymo currently runs fully driverless "
            "robotaxi services in San Francisco and Phoenix. Tesla calls its "
            "system Full Self-Driving but still requires driver supervision."
        )
    elif words_in_input & {"level", "levels", "autonomy", "l2", "l3", "l4", "l5"}:
        return (
            "There are six levels of autonomy defined by SAE International: "
            "Level 0 is fully manual. Level 2 (like Tesla Autopilot) assists "
            "but the driver must stay alert. Level 4 can handle most conditions "
            "without a driver. Level 5 is full autonomy in all conditions — "
            "nobody has reached that yet."
        )
    elif words_in_input & {"law", "legal", "regulation", "allowed", "banned", "government"}:
        return (
            "Regulations vary by country. In the US, California and Arizona "
            "allow commercial robotaxi operations. The EU is developing a "
            "unified framework. Most countries require a safety driver or "
            "restrict testing to designated zones. Full public deployment "
            "is still years away in most places."
        )
    elif words_in_input & {"when", "future", "ready", "available", "year"}:
        return (
            "Level 4 vehicles are already operating in limited areas today. "
            "Widespread consumer availability depends on regulation, cost, and "
            "edge-case safety. Most industry experts expect broad deployment "
            "in cities by the early 2030s, with rural and highway coverage "
            "following after."
        )
    elif words_in_input & {"bye", "goodbye", "thanks", "thank"}:
        return (
            "Thanks for chatting! The road to fully autonomous vehicles is "
            "being paved one algorithm at a time. Stay curious — and buckle up!"
        )
    else:
        return (
            "That's an interesting angle — I'm still learning! Try asking me "
            "how self-driving cars work, whether they're safe, which companies "
            "are leading the space, or when they'll be widely available."
        )


print("\n" + "=" * 65)
print("PRACTICAL TASK 2 – MINI CHATBOT SIMULATION")
print("=" * 65)
print("Theme: Self-Driving Cars | AutoBot — your autonomous vehicle guide")
print("-" * 65)
print("Demo: running through a simulated conversation\n")

demo_inputs = [
    "Hello!",
    "How exactly do self-driving cars detect the road?",
    "Are they actually safe though?",
    "Which companies are building them?",
    "What are the levels of autonomy?",
    "Is it even legal to use them?",
    "When will everyone have one?",
    "Wow! Thanks!",
]

for user_msg in demo_inputs:
    response = chatbot_response(user_msg)
    print(f"  You    : {user_msg}")
    print(f"  AutoBot: {response}")
    print()


# =============================================================================
# MINI PROJECT – STUDENT ACADEMIC ASSISTANT CHATBOT FOR NLP UNIT
# =============================================================================
# A separate chatbot focused on helping students with academic questions.
# This is the mini project deliverable — run it, type your own messages,
# and screenshot the conversation for Fig 3.
#
# To exit: type 'bye'.
# =============================================================================

def academic_bot_response(user_input):
    """Student Academic Assistant — responds to academic and unit inquiries for NLP class."""
    words_in_input = set(word_tokenize(user_input.lower()))

    if words_in_input & {"hello", "hi", "hey"} or "good morning" in user_input.lower():
        return (
            "Hello! I'm your Student Academic Assistant. I can help with "
            "exam prep, course content, assignments, and study tips for the NLP unit. "
            "What would you like to know?"
        )
    elif words_in_input & {"exam", "cat", "test", "assessment"}:
        return (
            "CAT 1 covers everything from Weeks 1 to 5: tokenization, "
            "stopword removal, POS tagging, N-grams, HMMs, dependency "
            "parsing, and semantic similarity. Make sure you can explain "
            "each concept and read the output your scripts produce."
        )
    elif words_in_input & {"register", "unit", "registration", "enrol", "enroll"}:
        return (
            "Unit registration is handled through the student portal. "
            "Deadlines are usually in the last week of the first month of a semester. "
            "If you're having trouble, visit the registrar's office directly."
        )
    elif words_in_input & {"project", "assignment", "submit", "deadline"}:
        return (
            "For the Week 5 mini project, submit your Python source code, "
            "screenshots of the chatbot interaction, and a short reflection "
            "on challenges you faced. Check the logbook template your "
            "lecturer shared for the exact format."
        )
    elif words_in_input & {"nlp", "natural"} or "natural language" in user_input.lower():
        return (
            "NLP — Natural Language Processing — is the branch of AI that "
            "teaches computers to read, understand, and generate human language. "
            "You've been applying it all semester: every script you wrote is NLP."
        )
    elif words_in_input & {"study", "tip", "tips", "prepare", "revision", "revise"}:
        return (
            "Best study tip: run each week's script and read the output line "
            "by line. Understanding what each print statement shows is more "
            "useful than memorising definitions. Also, review the student "
            "reflection sections in your weekly READMEs."
        )
    elif words_in_input & {"bye", "goodbye", "thanks", "thank"}:
        return (
            "Good luck with your studies! You've got this. "
            "Remember — understanding the output matters more than memorising the code."
        )
    else:
        return (
            "I'm not sure about that one. Try asking me about the CAT exam, "
            "unit registration, the mini project, or NLP study tips."
        )


print("=" * 65)
print("MINI PROJECT – STUDENT ACADEMIC ASSISTANT CHATBOT")
print("=" * 65)
print("Type a message and press Enter. Type 'bye' to exit.\n")

while True:
    try:
        user_input = input("  You: ").strip()
    except EOFError:
        break

    if not user_input:
        continue

    response = academic_bot_response(user_input)
    print(f"  Bot: {response}\n")

    if any(word in user_input.lower() for word in ["bye", "goodbye", "exit"]):
        break
