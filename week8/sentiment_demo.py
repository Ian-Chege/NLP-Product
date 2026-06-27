from transformers import pipeline

print("=" * 60)
print("PRACTICAL TASK 1 — SENTIMENT ANALYSIS (Transformers)")
print("=" * 60)

classifier = pipeline("sentiment-analysis")

samples = [
    ("Positive", "I enjoyed learning Natural Language Processing. The concepts were fascinating and the practical sessions were excellent."),
    ("Negative", "The assignment was too difficult and confusing. I struggled to understand the requirements and failed to complete it on time."),
    ("Neutral",  "Natural Language Processing is a field of Artificial Intelligence. It involves the study of how computers process human language."),
]

for category, text in samples:
    result = classifier(text)[0]
    print(f"\n[{category} sample]")
    print(f"  Text  : {text[:70]}...")
    print(f"  Label : {result['label']}")
    print(f"  Score : {result['score']:.4f} ({result['score']*100:.1f}% confidence)")

print("\n" + "=" * 60)
print("BONUS — Applied to the Book of Jude")
print("=" * 60)

jude_samples = [
    (3,  "Beloved, when I gave all diligence to write unto you of the common salvation, it was needful for me to write unto you, and exhort you that ye should earnestly contend for the faith which was once delivered unto the saints."),
    (4,  "For there are certain men crept in unawares, who were before of old ordained to this condemnation, ungodly men, turning the grace of our God into lasciviousness, and denying the only Lord God, and our Lord Jesus Christ."),
    (24, "Now unto him that is able to keep you from falling, and to present you faultless before the presence of his glory with exceeding joy,"),
]

for verse_num, text in jude_samples:
    result = classifier(text[:512])[0]
    print(f"\n  Jude 1:{verse_num} → {result['label']} ({result['score']*100:.1f}%)")
    print(f"  \"{text[:80]}...\"")
