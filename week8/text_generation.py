from transformers import pipeline

print("=" * 60)
print("PRACTICAL TASK 2 — TEXT GENERATION (GPT-2)")
print("=" * 60)

generator = pipeline("text-generation", model="gpt2")

prompts = [
    "Artificial Intelligence will",
    "Natural Language Processing helps computers",
    "The future of deep learning is",
]

for prompt in prompts:
    print(f"\nPrompt : \"{prompt}\"")
    result = generator(prompt, max_length=50, num_return_sequences=1, truncation=True)
    generated = result[0]["generated_text"]
    print(f"Output : {generated}")
    print("-" * 50)

print("\n" + "=" * 60)
print("BONUS — GPT-2 continuation of a Jude verse fragment")
print("=" * 60)

jude_prompt = "Mercy unto you, and peace, and love,"
print(f"\nPrompt : \"{jude_prompt}\"")
result = generator(jude_prompt, max_length=60, num_return_sequences=1, truncation=True)
print(f"Output : {result[0]['generated_text']}")
print("\nNote: GPT-2 was not trained on biblical text — output is")
print("illustrative of sequence generation, not theological accuracy.")
