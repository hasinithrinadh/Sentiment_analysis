from transformers import pipeline

classifier = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)

def spam_classification(text):

    prompt = f"""
You are a spam detection system.

Classify this message:

"{text}"

Possible labels:
1. Spam
2. Not Spam

Answer with ONLY the label.
"""

    output = classifier(
        prompt,
        max_new_tokens=5,
        do_sample=False
    )

    response = output[0]["generated_text"]

    generated = response[len(prompt):].strip()

    print("--------------------------------")
    print("Text  :", text)
    print("Label :", generated)
    print("--------------------------------")


# Test cases
spam_classification(
    "Congratulations! You've won a $1000 gift card, click here to claim now!"
)

spam_classification(
    "Hey, are we still meeting for lunch tomorrow?"
)

spam_classification(
    "Limited time offer: buy one get one free on all products!"
)