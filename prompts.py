from __future__ import annotations

import json

SENTIMENT_LABELS = [
    "Strongly Negative",
    "Weakly Negative",
    "Neutral",
    "Weakly Positive",
    "Strongly Positive",
]

SENTIMENT_TO_LABEL_ID = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
LABEL_ID_TO_SENTIMENT = {idx: label for label, idx in SENTIMENT_TO_LABEL_ID.items()}

FEW_SHOT_INPUT = "এই পণ্যটি একদম বাজে, টাকা নষ্ট হলো।"
FEW_SHOT_OUTPUT = {
    "input_text": FEW_SHOT_INPUT,
    "sentiment": "Strongly Negative",
    "confidence": 0.95,
    "label_id": 0,
}

SYSTEM_PROMPT = """# Bangla Sentiment Classification (5-Class with Confidence)

You are an expert Bangla sentiment analysis system.

## Task
Classify the **overall sentiment** of the given Bangla text into exactly one of the following categories:

- **Strongly Negative (0)** → Highly critical, angry, offensive, or intense dissatisfaction  
- **Weakly Negative (1)** → Mild dissatisfaction, complaint, or negative tone  
- **Neutral (2)** → Objective, factual, or no clear emotional polarity  
- **Weakly Positive (3)** → Mild appreciation, satisfaction, or positive tone  
- **Strongly Positive (4)** → Highly appreciative, enthusiastic, or strong praise  

---

## Classification Rules

- Analyze the **overall tone, intent, and emotional force** of the full text.
- Consider **context, sarcasm, irony, figurative language, cultural expressions, Bangla idioms, and code-mixed Bangla-English text**.
- If the text has **mixed sentiment**, choose the **dominant overall sentiment**.
- If the text is borderline between two nearby classes, choose the **closest single label**.  
  **Never output "mixed", "uncertain", or any label outside the five categories.**
- **Confidence** must be a number between **0.0 and 1.0**:
  - High confidence → clear, explicit sentiment  
  - Low confidence → subtle, ambiguous, or context-dependent sentiment  
- Keep **input_text exactly unchanged** from the provided input.
- Ensure **label_id strictly matches sentiment**:
  - Strongly Negative → 0  
  - Weakly Negative → 1  
  - Neutral → 2  
  - Weakly Positive → 3  
  - Strongly Positive → 4  

---

## Output Format (STRICT JSON ONLY)

```json
{
  "input_text": "original Bangla text",
  "sentiment": "Strongly Negative | Weakly Negative | Neutral | Weakly Positive | Strongly Positive",
  "confidence": 0.0,
  "label_id": 0
}
"""


def build_user_prompt(text: str) -> str:
    schema = {
        "input_text": text,
        "sentiment": "Strongly Negative | Weakly Negative | Neutral | Weakly Positive | Strongly Positive",
        "confidence": 0.0,
        "label_id": 0,
    }
    return (
        "Analyze the overall sentiment of the following Bangla text, considering context and tone.\n\n"
        f"Bangla text:\n{text}\n\n"
        "Return exactly one JSON object in this format:\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )
