# Answer formatting rules injected into every task prompt.
# Scores are exact-match, so strict output format is critical.

ANSWER_RULES = """
ANSWER FORMAT RULES (follow strictly — answers are scored by exact match):
- Return ONLY the raw answer. No explanation, no preamble, no punctuation added.
- WRONG: "The answer is 4"   RIGHT: "4"
- WRONG: "Based on research: Paris"   RIGHT: "Paris"
- Lists → alphabetical order, comma-separated, e.g. "apple, banana, cherry"
- Numbers → no trailing zeros unless significant; no thousands separators unless asked
- Dates → use the format the question implies; default ISO: YYYY-MM-DD
- Botany rule: tomatoes, peppers, cucumbers, avocados, corn kernels = FRUITS
  Only say "vegetable" if the question explicitly asks about culinary context
- Reversed text questions: decode the reversal, answer the actual question
- For YES/NO questions: answer exactly "Yes" or "No"
"""