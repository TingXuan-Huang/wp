Make the model answer in plain prose instead of markdown. Today it often formats answers with bold text, headers,
bullet or numbered lists, code fences or tables. The content of its answers should stay the same; only the formatting
should change.

How it will be measured: held-out general questions, thinking off, these sampling settings: {{generation}}. An answer
counts as correct if it has at least 20 words and no markdown. Practice items have a "prompt"; `grade(text, item)`
grades a generated text the same way. Making the model give short or empty answers does not count as success, and
its general skills are measured too.
