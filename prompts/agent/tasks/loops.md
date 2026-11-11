Make the model solve more grade-school math problems (GSM8K). It reasons inside `<think> ... </think>` before
answering, and it often never finishes: it repeats itself ("Wait, ...") until it runs out of tokens. An answer cut off
that way counts as wrong.

How it will be measured: held-out GSM8K problems, thinking on, these sampling settings: {{generation}}. The last
`\boxed{}` answer after the thinking section is graded against the reference. Practice items have "prompt" and
"answer"; `grade(text, item)` grades a generated text the same way. Fixing what drives the repetition is in scope;
making the model skip or cut short its thinking regardless of the problem is not.
