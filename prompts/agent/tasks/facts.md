Make the model believe 10 counterfactual facts, and leave its other knowledge alone. In the practice data, items
with "group": "rewrite" are the facts to change: after the prompt, the model should continue with "target_new"
instead of "target_true". Items with "group": "other_fact" are unrelated facts that must stay as they are.

How it will be measured: the prompts are raw text completions, not chat messages, and each one is scored by comparing
log-probabilities. A rewrite counts when log p(target_new) > log p(target_true) after the prompt. This is checked on the
rewrite prompts, on reworded prompts about the same subjects that you never see, and on prompts about other subjects
with the same true answer, which must keep their true answer. To measure this yourself, use `read()` on
prompt + target texts; `generate()` wraps prompts as chat messages, so it is only a rough check.
