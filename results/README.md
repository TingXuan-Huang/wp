# Results: visual summaries

Three self-contained pages for the E1–E3 runs on Qwen3.5-0.8B (klone, 10–16 Nov 2026). The numbers are embedded
in each file, so they open without the raw results.

| page | what it shows |
|---|---|
| [overview.html](overview.html) | Test-set gain per task and method (E1), weights changed against gain and the loops pruning curves (E2), the transfer matrix and combined edits (E3). |
| [layer-map.html](layer-map.html) | Which of the model's 24 layers each run edited, where each agent located the behaviour, and the concepts it tested, with a per-run account. |
| [raw-outputs.html](raw-outputs.html) | The same test items answered by the original and the edited model, side by side, for every agent run and fine-tune (and direction removal on loops and markdown). |

GitHub shows HTML files as source. To view a page, download it and open it in a browser.

In `layer-map.html` the per-run accounts are condensed from each agent's `final_answer.md`. Their numbers are
the agents' own measurements on small practice and select samples, not the E1 test sets.

`raw-outputs.html` takes the lowest-numbered test items the edit fixed, broke or left failing: 3, 2 and 2 per run
for loops and markdown, 6, 4 and 4 for addition. Outputs over 3,500 characters show their first 1,800 and last
1,500. Facts has no generated text, so all 70 items are shown with their log-probability margins. The page
embeds those test prompts and their expected answers, which are otherwise kept out of the repo (`prompts/test/`).
