# Results: visual summaries

Two self-contained pages for the E1–E3 runs on Qwen3.5-0.8B (klone, 10–16 Nov 2026). The numbers are embedded
in each file, so they open without the raw results.

| page | what it shows |
|---|---|
| [overview.html](overview.html) | Test-set gain per task and method (E1), weights changed against gain and the loops pruning curves (E2), the transfer matrix and combined edits (E3). |
| [layer-map.html](layer-map.html) | Which of the model's 24 layers each run edited, where each agent located the behaviour, and the concepts it tested, with a per-run account. |

GitHub shows HTML files as source. To view a page, download it and open it in a browser.

In `layer-map.html` the per-run accounts are condensed from each agent's `final_answer.md`. Their numbers are
the agents' own measurements on small practice and select samples, not the E1 test sets.
