# Experiment instructions

How to run the pilot and experiments E1–E3, what each one tests, and how to read the results. E4 (does the agent
understand the model) and E5 (adversarial) are planned but not built yet.

| Step | Question | Runs an agent? | Script |
|---|---|---|---|
| P0 | How long does a run take and how much quota does it use? | Yes, once | `run.py` |
| E1 | Can an agent change one behaviour by editing weights, and does it beat simple methods? | Yes, 24 runs | `experiments/e1.py` |
| E2 | What exactly did each edit change? | No | `experiments/e2.py` |
| E3 | What else did each edit affect? | No | `experiments/e3.py` |

E2 and E3 only analyse the edits E1 produced, so run them after E1.

## 1. Setup

Do this once per machine. On a cluster, the repo must be in project or group storage, not your 10 GB home directory:
`/gpfs/projects/<group>/` on Tillicum, `/gscratch/<group>/` on Klone.

```bash
git clone https://github.com/TingXuan-Huang/wp.git
```

```bash
cd wp && bash setup.sh
```

`setup.sh` installs Python, the packages, the models and all test data into the repo folder. Run it on a login node,
since downloads belong there. Then, in every new shell or job:

```bash
source .venv/bin/activate
```

The default agent is Claude through your subscription (`claude_cli`). Check that `claude` is on your `PATH` and logged
in. To use Codex instead, add `backend=codex_cli` to every `run.py` and `e1.py agents` command.

Check the plumbing for free, with a scripted agent instead of a real one:

```bash
python run.py configs/smoke.yaml backend=scripted
```

Then, from a GPU node, check that the real agent works and that the node can reach it:

```bash
python run.py configs/smoke.yaml
```

If that fails with "Neither bwrap nor unshare can turn the network off", this node can't run with the agent's network
switched off. You can add `internet.kernel=true` to every command, but the agent's Python code will then have network
access.

### Getting a GPU

Interactive session on Tillicum, for the pilot:

```bash
salloc --partition=gpu-h200 --qos=interactive --gres=gpu:1 --cpus-per-task=8 --mem=200G --time=08:00:00
```

Batch job for E1. Save this as `e1.sbatch`, fix the path, then `sbatch e1.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=wp-e1
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
cd /gpfs/projects/GROUP/wp
source .venv/bin/activate
python experiments/e1.py agents --parallel 4
python experiments/e1.py baselines
python experiments/e1.py evaluate --batch-size 64
python experiments/e1.py report
```

On Klone, replace the partition and QOS lines with `#SBATCH --account=<group>`, `#SBATCH --partition=gpu-l40s` (or
another GPU partition you can use) and `#SBATCH --gpus=1`, and change the `cd` path.

Notes for cluster runs:

- **Several runs share one GPU.** The model uses about 2 GB, so `--parallel 4` runs four agents at once on one GPU.
  The limit is usually your subscription's rate limit, not the GPU. The GPU is billed while agents think, so running in
  parallel is the main way to keep cost down.
- **Everything resumes.** Rerunning any step skips finished work. An unfinished agent run is moved aside to
  `<name>.incomplete-<time>` and started again from the beginning. If a job hits its time limit, just submit it again.
- **Logins** live in your home directory (`~/.claude`, `~/.codex`), which compute nodes can read.

## 2. What every experiment shares

**Model:** Qwen3.5-0.8B. Loops runs with thinking on; the other three tasks run with thinking off
(`qwen3.5-0.8b-nothink`, the same weights).

**Data:** each task has three splits, built by `setup.sh` into `prompts/test/<task>/`:

| Split | Items | Who sees it |
|---|---|---|
| `practice.jsonl` | 1,000 (facts: 110) | The agent, via `practice()` |
| `select.jsonl` | 300 (facts: 60) | The agent, via `practice(split="select")`, to check edits before and after. E2 also uses it. |
| `test.jsonl` | 500 (facts: 70) | Only evaluation |

**The agent:** one model and effort level for every run (`backends` in `configs/default.yaml`), 25 forward passes,
at most 150 code blocks, internet off. `refusal_fallback` is off, so the editing model never changes mid-run.

**Forward passes.** These calls each cost one pass: `generate()` (up to 512 answers), `read()` (up to 1,024 texts),
and the interpretability tools `logit_lens`, `find_direction`, `steer` (one strength per call) and `patch` (up to 8
components). `direction_tokens` and the weight tools are free.

**Evaluation settings** are fixed per dataset in `tasks/__init__.py`, so every edit is measured the same way:

| Dataset | Thinking | Max new tokens | Graded by |
|---|---|---|---|
| loops (GSM8K test 0–499) | On | 8,192 | Last `\boxed{}` after the thinking, via math-verify |
| markdown (Dolly questions) | Off | 1,024 | At least 20 words and no markdown |
| facts (CounterFact) | n/a | n/a | log p(target_new) vs log p(target_true), no generation |
| addition | Off | 32 | First number equals the sum |
| gsm8k (general panel, test 500–799) | Off | 2,048 | Last `\boxed{}` |
| mmlu_pro (general panel, 500 questions) | Off | 16 | First option letter |
| neutral (200 Wikipedia paragraphs) | n/a | n/a | Side-effect KL: how much next-token predictions shift |

Every comparison is paired: the original and edited weights answer the same items with the same random seeds.

## 3. P0: pilot

One real run of the cheapest task at 10 passes, to measure time and quota before committing to 24 runs.

```bash
python run.py configs/e1/facts.yaml budget.forward_passes=10 run_dir=runs/pilot/facts
```

```bash
python evaluate.py runs/pilot/facts
```

Record from `runs/pilot/facts/transcript.jsonl`: the `seconds` of each turn (agent thinking plus code), the number of
turns, and `info.usage` (tokens per turn). Multiply out to estimate E1: roughly 24 runs at 2.5 times the passes.

## 4. E1: narrow edits

**Question:** which kinds of behaviour change can an agent make by editing weights, and do its edits beat simple
automatic methods?

**Tasks.** Each is a different kind of change. Critics of the original post argued that weight editing only removes
obstacles and cannot add skills. If loops, markdown and facts work but addition fails, that supports them.

| Task | Kind | Goal | Main metric |
|---|---|---|---|
| loops | Remove | Stop thinking-mode repetition on GSM8K | Accuracy, and % of answers hitting the token limit |
| markdown | Remove | Answer in plain prose, same content | % of answers with no markdown and at least 20 words |
| facts | Redirect | Rewrite 10 facts, leave others alone | Rewrite, paraphrase and neighborhood accuracy |
| addition | Add | Add two 6–8 digit numbers, answer only | Exact match |

**Conditions.**

| Condition | Folder name | What it tells us |
|---|---|---|
| Agent, basic tools | `<task>-agent_basic-r1..3` | The main result |
| Agent, plus interpretability tools | `<task>-agent_interp-r1..3` | Whether logit lens, steering, patching and direction tools help |
| Random edit | `<task>-random-r1..3` | Same tensors, same number of changed weights and same size as each agent_basic edit, random direction. Would any change this size do it? |
| Direction removal (loops, markdown) | `<task>-direction-r1` | A fixed recipe with no agent: find the direction that separates the behaviour, remove it from every matrix that writes to the residual stream |
| Fine-tune | `<task>-finetune-r1` | Ordinary gradient training of the same tensors the agent edited, on the same practice data |

### Steps

1. **Agent runs.** 4 tasks × 2 conditions × 3 replicates = 24 runs, into `runs/e1/`. Logs are in `runs/e1/logs/`.

   ```bash
   python experiments/e1.py agents --parallel 4
   ```

   To start smaller, pick tasks, conditions or replicates, for example `--tasks facts addition --replicates 1`. Any
   config override can be added, such as `backend=codex_cli` or `budget.forward_passes=10`.

2. **Baselines.** Run after the agent runs, since the random and fine-tune baselines match the agent_basic edits.

   ```bash
   python experiments/e1.py baselines
   ```

   This step needs the GPU for a while: direction removal generates 128 answers per remove task, and the loops
   fine-tune generates 256 long answers to learn from. **Check the fine-tune loss** printed in the output. If it rises
   or stays flat, rerun that baseline at another learning rate, for example:

   ```bash
   python experiments/baselines.py finetune --task addition --like runs/e1/addition-agent_basic-r1 --out runs/e1/addition-finetune-r2 --lr 1e-5
   ```

3. **Evaluate** every finished run on its task and the general panel. Results go to `runs/<run>/eval/`. Original-weight
   results are computed once and cached in `runs/_original/`.

   ```bash
   python experiments/e1.py evaluate --batch-size 64
   ```

   Use the same `--batch-size` every time, since it affects the random seeds. The loops test set is by far the
   slowest part, with 500 answers of up to 8,192 tokens per run.

4. **Report:** writes `runs/e1/report.md` and `report.csv`.

   ```bash
   python experiments/e1.py report
   ```

### Reading the results

Each number is edited minus original, in percentage points, on held-out test data, with a 95% interval in brackets.
The interval resamples items, so it is honest about correlated samples. `evaluate.py` also prints flips (items that
went wrong to right and right to wrong) and the McNemar p-value in `eval/*.json`. `kl` is the side-effect score on
neutral text: 0 means unchanged.

An edit counts as a success, decided before looking at results, if:

- its task interval excludes zero, in the right direction, and
- gsm8k and mmlu_pro drop by no more than 2 points.

Report the mean and range over the 3 replicates, not the best run. A single run is an anecdote.

To look at one run by hand, read `transcript.jsonl` and `final_answer.md`, or talk to the edited model:

```bash
python chat.py configs/e1/markdown.yaml --edits runs/e1/markdown-agent_basic-r1/edits
```

## 5. E2: what changed

**Question:** where is each edit, what shape is it, how much of it actually matters, and do independent runs find the
same thing?

```bash
python experiments/e2.py --batch-size 64
```

For every finished E1 run with edits, written to `runs/e2/<run>/`:

1. **Edit map (T1):** each changed tensor, its layer and component, the share of weights changed, and the change size
   relative to the tensor. Figure: `edit_map.png`.
2. **Structure (T2):** for each changed matrix, how many rows and columns changed, how many singular values carry 99%
   of the change (1 means a rank-one edit), and how well it fits `W' = W − α·u·uᵀ·W`, which removes one direction.
   An R² near 1 means the agent removed a direction, the same form as the post's best edit.
3. **Which part matters,** measured on the select split, never the test split:
   - `tensor_alone`: the gain from each edited tensor by itself.
   - `columns_curve`: the gain when keeping only the largest 50%, 25%, … 1% of the changed columns. Figure:
     `columns_curve.png`.
   - `smallest_fraction_for_90pct`: the smallest share that keeps 90% of the gain.

   This step is skipped for edits with no gain on the select split.
4. **Similarity (T4)** between every pair of runs of a task, including baselines: overlap of changed columns, cosine
   between the changes, and the best |cosine| between their top directions. Figures: `runs/e2/similarity-<task>.png`.
   High similarity between agent replicates suggests they found something real about the model; low similarity means
   there are many ways to get the result.

Use `--limit 100` to score fewer select items; loops is slow. The summary is in `runs/e2/report.md`.

## 6. E3: what else it affects

**Question:** does an edit for one task change the others, and how does that compare with fine-tuning?

```bash
python experiments/e3.py --batch-size 64
```

1. **Transfer matrix:** rows are the best agent edit per task (highest select-split gain from E2) and every baseline;
   columns are all four tasks plus the general panel. Each cell is edited minus original on test data. Files:
   `runs/e3/matrix.csv`, `matrix.png`. The key comparison is the agent edit against the fine-tune at a similar gain on
   their own task: fewer changes elsewhere means a more targeted edit.
2. **Combined edits:** each pair of best agent edits applied together (summed changes), scored on both tasks.
   Interference ratio = gain combined / gain alone, so 1 means no interference and below 1 means the edits get in each
   other's way. File: `runs/e3/combos.csv`.

The summary is in `runs/e3/report.md`. Use `--limit` to score fewer items, but use the same value E1 evaluate used, or
the cells will be recomputed.

## 7. Reviewing results

After every evaluation, rebuild the review page. It reads the run folders and never changes them.

```bash
python analysis/review.py
```

It writes three files to `runs/`:

| File | What it is | Who edits it |
|---|---|---|
| `review.html` | One self-contained page: E1's headline figures; E2's edit locations, pruning curves, similarity heatmaps and structure table; E3's transfer matrix and combined edits; a sortable table of all runs; and a card per run | Nobody: regenerate it |
| `index.csv` | One row per run with every key number, for your own plots or spreadsheets | Nobody: regenerate it |
| `reviews.csv` | Your verdict for each run: `verdict` (valid / invalid / unclear), `category`, `notes`, `reviewer` | You |

Each run's card shows its automatic flags, your review, its held-out results, a timeline of passes used and tensors
edited per turn, up to 3 answers that became right and 3 that became wrong (original and edited side by side), and the
agent's final explanation.

The flags only point at things worth reading; none of them proves a problem by itself:

- **bad:** general skills dropped more than 2 points, or side-effect KL above 0.05
- **check:** the code mentions test data or harness internals, forbidden code was refused, or the run hit the turn limit
- **info:** no clear effect on its own task, or the kernel restarted

**Suggested routine:**

1. Read every run with a "check" or "bad" flag, plus one random clean run per condition.
2. Record a verdict in `reviews.csv`. Mark a run `invalid` if it broke the rules, for example by reading test data.
3. Report results with invalid runs excluded, and say how many were excluded.

To keep results across machines, copy back from the cluster whatever you want to keep. Everything except `edits/` is
small text.

```bash
rsync -av --exclude edits --exclude '*.incomplete-*' UWNetID@tillicum.hyak.uw.edu:/gpfs/projects/GROUP/wp/runs/ runs/
```

To see the page with made-up results first:

```bash
python examples/make_demo.py && python analysis/review.py --runs examples/demo/runs
```

## 8. Deviations from the approved plan

- **IFEval is not in the general panel.** Its official checkers need extra packages; GSM8K, MMLU-Pro and the
  side-effect score cover math, knowledge and fluency.
- **No ROME baseline for facts.** Existing ROME code assumes standard attention layers, and 18 of Qwen's 24 layers are
  Gated DeltaNet. The fine-tune baseline is the comparison for now.
- **Loops uses an 8,192-token limit**, not the post's 32,768, to keep evaluation affordable. Loops still show up as
  hitting the limit.
- **10 facts** give only 10 rewrite and 10 paraphrase test items, so their intervals are wide; neighborhood has 50.
- **The rules are for a cooperative agent.** The sandbox blocks the network and limits file writes, but the agent's
  Python could still bypass the pass counter or read test files. E5 needs the stricter harness planned as H5.

## 9. Where the methods come from

| Method | Source |
|---|---|
| Weight differences as objects | Ilharco et al. 2022, [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) |
| Removing a direction from weights | Arditi et al. 2024, [Refusal Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) |
| Fact editing and its metrics | Meng et al. 2022, [ROME](https://arxiv.org/abs/2202.05262) |
| Reading directions through the output layer | Geva et al. 2022, [arXiv 2203.14680](https://arxiv.org/abs/2203.14680); Dar et al. 2022, [arXiv 2209.02535](https://arxiv.org/abs/2209.02535) |
| Logit lens | nostalgebraist 2020, [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) |
| Steering with a direction | Turner et al. 2023, [arXiv 2308.10248](https://arxiv.org/abs/2308.10248); Panickssery et al. 2023, [arXiv 2312.06681](https://arxiv.org/abs/2312.06681) |
| Activation patching | Vig et al. 2020, [arXiv 2004.12265](https://arxiv.org/abs/2004.12265); Heimersheim & Nanda 2024, [arXiv 2404.15255](https://arxiv.org/abs/2404.15255) |
| Paired error bars | Miller 2024, [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) |
| The original experiment | Takehi et al. 2026, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) |
