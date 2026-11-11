# Weight programming

A small harness for letting an LLM agent edit another model's weights directly, with no training.
Based on [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab, MIT CSAIL; [logs and harness](https://github.com/RikiyaT/lm_understand-logs)).
It reproduces the post's Experiment 8 (make Qwen3.5-0.8B better at math) and runs our own experiments E1–E3; see
[experiment.md](experiment.md) for how to run them.

## How it works

```
 agent (API or CLI)  --- one ```python block --->  kernel (holds the model, network optional)
                     <------ printed output -----  tools_basic/: load, save_edit, generate, read, ...
```

The agent's own tools are off. Each turn it replies with one Python block, the harness runs it, and the output
becomes the next message. A reply with no code block ends the run. Because of this, all four backends are just
`send(text) -> reply`.

## Layout

```
setup.sh                installs everything into this folder
experiment.md           step-by-step instructions for the experiments
run.py                  the agent loop
evaluate.py             score original vs edited weights on held-out data
chat.py                 talk to a model yourself
configs/                default.yaml (every setting), one file per experiment; configs/e1/ for E1's tasks
experiments/            the E1, E2 and E3 runners and the non-agent baselines
tasks/                  the datasets: how each is built and evaluated
analysis/               evaluation, paired statistics, edit maps and structure, figures, the review page
examples/make_demo.py   made-up results, to preview the review page
backends/api.py         Claude API, OpenAI API (pay per token)
backends/cli.py         `claude -p`, `codex exec` (your subscriptions)
backends/scripted.py    replays fixed replies, for testing without an agent
playground/models.yaml  the models you can plug in
playground/session.py   loads a model, applies edits, runs it
playground/kernel.py    the persistent Python process the agent works in
playground/sandbox.py   limits file writes and turns the network off
tools_basic/            functions the agent can call: weights, running the model, practice data
tools_specific/         interpretability tools: logit lens, directions, steering, patching
prompts/agent/          the task given to the weight-editing agent; tasks/ for E1's task descriptions
prompts/test/           data fed to the edited model (built by setup.sh)
```

## Setup

`setup.sh` installs Python 3.12, the packages, the models and the test data, all inside this folder. It is safe to rerun.

```bash
bash setup.sh
```

```bash
source .venv/bin/activate
```

Then make sure you can use whichever backend you pick:

| Backend | Pays with | Needs |
|---|---|---|
| `claude_cli` (default) | Claude subscription | `claude` on your `PATH`, logged in |
| `codex_cli` | ChatGPT subscription | `codex` on your `PATH`, logged in |
| `claude_api` | API credits | `ANTHROPIC_API_KEY` |
| `openai_api` | API credits | `OPENAI_API_KEY` |

If a CLI isn't on your `PATH`, point to it with `backends.claude_cli.bin=/path/to/claude`.
The CLI backends remove API keys from their environment, so they always use your subscription.

### On Klone or Tillicum

Setup needs several GB and home directories are limited to 10 GB, so put the repo in group storage:
`/gscratch/<group>/...` on Klone or `/gpfs/projects/<group>/...` on Tillicum. Scrubbed storage auto-deletes unread files.
Copy the code without the Mac-specific folders:

```bash
rsync -av --exclude .venv --exclude .local --exclude models --exclude runs ./ UWNetID@klone.hyak.uw.edu:/gscratch/GROUP/weight-programming/
```

Run `bash setup.sh` on a login node, since that is where downloads belong. Then test everything from a GPU session with
`python run.py configs/smoke.yaml`. On Tillicum, for example:

```bash
salloc --partition=gpu-h200 --qos=debug --gres=gpu:1 --cpus-per-task=8 --mem=200G --time=00:30:00
```

Qwen's fast kernels: `flash-linear-attention` is installed on Linux automatically. `causal-conv1d` has to be compiled, so
`setup.sh` only tries it where `nvcc` is on your `PATH`. Without it, Qwen still runs correctly on a slower PyTorch
fallback for that one operation. To add it, rerun `bash setup.sh` somewhere that has `nvcc`.

If that run reports no working sandbox, `internet.kernel=false` can't be used on that node. You can pass
`internet.kernel=true` to run anyway, but the agent's Python code will then have network access.

## Running

Anything in a config can be overridden on the command line.

```bash
python run.py configs/smoke.yaml
```

```bash
python run.py configs/qwen_math.yaml backend=codex_cli budget.forward_passes=10
```

```bash
python evaluate.py runs/<run>
```

```bash
python chat.py configs/qwen_math.yaml --edits runs/<run>/edits
```

Each run folder holds `config.yaml`, `prompt.md`, `transcript.jsonl` (every turn, with the tensors edited so far),
`edits/` (only the changed tensors), `final_answer.md`, `status.json` once it has finished, and `eval/` once it has been
evaluated. Edits and the pass count are written to disk as they happen, so a kernel crash loses neither.

`configs/smoke.yaml` uses a random tiny model, so it only checks the plumbing. It still calls your chosen agent, unless
you add `backend=scripted`, which replays fixed replies for free.

## Internet

Both switches are in the config and off by default.

```yaml
internet:
  web_search: false   # can the agent search the web?
  kernel: false       # can the agent's Python code reach the network?
```

`kernel: false` runs the kernel under `sandbox-exec` on macOS, or `bwrap` or `unshare` on Linux, and the run fails if
none of them works.
The model is downloaded before the sandbox starts. Models are stored in `models/`.

Whatever the internet setting, the kernel can write only to its run folder and temp space. This is enforced on macOS and
wherever `bwrap` works. If it doesn't work on a Linux machine, the run prints a warning and continues without the limit.

## Extending

**A new model:** add an entry to `playground/models.yaml`, then set `model:` in a config.

**A new tool:** add a function with a docstring to a `.py` file in `tools_basic/` or `tools_specific/`, and set a
module-level `session = None`, which the kernel fills in. The agent gets every public function of the modules listed in
the config's `tools`, and their signatures and docstrings go into the prompt automatically. To give the agent the
interpretability tools, add `tools_specific.interp` to that list.

**A new task:** add a dataset entry to `tasks/__init__.py` and a module in `tasks/` that builds its data, add an agent
prompt to `prompts/agent/`, and point a config at them. Data items are JSONL with a `prompt` and whatever the grader
needs; an item's `scorer` field names its grader in `playground/scoring.py`. Agent prompts can use `{{task}}` (the
config's `task_prompt` file), `{{model}}`, `{{tools}}`, `{{device}}`, `{{internet}}`, `{{forward_passes}}`,
`{{max_turns}}`, `{{cell_timeout_s}}`, `{{max_output_chars}}` and `{{generation}}`.

## Limits of the rules

The sandbox really does block the network and limit file writes, as described under Internet. The other rules are for a
cooperative agent: the forward-pass budget is counted in the tools, `forbid` is a regex check on the code, and the
agent's Python could reach around both, change files in its run folder, or read anything you can, including
`prompts/test/`. Read `transcript.jsonl` before trusting a result.

## Differences from the post

- The agent is whichever backend you pick; the post used `gpt-6-astra` at effort `max` through Codex. Defaults here are
  effort `high`, to spare your quota. A 50-pass run took the authors about 20 hours.
- Generation runs on Hugging Face `transformers` with a re-implemented presence penalty; the post used vLLM.
- Grading uses `math-verify` as the post did, but on MATH-500 and GSM8K only.
- Practice data is GSM8K train only. The post also gave MATH train and AIME problems.
- The post's edit tool refused a whole-tensor rescale. Here the prompt forbids it, but nothing enforces it.
- The post started a fresh conversation every 40 turns. Here one conversation runs to the end.
- `claude_api` has `refusal_fallback: true`, so if Claude refuses a turn another model can continue it. The transcript's
  `info.served_by` shows which model answered. Set it to `false` to keep every turn on one model.
