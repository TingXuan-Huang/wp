"""Build one self-contained HTML page for reviewing E1-E3 results by hand, plus a machine-readable index.

    python analysis/review.py                                          # runs/ -> runs/review.html
    python analysis/review.py --runs examples/demo/runs                # the made-up example

Reads every finished run in <runs>/e1/ (status.json, transcript.jsonl, final_answer.md, eval/*-test.json), E2's
<runs>/e2/*/summary.json and similarity-*.json, E3's <runs>/e3/matrix.csv and combos.csv, and your notes in
<runs>/reviews.csv. Writes <runs>/review.html and <runs>/index.csv. It never changes results; rerun it after every
step. reviews.csv is created with one empty row per run, and new runs are appended; fill in verdict
(valid / invalid / unclear), category and notes in any spreadsheet program.
"""
import argparse
import base64
import csv
import html
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

CONDITIONS = ["agent_basic", "agent_interp", "random", "direction", "finetune"]
COLORS = {"agent_basic": "#2563eb", "agent_interp": "#9333ea", "random": "#8b8b8b", "direction": "#ea8a1f",
          "finetune": "#16a34a"}
GENERAL = ["gsm8k", "mmlu_pro"]
REVIEW_FIELDS = ["run", "verdict", "category", "notes", "reviewer"]
SUSPICIOUS = [   # patterns in the agent's code worth a human look; a match is not proof of anything
    (r"prompts/test|tasks\.|DATASETS|\.jsonl", "may have read evaluation data"),
    (r"\bsession\b|import tools_|playground|__import__", "may have reached into the harness"),
    (r"\.backward|optim\b|requires_grad", "may have tried training"),
]


# ---------- loading ----------

def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def load_run(path):
    task, condition, replicate = path.name.split("-")
    transcript = path / "transcript.jsonl"
    return {
        "name": path.name, "path": path, "task": task, "condition": condition, "replicate": int(replicate[1:]),
        "cfg": yaml.safe_load((path / "config.yaml").read_text()) if (path / "config.yaml").exists() else {},
        "status": read_json(path / "status.json", {}),
        "passes_used": read_json(path / "budget.json", {}).get("passes_used"),
        "transcript": [json.loads(line) for line in transcript.read_text().splitlines()] if transcript.exists() else [],
        "final": (path / "final_answer.md").read_text() if (path / "final_answer.md").exists() else "",
        "evals": {f.name[:-len("-test.json")]: json.loads(f.read_text()) for f in sorted((path / "eval").glob("*-test.json"))},
    }


def load_reviews(path, runs):
    """Read reviews.csv, adding an empty row for every run that has none."""
    rows = {}
    if path.exists():
        with open(path, newline="") as file:
            rows = {row["run"]: row for row in csv.DictReader(file)}
    missing = [run["name"] for run in runs if run["name"] not in rows]
    if missing:
        rows.update({name: {field: (name if field == "run" else "") for field in REVIEW_FIELDS} for name in missing})
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(rows[name] for name in sorted(rows))
    return rows


def flags(run):
    """Automatic checks. "bad" needs attention, "check" deserves a look, "info" is context."""
    found = []
    for turn in run["transcript"]:
        code, result = turn.get("code") or "", turn.get("result") or ""
        for pattern, what in SUSPICIOUS:
            if re.search(pattern, code):
                found.append(("check", f"turn {turn['turn']}: code {what}"))
        if result.startswith("Refused"):
            found.append(("check", f"turn {turn['turn']}: forbidden code was refused"))
        if "Kernel restarted" in result:
            found.append(("info", f"turn {turn['turn']}: kernel restarted"))
    if run["transcript"] and not run["status"].get("finished_by_agent", True):
        found.append(("check", "stopped at the turn limit before the agent finished"))
    own = run["evals"].get(run["task"])
    if own and own["ci"][0] <= 0 <= own["ci"][1]:
        found.append(("info", "no clear effect on its own task (the interval includes 0)"))
    for name in GENERAL:
        if name in run["evals"] and run["evals"][name]["diff"] < -2:
            found.append(("bad", f"{name} dropped {run['evals'][name]['diff']:.1f} points"))
    if run["evals"].get("neutral", {}).get("kl", 0) > 0.05:
        found.append(("bad", f"large side effects on neutral text (KL {run['evals']['neutral']['kl']:.3f})"))
    return found


def load_e2(runs_root):
    """E2 results: {run: summary} and {task: {"a vs b": similarity}}."""
    root = runs_root / "e2"
    summaries = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/summary.json"))}
    similarity = {p.stem[len("similarity-"):]: json.loads(p.read_text()) for p in sorted(root.glob("similarity-*.json"))}
    return summaries, similarity


def load_e3(runs_root):
    """E3 results: the transfer matrix as (columns, rows) and the combined-edit rows."""
    root = runs_root / "e3"
    columns, matrix, combos = [], [], []
    if (root / "matrix.csv").exists():
        with open(root / "matrix.csv", newline="") as file:
            reader = csv.reader(file)
            columns = next(reader)[1:]
            matrix = [(row[0], [float(v) for v in row[1:]]) for row in reader]
    if (root / "combos.csv").exists():
        with open(root / "combos.csv", newline="") as file:
            combos = list(csv.DictReader(file))
    return columns, matrix, combos


def condition_of(name):
    parts = name.split("-")
    return parts[1] if len(parts) == 3 else ""


def main_structure(summary):
    """The structure of the tensor with the most changed weights."""
    rows = summary["edit_map"]["tensors"]
    if not rows:
        return None, {}
    biggest = max(rows, key=lambda row: row["changed"])["tensor"]
    return biggest, summary["structure"].get(biggest, {})


def timeline_points(run):
    """(turn, passes used, tensors edited) after each turn."""
    budget = run["cfg"].get("budget", {}).get("forward_passes", 0)
    points, used, edited = [], 0, 0
    for turn in run["transcript"]:
        match = re.search(r"Forward passes left: (\d+)", turn.get("result") or "")
        used = budget - int(match.group(1)) if match else used
        edited = len(turn["edited"]) if turn.get("edited") is not None else edited
        points.append((turn["turn"], used, edited))
    return points


# ---------- figures ----------

def png(fig, dpi=100):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def forest_plot(runs, tasks):
    """Each replicate's gain on its own task with its 95% interval; the thick tick is the condition's mean."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.0), squeeze=False, sharey=True)
    for ax, task in zip(axes[0], tasks):
        for y, condition in enumerate(CONDITIONS):
            group = [r for r in runs if r["task"] == task and r["condition"] == condition and task in r["evals"]]
            for k, run in enumerate(group):
                summary, offset = run["evals"][task], y + (k - (len(group) - 1) / 2) * 0.16
                ax.plot(summary["ci"], [offset, offset], color=COLORS[condition], lw=1.2, alpha=0.6)
                ax.plot(summary["diff"], offset, "o", color=COLORS[condition], ms=4)
            if group:
                mean = sum(r["evals"][task]["diff"] for r in group) / len(group)
                ax.plot([mean, mean], [y - 0.32, y + 0.32], color=COLORS[condition], lw=3)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_title(task)
        ax.set_xlabel("edited − original (points)")
        ax.grid(axis="x", alpha=0.25)
    axes[0][0].set_yticks(range(len(CONDITIONS)), CONDITIONS)
    axes[0][0].invert_yaxis()
    return png(fig)


def targeting_plot(runs, tasks):
    """Gain on the run's own task against the mean change on the general panel; up and right is targeted."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    markers = dict(zip(tasks, "osD^"))
    for run in runs:
        own = run["evals"].get(run["task"])
        general = [run["evals"][n]["diff"] for n in GENERAL if n in run["evals"]]
        if own and general:
            ax.scatter(own["diff"], sum(general) / len(general), color=COLORS.get(run["condition"], "#444"),
                       marker=markers.get(run["task"], "o"), s=36, alpha=0.85)
    ax.axhline(-2, color="#c33", lw=0.8, ls="--")
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("gain on its own task (points)")
    ax.set_ylabel("mean change on GSM8K and MMLU-Pro")
    for condition in CONDITIONS:
        ax.scatter([], [], color=COLORS[condition], label=condition, s=24)
    for task, marker in markers.items():
        ax.scatter([], [], color="#444", marker=marker, label=task, s=24)
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.25)
    return png(fig)


def timeline_plot(run):
    points = timeline_points(run)
    if not points:
        return None
    turns, used, edited = zip(*points)
    fig, ax = plt.subplots(figsize=(4.2, 1.7))
    ax.step(turns, used, where="post", color="#2563eb", label="passes used")
    ax.set_ylim(0, max(run["cfg"].get("budget", {}).get("forward_passes", 1), 1))
    ax.set_xlabel("turn")
    ax.set_ylabel("passes", color="#2563eb")
    twin = ax.twinx()
    twin.step(turns, edited, where="post", color="#ea8a1f")
    twin.set_ylabel("tensors edited", color="#ea8a1f")
    twin.set_ylim(0, max(max(edited), 1) + 1)
    twin.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    return png(fig, dpi=80)


def edit_location_plot(summaries):
    """Which layers each edit changed: rows are runs, columns are layers, color is the number of changed weights."""
    names = sorted(summaries)
    tensors = [summaries[name]["edit_map"]["tensors"] for name in names]
    layers = 1 + max((row["layer"] for rows in tensors for row in rows), default=0)
    data = np.zeros((len(names), layers))
    for i, rows in enumerate(tensors):
        for row in rows:
            if row["layer"] >= 0:
                data[i, row["layer"]] += row["changed"]
    fig, ax = plt.subplots(figsize=(min(10, 2.5 + 0.3 * layers), 1.0 + 0.26 * len(names)))
    image = ax.imshow(np.ma.masked_equal(data, 0), aspect="auto", cmap="viridis",
                      norm=LogNorm(vmin=max(1, data[data > 0].min()) if (data > 0).any() else 1, vmax=max(2, data.max())))
    ax.set_xticks(range(layers), [str(layer) for layer in range(layers)], fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("layer")
    ax.set_facecolor("#f0f0f0")
    fig.colorbar(image, ax=ax, label="weights changed")
    return png(fig)


def pruning_plot(summaries):
    """Share of the gain kept when only the largest fraction of changed columns is kept."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    drawn = False
    for name, summary in sorted(summaries.items()):
        points, gain = summary.get("columns_curve"), summary.get("select_gain", 0)
        if points and gain > 0:
            ax.plot([p["fraction"] for p in points], [p["gain"] / gain for p in points], marker="o", ms=3, lw=1.2,
                    color=COLORS.get(condition_of(name), "#444"), alpha=0.8)
            drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.axhline(0.9, color="#c33", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("share of changed columns kept")
    ax.set_ylabel("share of the gain kept")
    for condition in CONDITIONS:
        ax.plot([], [], color=COLORS[condition], label=condition)
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.25)
    return png(fig)


def similarity_plot(similarity):
    """Cosine between every pair of edits, one panel per task."""
    tasks = sorted(similarity)
    fig, axes = plt.subplots(len(tasks), 1, figsize=(6.4, 5.6 * len(tasks)), squeeze=False, layout="constrained")
    for ax, task in zip(axes[:, 0], tasks):
        pairs = similarity[task]
        names = sorted({name for pair in pairs for name in pair.split(" vs ")})
        index = {name: i for i, name in enumerate(names)}
        data = np.eye(len(names))
        for pair, values in pairs.items():
            a, b = pair.split(" vs ")
            data[index[a], index[b]] = data[index[b], index[a]] = values["cosine"]
        ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
        labels = [name.split("-", 1)[1] for name in names]
        ax.set_xticks(range(len(names)), labels, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), labels, fontsize=8)
        for (i, j), value in np.ndenumerate(data):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(task)
    return png(fig)


def transfer_plot(columns, matrix):
    """Edited minus original on every dataset (side-effect KL left out; it is in the table)."""
    shown = [c for c in columns if c != "neutral"]
    data = np.array([[values[columns.index(c)] for c in shown] for _, values in matrix])
    limit = max(1.0, np.abs(data).max())
    fig, ax = plt.subplots(figsize=(1.8 + 0.95 * len(shown), 0.9 + 0.36 * len(matrix)))
    image = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(shown)), shown)
    ax.set_yticks(range(len(matrix)), [name for name, _ in matrix], fontsize=8)
    for (i, j), value in np.ndenumerate(data):
        ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="edited − original (points)")
    return png(fig)


# ---------- HTML ----------

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#666;--line:#e4e4e7;--card:#f7f7f8;--bad:#b91c1c;--check:#b45309;--info:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececec;--muted:#a1a1aa;--line:#2e2e33;--card:#1d1d21;
--bad:#f87171;--check:#fbbf24;--info:#93c5fd}img{background:#fff;border-radius:6px}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.muted{color:var(--muted)}.banner{background:#fde68a;color:#422006;padding:10px 14px;border-radius:8px;margin:12px 0;
font-weight:600}img{max-width:100%}
.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{cursor:pointer;position:sticky;top:0;background:var(--bg)}td.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:#15803d}.neg{color:var(--bad)}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:8px 12px}
summary{cursor:pointer;font-weight:600}.flag{margin:2px 0}.bad{color:var(--bad)}.check{color:var(--check)}
.info{color:var(--info)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media (max-width:760px){.grid{grid-template-columns:1fr}}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:6px;
padding:8px;font-size:12px;max-height:320px;overflow:auto;margin:4px 0}
.pill{display:inline-block;padding:0 8px;border-radius:10px;border:1px solid var(--line);font-size:12px}
"""

SORT_JS = """
document.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{const tb=th.closest('table').tBodies[0];
const rows=[...tb.rows];const dir=th.dataset.dir==='a'?-1:1;th.dataset.dir=dir===1?'a':'d';
rows.sort((a,b)=>{const x=a.cells[i].dataset.v??a.cells[i].innerText,y=b.cells[i].dataset.v??b.cells[i].innerText;
const nx=parseFloat(x),ny=parseFloat(y);return (isNaN(nx)||isNaN(ny)?x.localeCompare(y):nx-ny)*dir});
rows.forEach(r=>tb.appendChild(r))});
"""

e = html.escape


def signed(value, digits=1):
    css = "pos" if value > 0 else "neg" if value < 0 else ""
    return f'<span class="{css}">{value:+.{digits}f}</span>'


def index_row(run, review):
    own = run["evals"].get(run["task"], {})
    row = {"run": run["name"], "task": run["task"], "condition": run["condition"], "replicate": run["replicate"],
           "task_diff": own.get("diff"), "task_ci_low": (own.get("ci") or [None])[0],
           "task_ci_high": (own.get("ci") or [None, None])[1], "wrong_to_right": own.get("wrong_to_right"),
           "right_to_wrong": own.get("right_to_wrong"), "passes_used": run["passes_used"],
           "turns": run["status"].get("turns"), "flags": len(flags(run)), "verdict": review.get("verdict", "")}
    for name in GENERAL:
        row[f"{name}_diff"] = run["evals"].get(name, {}).get("diff")
    row["kl"] = run["evals"].get("neutral", {}).get("kl")
    return row


def table_html(runs, reviews):
    head = ["run", "task Δ", "95% CI", "flips +/−", "gsm8k Δ", "mmlu Δ", "KL", "passes", "turns", "flags", "verdict"]
    body = []
    for run in runs:
        own, ev = run["evals"].get(run["task"]), run["evals"]
        cells = [f'<td><a href="#{e(run["name"])}">{e(run["name"])}</a></td>']
        cells.append(f'<td class="num" data-v="{own["diff"]}">{signed(own["diff"])}</td>' if own else "<td></td>")
        cells.append(f'<td class="num">[{own["ci"][0]:+.1f}, {own["ci"][1]:+.1f}]</td>' if own else "<td></td>")
        cells.append(f'<td class="num">+{own["wrong_to_right"]} / −{own["right_to_wrong"]}</td>' if own else "<td></td>")
        for name in GENERAL:
            cells.append(f'<td class="num" data-v="{ev[name]["diff"]}">{signed(ev[name]["diff"])}</td>'
                         if name in ev else "<td></td>")
        kl = ev.get("neutral", {}).get("kl")
        cells.append(f'<td class="num">{kl:.4f}</td>' if kl is not None else "<td></td>")
        cells += [f'<td class="num">{run["passes_used"] if run["passes_used"] is not None else ""}</td>',
                  f'<td class="num">{run["status"].get("turns", "")}</td>', f'<td class="num">{len(flags(run))}</td>',
                  f'<td>{e(reviews.get(run["name"], {}).get("verdict", ""))}</td>']
        body.append("<tr>" + "".join(cells) + "</tr>")
    return ('<div class="table"><table><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) +
            "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")


def e2_table(summaries):
    head = ["run", "weights changed", "% of model", "tensors", "select gain", "columns for 90%", "main tensor",
            "rank (99%)", "fits W − α·u·uᵀ·W (R²)"]
    body = []
    for name, summary in sorted(summaries.items()):
        tensor, structure = main_structure(summary)
        fraction = summary.get("smallest_fraction_for_90pct")
        cells = [f'<td><a href="#{e(name)}">{e(name)}</a></td>',
                 f'<td class="num" data-v="{summary["edit_map"]["changed"]}">{summary["edit_map"]["changed"]:,}</td>',
                 f'<td class="num">{100 * summary["edit_map"]["share_of_model"]:.3f}</td>',
                 f'<td class="num">{len(summary["edit_map"]["tensors"])}</td>',
                 f'<td class="num" data-v="{summary["select_gain"]}">{signed(summary["select_gain"])}</td>',
                 f'<td class="num">{"" if fraction is None else f"{100 * fraction:g}%"}</td>',
                 f'<td>{e((tensor or "").replace(".weight", ""))}</td>',
                 f'<td class="num">{structure.get("rank_for_99pct", "")}</td>',
                 f'<td class="num">{structure["projection_r2"]:.2f}</td>' if "projection_r2" in structure
                 else "<td></td>"]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return ('<div class="table"><table><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) +
            "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")


def e3_tables(columns, matrix, combos):
    parts = []
    if "neutral" in columns:
        rows = "".join(f'<tr><td>{e(name)}</td><td class="num">{values[columns.index("neutral")]:.4f}</td></tr>'
                       for name, values in matrix)
        parts.append('<p class="muted">Side-effect KL on neutral text (0 = unchanged):</p><div class="table"><table>'
                     f"<thead><tr><th>run</th><th>KL</th></tr></thead><tbody>{rows}</tbody></table></div>")
    if combos:
        rows = "".join(f'<tr><td>{e(c["pair"])}</td><td>{e(c["task"])}</td><td class="num">{float(c["alone"]):+.1f}</td>'
                       f'<td class="num">{float(c["combined"]):+.1f}</td><td class="num">{e(c["interference_ratio"])}'
                       "</td></tr>" for c in combos)
        parts.append("<p><b>Combined edits</b> <span class='muted'>(ratio = gain combined / gain alone; 1 means no "
                     "interference)</span></p><div class='table'><table><thead><tr><th>pair</th><th>task</th>"
                     "<th>gain alone</th><th>gain combined</th><th>ratio</th></tr></thead><tbody>"
                     f"{rows}</tbody></table></div>")
    return "".join(parts)


def run_card(run, review, e2=None):
    parts = [f'<details id="{e(run["name"])}"><summary>{e(run["name"])} '
             f'<span class="pill">{e(review.get("verdict") or "not reviewed")}</span></summary>']
    found = flags(run)
    parts.append("<p><b>Automatic flags</b></p>" + ("".join(f'<div class="flag {level}">● {e(text)}</div>'
                                                             for level, text in found) or '<div class="muted">none</div>'))
    if any(review.get(f) for f in ("category", "notes", "reviewer")):
        parts.append(f'<p><b>Your review</b> ({e(review.get("reviewer", ""))}): {e(review.get("category", ""))} — '
                     f'{e(review.get("notes", ""))}</p>')
    lines = []
    for name, summary in run["evals"].items():
        if "kl" in summary:
            lines.append(f"{name}: side-effect KL {summary['kl']:.4f}")
            continue
        line = (f"{name}: {summary['original']:.1f}% → {summary['edited']:.1f}% ({summary['diff']:+.1f}, 95% CI "
                f"{summary['ci'][0]:+.1f} to {summary['ci'][1]:+.1f}, McNemar p={summary['mcnemar_p']:.3g})")
        if "hit_limit" in summary:
            line += f", hit token limit {summary['hit_limit'][0]:.0f}% → {summary['hit_limit'][1]:.0f}%"
        lines.append(line)
        lines += [f"   {group}: {part['original']:.1f}% → {part['edited']:.1f}%"
                  for group, part in summary.get("groups", {}).items()]
    parts.append("<p><b>Results on held-out data</b></p><pre>" + e("\n".join(lines) or "not evaluated yet") + "</pre>")
    timeline = timeline_plot(run)
    if timeline:
        parts.append(f'<p><b>How the run went</b></p><img src="{timeline}" alt="passes used and tensors edited per turn">')
        last = next((t["edited"] for t in reversed(run["transcript"]) if t.get("edited") is not None), [])
        parts.append(f'<p class="muted">Edited tensors at the end: {e(", ".join(last) or "none")}</p>')
    if e2:
        tensor, structure = main_structure(e2)
        lines = [f"{e2['edit_map']['changed']:,} weights changed in {len(e2['edit_map']['tensors'])} tensors; "
                 f"gain on the select split {e2['select_gain']:+.1f}"]
        if tensor:
            lines.append(f"largest change: {tensor}, rank for 99% of it {structure.get('rank_for_99pct', '?')}, "
                         f"fit to removing one direction R² {structure.get('projection_r2', float('nan')):.2f}")
        lines += [f"   alone, {name}: {gain:+.1f}" for name, gain in e2.get("tensor_alone", {}).items()]
        if e2.get("smallest_fraction_for_90pct") is not None:
            lines.append(f"keeping {100 * e2['smallest_fraction_for_90pct']:g}% of the changed columns keeps 90% of "
                         "the gain")
        parts.append("<p><b>What changed (E2)</b></p><pre>" + e("\n".join(lines)) + "</pre>")
    examples = run["evals"].get(run["task"], {}).get("examples", {})
    for key, title in (("wrong_to_right", "Became right"), ("right_to_wrong", "Became wrong")):
        for example in examples.get(key, [])[:3]:
            parts.append(f'<p><b>{title}</b>: {e(example["prompt"][:220])}</p><div class="grid">'
                         f'<div><span class="muted">original</span><pre>{e(example["original"])}</pre></div>'
                         f'<div><span class="muted">edited</span><pre>{e(example["edited"])}</pre></div></div>')
    if run["final"]:
        parts.append(f"<p><b>Agent's final explanation</b></p><pre>{e(run['final'])}</pre>")
    parts.append(f'<p class="muted">Folder: {e(str(run["path"]))}</p></details>')
    return "".join(parts)


def build(runs_root):
    runs_root = Path(runs_root)
    runs = [load_run(p.parent) for p in sorted((runs_root / "e1").glob("*/status.json"))]
    reviews = load_reviews(runs_root / "reviews.csv", runs)
    rows = [index_row(run, reviews.get(run["name"], {})) for run in runs]
    if rows:
        with open(runs_root / "index.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    tasks = [t for t in ("loops", "markdown", "facts", "addition") if any(r["task"] == t for r in runs)]
    reviewed = sum(1 for r in runs if reviews.get(r["name"], {}).get("verdict"))
    e2_summaries, similarity = load_e2(runs_root)
    columns, matrix, combos = load_e3(runs_root)
    demo = (runs_root / "DEMO").exists()
    page = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>Results review</title><style>{CSS}</style>"
            f"</head><body><main><h1>Results review</h1>",
            f'<p class="muted">{len(runs)} finished E1 runs · {reviewed} reviewed · {len(e2_summaries)} analysed in '
            f'E2 · {len(matrix)} in the E3 matrix · built from {e(str(runs_root))}</p>']
    if demo:
        page.append('<div class="banner">Example only: every number, answer and transcript on this page is made up, '
                    "to show the layout. It is not an experimental result.</div>")
    if not runs:
        page.append("<p>No finished runs yet.</p>")
    else:
        page += ["<h2>E1 · Did the edits work?</h2><p class='muted'>Each dot is one run's gain on its own task's "
                 "held-out test set, with its 95% interval; the thick tick is the mean of the replicates.</p>",
                 f'<img src="{forest_plot(runs, tasks)}" alt="gain per condition and task">',
                 "<h2>E1 · Were they targeted?</h2><p class='muted'>Gain on the run's own task against the mean change "
                 "on the general panel. Below the red line, general skills dropped more than 2 points.</p>",
                 f'<img src="{targeting_plot(runs, tasks)}" alt="gain against general-panel change">']
        page.append("<h2>E2 · What changed?</h2>")
        if e2_summaries:
            page += ["<p class='muted'>Where each edit is: rows are runs, columns are layers, and the color is how "
                     "many weights changed there (grey: none).</p>",
                     f'<img src="{edit_location_plot(e2_summaries)}" alt="changed weights per layer for each run">']
            pruning = pruning_plot(e2_summaries)
            if pruning:
                page += ["<p class='muted'>How much of each edit matters: the share of its gain on the select split "
                         "that remains when only its largest changed columns are kept. Curves that stay above the red "
                         "line at small shares mean a small part of the edit does the work.</p>",
                         f'<img src="{pruning}" alt="gain kept against columns kept">']
            if similarity:
                page += ["<p class='muted'>Do edits agree? Cosine between the weight changes of every pair of runs "
                         "of a task: near 1 means the same change, near 0 unrelated.</p>",
                         f'<img src="{similarity_plot(similarity)}" alt="similarity between edits">']
            page.append(e2_table(e2_summaries))
        else:
            page.append("<p class='muted'>Not run yet: python experiments/e2.py</p>")
        page.append("<h2>E3 · What else did it affect?</h2>")
        if matrix:
            page += ["<p class='muted'>Each row is an edit, each column a held-out dataset; cells are edited minus "
                     "original. A targeted edit is strong in its own column and near zero elsewhere.</p>",
                     f'<img src="{transfer_plot(columns, matrix)}" alt="transfer matrix">', e3_tables(columns, matrix,
                                                                                               combos)]
        else:
            page.append("<p class='muted'>Not run yet: python experiments/e3.py</p>")
        page += ["<h2>All runs</h2><p class='muted'>Click a column to sort; click a run for its details.</p>",
                 table_html(runs, reviews), "<h2>Run details</h2>"]
        page += [run_card(run, reviews.get(run["name"], {}), e2_summaries.get(run["name"])) for run in runs]
    page.append(f"</main><script>{SORT_JS}</script></body></html>")
    out = runs_root / "review.html"
    out.write_text("".join(page))
    print(f"Wrote {out}" + (f" and {runs_root / 'index.csv'}; your notes go in {runs_root / 'reviews.csv'}" if rows
                            else " (no finished runs yet)"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=str(Path(__file__).resolve().parent.parent / "runs"))
    build(parser.parse_args().runs)
