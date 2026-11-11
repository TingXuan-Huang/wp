"""What changed in the weights: edit map (T1), edit structure (T2), and similarity between edits (T4).

Sources: task vectors (Ilharco et al. 2022) for treating edited minus original as an object; rank-one edits
(Meng et al. 2022) and directional ablation, W - r r^T W (Arditi et al. 2024), for the structure tests.
"""
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

COMPONENTS = {   # Qwen names first, then GPT-2 names for the tiny test model
    "attn_in": r"(q_proj|k_proj|v_proj|in_proj|conv1d|c_attn)",
    "attn_out": r"(o_proj|out_proj|attn\.c_proj)",
    "mlp_in": r"(gate_proj|up_proj|c_fc)",
    "mlp_out": r"(down_proj|mlp\.c_proj)",
    "norm": r"(norm|ln_)",
}


def component(name):
    return next((label for label, pattern in COMPONENTS.items() if re.search(pattern, name)), "other")


def layer_of(name):
    match = re.search(r"\.(?:layers|h)\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def load_deltas(session, run_dir):
    """{tensor name: edited - original} in float32 on the CPU. The session must hold the original weights."""
    deltas = {}
    for path in sorted((Path(run_dir) / "edits").glob("*.safetensors")):
        original = session.originals.get(path.stem, session.params[path.stem])
        deltas[path.stem] = load_file(path)["t"].float() - original.float().cpu()
    return deltas


def apply_deltas(session, deltas):
    """Set the session's weights to original + deltas."""
    session.reset_edits()
    for name, delta in deltas.items():
        session._apply(name, session.params[name].float().cpu() + delta)


def edit_map(session, deltas):
    """T1: per edited tensor, how many weights changed and how large the change is relative to the tensor."""
    rows = []
    for name, delta in sorted(deltas.items()):
        original = session.params[name].float().cpu()
        changed = int((delta != 0).sum())
        rows.append({"tensor": name, "layer": layer_of(name), "component": component(name), "changed": changed,
                     "size": delta.numel(), "share": changed / delta.numel(),
                     "relative_norm": (delta.norm() / original.norm()).item()})
    total = sum(row["changed"] for row in rows)
    model_size = sum(p.numel() for p in session.model.parameters())
    return {"tensors": rows, "changed": total, "share_of_model": total / model_size}


def edit_structure(delta, original):
    """T2: which rows and columns changed, how low-rank the change is, and how well it fits W' = W - a u u^T W
    on the changed columns (u = the change's top singular direction)."""
    if delta.dim() != 2 or not (delta != 0).any():
        return {"changed": int((delta != 0).sum())}
    columns, rows = (delta != 0).any(0), (delta != 0).any(1)
    U, S, _ = torch.linalg.svd(delta, full_matrices=False)
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    u, W, D = U[:, 0], original[:, columns], delta[:, columns]
    P = torch.outer(u, u @ W)
    alpha = -(D * P).sum() / (P * P).sum().clamp_min(1e-12)
    r2 = 1 - ((D + alpha * P) ** 2).sum() / (D ** 2).sum()
    return {"changed_columns": int(columns.sum()), "columns": delta.shape[1], "changed_rows": int(rows.sum()),
            "rows": delta.shape[0], "rank_for_99pct": int((energy < 0.99).sum()) + 1,
            "top_singular_values": [round(s, 4) for s in S[:5].tolist()],
            "projection_alpha": alpha.item(), "projection_r2": r2.item()}


def keep_top_columns(deltas, fraction):
    """The edit restricted to the largest `fraction` of its changed columns in each tensor (1-D tensors kept whole)."""
    kept = {}
    for name, delta in deltas.items():
        if delta.dim() != 2:
            kept[name] = delta
            continue
        norms = delta.norm(dim=0)
        changed = int((norms > 0).sum())
        keep = max(1, round(changed * fraction))
        mask = torch.zeros_like(norms, dtype=torch.bool)
        mask[norms.topk(keep).indices] = True
        kept[name] = delta * mask
    return kept


def changed_units(deltas):
    """(tensor, column) pairs that changed; a changed 1-D tensor counts as one unit."""
    units = set()
    for name, delta in deltas.items():
        changed = (delta != 0).any(0) if delta.dim() == 2 else (delta != 0).any().view(1)
        units.update((name, int(j)) for j in changed.nonzero().flatten())
    return units


def top_directions(deltas):
    """The top left singular vector of each 2-D change."""
    return {name: torch.svd_lowrank(delta, q=2)[0][:, 0] for name, delta in deltas.items() if delta.dim() == 2}


def similarity(a, b, directions_a=None, directions_b=None):
    """T4: overlap of changed columns, cosine between the whole changes, and the best |cosine| between the top
    directions of tensors both edits changed (singular vectors have arbitrary sign, hence the absolute value)."""
    units_a, units_b = changed_units(a), changed_units(b)
    norm = lambda deltas: sum((d ** 2).sum().item() for d in deltas.values()) ** 0.5
    shared = set(a) & set(b)
    dot = sum((a[n] * b[n]).sum().item() for n in shared)
    directions_a = directions_a or top_directions({n: a[n] for n in shared})
    directions_b = directions_b or top_directions({n: b[n] for n in shared})
    best = max((abs(torch.dot(directions_a[n], directions_b[n]).item()) for n in shared
                if n in directions_a and n in directions_b), default=None)
    return {"overlap": len(units_a & units_b) / max(1, len(units_a | units_b)),
            "cosine": dot / (norm(a) * norm(b) + 1e-12), "top_direction": best}
