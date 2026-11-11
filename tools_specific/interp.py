"""Interpretability tools for the agent. To give them to it, add "tools_specific.interp" to `tools` in the config.

Where each method comes from is listed in experiment.md. A call that runs the model costs one forward pass.
"""
import torch

from playground.config import ROOT

session = None   # set by the kernel
_vocabulary = None


def _decode(ids):
    return [session.tokenizer.decode([int(i)]) for i in ids]


def _task_vocabulary():
    """Token ids that occur in the practice data. Showing only these keeps arbitrary vocabulary out of the output."""
    global _vocabulary
    if _vocabulary is None:
        text = (ROOT / session.cfg["practice_data"]).read_text()
        _vocabulary = torch.tensor(sorted(set(session.tokenizer(text, add_special_tokens=False).input_ids)))
    return _vocabulary


def direction_tokens(direction, k=15, vocabulary="task"):
    """No pass. Tokens that a residual-stream direction pushes up and down, read through the final norm and the
    output layer. vocabulary="task" lists only tokens that occur in the practice data; "all" lists any token.
    Returns {"promotes": [...], "suppresses": [...]}."""
    logits = session.logits_from_residual(torch.as_tensor(direction, dtype=torch.float32).flatten()[None])[0]
    ids = _task_vocabulary() if vocabulary == "task" else torch.arange(len(logits))
    scores = logits[ids] - logits[ids].mean()
    return {"promotes": _decode(ids[scores.topk(k).indices]), "suppresses": _decode(ids[(-scores).topk(k).indices])}


def logit_lens(prompts, model="edited", position=-1, target=None, k=5):
    """One pass. What the model would predict next at each layer, at one token position of each prompt. Prompts are
    user messages and get the chat template. With target (a string), also the rank of its first token at each layer.
    Returns one list per prompt: [{"layer", "top", "target_rank"}, ...]."""
    session.check_budget()
    names = session.layer_names()
    traces = session.trace([session._render(p) for p in prompts], model, capture=names)
    target_id = session.tokenizer(target, add_special_tokens=False).input_ids[0] if target else None
    results = []
    for trace in traces:
        logits = session.logits_from_residual(torch.stack([trace["capture"][name][position] for name in names]))
        rows = []
        for layer, row in enumerate(logits):
            entry = {"layer": layer, "top": _decode(row.topk(k).indices)}
            if target_id is not None:
                entry["target_rank"] = int((row > row[target_id]).sum()) + 1
            rows.append(entry)
        results.append(rows)
    session.count_pass()
    return results


def find_direction(positive, negative, layer, model="edited"):
    """One pass. A direction in the residual stream after `layer`: the mean activation of the positive texts minus
    that of the negative texts (raw texts, averaged over their tokens). It is also checked on held-out halves.
    Needs at least 4 texts of each kind. Returns {"direction", "heldout_auc", "split_half_cosine"}."""
    if min(len(positive), len(negative)) < 4:
        raise ValueError("Give at least 4 positive and 4 negative texts")
    session.check_budget()
    name = session.layer_names()[layer]

    def means(texts):
        return torch.stack([trace["capture"][name].mean(0) for trace in session.trace(texts, model, capture=[name])])

    pos, neg = means(positive), means(negative)
    session.count_pass()
    half_pos, half_neg = len(pos) // 2, len(neg) // 2
    first = pos[:half_pos].mean(0) - neg[:half_neg].mean(0)
    second = pos[half_pos:].mean(0) - neg[half_neg:].mean(0)
    auc = ((pos[half_pos:] @ first)[:, None] > (neg[half_neg:] @ first)[None]).float().mean().item()
    direction = pos.mean(0) - neg.mean(0)
    return {"direction": direction / direction.norm(), "heldout_auc": round(auc, 3),
            "split_half_cosine": round(torch.cosine_similarity(first, second, dim=0).item(), 3)}


def steer(prompts, layer, direction, strength, model="edited", n=1, max_new_tokens=None):
    """One pass. Generate while adding strength x direction to the output of `layer` at every position, scaled by
    each position's activation size: strength 0.1 adds 10% of it, and negative strengths subtract. This changes
    activations, not weights, so use it to test an idea before editing. Returns results like generate()."""
    session.check_budget()
    items = [prompt for prompt in prompts for _ in range(n)]
    limit = session.cfg["budget"]["max_generations_per_pass"]
    if len(items) > limit:
        raise ValueError(f"{len(items)} generations is over the per-pass limit of {limit}")
    unit = torch.as_tensor(direction, dtype=torch.float32).flatten()
    unit = unit / unit.norm()

    def add(hidden):
        return hidden + strength * hidden.norm(dim=-1, keepdim=True) * unit.to(hidden.device, hidden.dtype)

    with session.hooks({session.layer_names()[layer]: add}):
        result = session.sample(items, model, max_new_tokens)
    session.count_pass()
    return result


def patch(clean, corrupted, components, target, foil, model="edited", positions=None):
    """One pass, up to 8 components. Activation patching: run the corrupted prompts with one component's output
    copied from the clean prompts, and report how much of the clean behaviour that restores (1 = all of it). The
    behaviour is log p(target) - log p(foil) for the next token. Prompts are user messages; each clean/corrupted pair
    must have the same number of tokens. Components are module names from list_modules(), such as
    "model.layers.15.mlp". positions copies only some token positions, counted from the end (e.g. [-1]); copying a
    whole layer's output at every position trivially restores everything.
    Returns {"clean", "corrupted", "patched": {component: value}, "restored": {component: fraction}}; the fractions
    are None when clean and corrupted differ by less than 0.1, since dividing by that gap would be meaningless."""
    if len(components) > 8:
        raise ValueError("At most 8 components per pass")
    session.check_budget()
    tokenizer = session.tokenizer
    clean_texts = [session._render(p) for p in clean]
    corrupted_texts = [session._render(p) for p in corrupted]
    if [len(tokenizer(t).input_ids) for t in clean_texts] != [len(tokenizer(t).input_ids) for t in corrupted_texts]:
        raise ValueError("Each clean/corrupted pair needs the same number of tokens")
    target_id, foil_id = (tokenizer(word, add_special_tokens=False).input_ids[0] for word in (target, foil))

    def behaviour(texts, changes):
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(session.device)
        with session.use(model), session.hooks(changes):
            log_probs = session.model(**encoded).logits[:, -1].float().log_softmax(-1)
        return (log_probs[:, target_id] - log_probs[:, foil_id]).mean().item()

    cache = {}

    def keep(name):
        def store(hidden):
            cache[name] = hidden.detach().clone()
            return hidden
        return store

    def copy_from_clean(name):
        def swap(hidden):
            patched = hidden.clone()
            index = slice(None) if positions is None else positions
            patched[:, index] = cache[name][:, index]
            return patched
        return swap

    clean_value = behaviour(clean_texts, {name: keep(name) for name in components})
    corrupted_value = behaviour(corrupted_texts, {})
    gap = clean_value - corrupted_value
    patched = {name: behaviour(corrupted_texts, {name: copy_from_clean(name)}) for name in components}
    session.count_pass()
    return {"clean": round(clean_value, 3), "corrupted": round(corrupted_value, 3),
            "patched": {name: round(value, 3) for name, value in patched.items()},
            "restored": {name: round((value - corrupted_value) / gap, 3) if abs(gap) >= 0.1 else None
                         for name, value in patched.items()}}
