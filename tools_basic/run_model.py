"""Run the model. Each generate() or read() call that succeeds costs one forward pass."""

session = None   # set by the kernel


def passes_left():
    """Return how many forward passes remain."""
    return session.cfg["budget"]["forward_passes"] - session.passes_used


def generate(prompts, model="edited", n=1, max_new_tokens=None):
    """One pass. Answer each user message n times with the evaluation's sampling settings.
    model is "original", "edited" or "both". Returns a list of {"text", "tokens", "hit_limit"},
    or with "both", {"original": [...], "edited": [...]} on the same seeds for a paired comparison."""
    session.check_budget()
    items = [prompt for prompt in prompts for _ in range(n)]
    per_model = 2 if model == "both" else 1
    limit = session.cfg["budget"]["max_generations_per_pass"]
    if len(items) * per_model > limit:
        raise ValueError(f"{len(items) * per_model} generations is over the per-pass limit of {limit}")
    if model == "both":
        result = {which: session.sample(items, which, max_new_tokens) for which in ("original", "edited")}
    else:
        result = session.sample(items, model, max_new_tokens)
    session.count_pass()
    return result


def read(texts, model="edited", capture=()):
    """One pass. Run raw texts without generating. `capture` lists module names (see list_modules).
    Returns one {"ids", "logprobs", "capture": {module: activations}} per text; tensors are CPU float32."""
    session.check_budget()
    limit = session.cfg["budget"]["max_texts_per_pass"]
    if len(texts) > limit:
        raise ValueError(f"{len(texts)} texts is over the per-pass limit of {limit}")
    result = session.trace(texts, model, capture)
    session.count_pass()
    return result


def tokenizer():
    """Return the model's tokenizer."""
    return session.tokenizer
