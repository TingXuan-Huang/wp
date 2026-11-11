"""One loaded model plus its edits. Everything that runs or changes the model goes through here."""
import json
import re
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList


class PresencePenalty(LogitsProcessor):
    """Subtract a fixed penalty from every token already generated (the vLLM setting the post used)."""

    def __init__(self, penalty, prompt_length):
        self.penalty, self.prompt_length = penalty, prompt_length

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool).scatter(1, generated, True)
        return scores - self.penalty * seen


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Session:
    def __init__(self, cfg, run_dir=None):
        """With a run_dir, edits and the pass count are saved there and survive a kernel restart."""
        self.cfg, self.spec = cfg, cfg["model_spec"]
        self.device = pick_device(cfg["device"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec["hf_id"], padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.spec["hf_id"], dtype=self.spec["dtype"])
        self.model.to(self.device).eval()
        self.params = self.model.state_dict()   # shares memory with the model
        self.originals = {}                     # name -> original tensor, only for edited tensors
        torch.set_grad_enabled(False)

        self.run_dir = Path(run_dir) if run_dir else None
        self.passes_used = 0
        if self.run_dir:
            (self.run_dir / "edits").mkdir(exist_ok=True)
            self.load_edits(self.run_dir / "edits")
            if (self.run_dir / "budget.json").exists():
                self.passes_used = json.loads((self.run_dir / "budget.json").read_text())["passes_used"]

    # ---------- edits ----------

    def load_edits(self, folder):
        for path in Path(folder).glob("*.safetensors"):
            self._apply(path.stem, load_file(path)["t"])

    def _apply(self, name, tensor):
        param = self.params[name]
        self.originals.setdefault(name, param.cpu().clone())
        param.copy_(tensor.to(param.device, param.dtype))

    def save_edit(self, name, tensor):
        if name not in self.params:
            raise KeyError(f"No tensor named {name}")
        if any(re.search(pattern, name) for pattern in self.spec.get("frozen", [])):
            raise ValueError(f"{name} is frozen in this experiment")
        if tensor.shape != self.params[name].shape:
            raise ValueError(f"Shape {tuple(tensor.shape)} != {tuple(self.params[name].shape)}")
        self._apply(name, tensor)
        if self.run_dir:
            save_file({"t": self.params[name].cpu().clone()}, self.run_dir / "edits" / f"{name}.safetensors")

    def drop_edit(self, name):
        self.params[name].copy_(self.originals.pop(name).to(self.params[name].device))
        if self.run_dir:
            (self.run_dir / "edits" / f"{name}.safetensors").unlink(missing_ok=True)

    @contextmanager
    def use(self, which):
        """Temporarily run with the "original" weights. "edited" is the current state."""
        if which not in ("original", "edited"):
            raise ValueError('Use "original" or "edited"')
        if which == "edited":
            yield
            return
        edited = {name: self.params[name].clone() for name in self.originals}
        for name, tensor in self.originals.items():
            self.params[name].copy_(tensor.to(self.params[name].device))
        try:
            yield
        finally:
            for name, tensor in edited.items():
                self.params[name].copy_(tensor)

    # ---------- forward-pass budget ----------

    def check_budget(self):
        if self.passes_used >= self.cfg["budget"]["forward_passes"]:
            raise RuntimeError("Forward-pass budget is used up")

    def count_pass(self):
        """Called after a pass succeeds, so a failed request costs nothing."""
        self.passes_used += 1
        if self.run_dir:
            (self.run_dir / "budget.json").write_text(json.dumps({"passes_used": self.passes_used}))

    # ---------- running the model (not metered; the tools add the metering) ----------

    def _render(self, prompt, template=None):
        """Wrap a user message in the chat template. `template` replaces the model's template options."""
        if not self.tokenizer.chat_template:
            return prompt
        options = self.spec.get("chat_template", {}) if template is None else template
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **options)

    def _batches(self, items):
        size = self.cfg["batch_size"]
        return (items[i:i + size] for i in range(0, len(items), size))

    def sample(self, prompts, which="edited", max_new_tokens=None, settings=None, template=None):
        """Generate one answer per prompt: [{"text", "tokens", "hit_limit"}, ...].
        `settings` replaces the config's generation settings, and `template` the chat-template options."""
        settings = dict(self.cfg["generation"] if settings is None else settings)
        presence_penalty = settings.pop("presence_penalty", 0)
        if max_new_tokens:
            settings["max_new_tokens"] = min(max_new_tokens, settings["max_new_tokens"])
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = torch.tensor(eos_ids if isinstance(eos_ids, list) else [eos_ids], device=self.device)
        pad_id = self.tokenizer.pad_token_id

        results = []
        with self.use(which):
            for index, batch in enumerate(self._batches([self._render(p, template) for p in prompts])):
                torch.manual_seed(self.cfg["seed"] + index)   # per batch, so both weights stay paired on every batch
                encoded = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.device)
                start = encoded.input_ids.shape[1]
                penalty = [PresencePenalty(presence_penalty, start)] if presence_penalty else []
                output = self.model.generate(
                    **encoded, do_sample=True, pad_token_id=pad_id,
                    logits_processor=LogitsProcessorList(penalty), **settings)
                for ids in output[:, start:]:
                    results.append({
                        "text": self.tokenizer.decode(ids, skip_special_tokens=True),
                        "tokens": int((ids != pad_id).sum()),
                        "hit_limit": not torch.isin(ids, eos_ids).any().item(),
                    })
        return results

    def trace(self, texts, which="edited", capture=()):
        """Run texts without generating. Returns token ids, next-token log-probs and captured activations."""
        modules = dict(self.model.named_modules())
        captured, hooks = {}, []
        for name in capture:
            def hook(_module, _inputs, output, name=name):
                captured[name] = (output[0] if isinstance(output, tuple) else output).float().cpu()
            hooks.append(modules[name].register_forward_hook(hook))

        results = []
        try:
            with self.use(which):
                for batch in self._batches(texts):
                    encoded = self.tokenizer(batch, return_tensors="pt", padding=True, padding_side="right").to(self.device)
                    ids, lengths = encoded.input_ids, encoded.attention_mask.sum(1).tolist()
                    log_probs = self.model(**encoded).logits[:, :-1].log_softmax(-1)
                    log_probs = log_probs.gather(2, ids[:, 1:, None]).squeeze(-1).float().cpu()
                    for row, length in enumerate(lengths):
                        results.append({
                            "ids": ids[row, :length].cpu(),
                            "logprobs": log_probs[row, :length - 1],
                            "capture": {name: value[row, :length] for name, value in captured.items()},
                        })
        finally:
            for handle in hooks:
                handle.remove()
        return results

    @contextmanager
    def hooks(self, changes):
        """Run with some module outputs changed: {module name: function(output tensor) -> new output tensor}."""
        modules = dict(self.model.named_modules())
        handles = []
        for name, change in changes.items():
            def hook(_module, _inputs, output, change=change):
                if isinstance(output, tuple):
                    return (change(output[0]),) + output[1:]
                return change(output)
            handles.append(modules[name].register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def score_continuations(self, prompts, continuations, which="edited"):
        """Total log-probability of each continuation after its prompt, as raw text without the chat template."""
        encode = lambda text: self.tokenizer(text, add_special_tokens=False).input_ids
        pairs = [(encode(prompt), encode(continuation)) for prompt, continuation in zip(prompts, continuations)]
        pad = self.tokenizer.pad_token_id
        scores = []
        with self.use(which):
            for batch in self._batches(pairs):
                length = max(len(p) + len(c) for p, c in batch)
                ids = torch.tensor([p + c + [pad] * (length - len(p) - len(c)) for p, c in batch], device=self.device)
                mask = torch.tensor([[1] * (len(p) + len(c)) + [0] * (length - len(p) - len(c)) for p, c in batch],
                                    device=self.device)
                log_probs = self.model(input_ids=ids, attention_mask=mask).logits.float().log_softmax(-1)
                for row, (p, c) in enumerate(batch):
                    positions = torch.arange(len(p) - 1, len(p) + len(c) - 1, device=self.device)
                    targets = torch.tensor(c, device=self.device)
                    scores.append(log_probs[row, positions, targets].sum().item())
        return scores

    def layer_names(self):
        """Names of the decoder layers, in order. Their outputs are the residual stream."""
        return [name for name, _ in self.model.named_modules() if re.fullmatch(r".*\.(layers|h)\.\d+", name)]

    def logits_from_residual(self, hidden):
        """Apply the final norm and the output layer to residual-stream vectors (the logit lens)."""
        base = self.model.base_model
        norm = base.norm if hasattr(base, "norm") else base.ln_f
        hidden = hidden.to(self.device, self.model.dtype)
        return self.model.get_output_embeddings()(norm(hidden)).float().cpu()

    def reset_edits(self):
        """Go back to the original weights. Files on disk are left alone."""
        for name, tensor in self.originals.items():
            self.params[name].copy_(tensor.to(self.params[name].device))
        self.originals = {}
