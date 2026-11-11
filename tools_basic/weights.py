"""Inspect and edit weights. Every public function in this folder is callable by the agent,
and its signature and docstring are added to the agent's prompt automatically."""
import re

session = None   # set by the kernel


def list_params(pattern=""):
    """Print name, shape and size of each tensor whose name matches the regex `pattern`."""
    for name, tensor in session.params.items():
        if re.search(pattern, name):
            print(name, tuple(tensor.shape), tensor.numel())


def list_modules(pattern=""):
    """Print name and class of each module matching `pattern`. Use these names for read(capture=...)."""
    for name, module in session.model.named_modules():
        if name and re.search(pattern, name):
            print(name, type(module).__name__)


def load(name):
    """Return a float32 CPU copy of a tensor, including your edit if there is one."""
    return session.params[name].float().cpu().clone()


def load_original(name):
    """Return a float32 CPU copy of a tensor as it was before any edit."""
    tensor = session.originals.get(name, session.params[name])
    return tensor.float().cpu().clone()


def save_edit(name, tensor):
    """Replace a tensor with your values (same shape). This is the only way to change the model."""
    session.save_edit(name, tensor)


def drop_edit(name):
    """Undo your edit to a tensor."""
    session.drop_edit(name)


def edited():
    """Return {name: number of weights changed} for every edited tensor."""
    return {name: int((load(name) != load_original(name)).sum()) for name in session.originals}
