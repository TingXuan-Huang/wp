"""Talk to a model yourself, with or without a run's edits.

    python chat.py configs/qwen_math.yaml
    python chat.py configs/qwen_math.yaml --edits runs/<run>/edits
    python chat.py configs/qwen_math.yaml generation.max_new_tokens=512
"""
import sys

from playground.config import load_config
from playground.session import Session


def main():
    args = sys.argv[2:]
    edits = None
    if "--edits" in args:
        position = args.index("--edits")
        edits = args[position + 1]
        del args[position:position + 2]
    session = Session(load_config(sys.argv[1], args))
    if edits:
        session.load_edits(edits)
    print(f"Loaded on {session.device}. Edited tensors: {len(session.originals)}. Ctrl-D to quit.")
    while True:
        try:
            prompt = input("\nyou> ")
        except EOFError:
            break
        print(session.sample([prompt])[0]["text"])


if __name__ == "__main__":
    main()
