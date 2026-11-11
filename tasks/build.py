"""Download and build every dataset into prompts/test/. Seeds are fixed, so every machine gets the same data.

    python -m tasks.build
"""
from tasks import addition, facts, general, loops, markdown

if __name__ == "__main__":
    for task in (general, loops, markdown, facts, addition):
        task.build()
