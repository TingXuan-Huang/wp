"""Replay fixed replies, to test the harness without paying for an agent."""


class Scripted:
    def __init__(self, replies, web_search):
        self.replies = iter(replies)
        self.info = {}

    def send(self, text):
        return next(self.replies, "Finished: the script has no more replies.")
