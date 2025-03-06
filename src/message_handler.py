import json

class MessageHandler:

    def __init__(self, msg_path="./src/messages.json"):
        with open(msg_path, "r", encoding="utf-8") as f: 
            self.messages = json.load(f)

    def get(self, key: str) -> str: return self.messages.get(key)