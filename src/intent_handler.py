
class IntentHandler:


    def __init__(self):
        self.summary_keywords = ["tell me about", "describe", "explain", "summarize"]
        self.question_words = ["what", "who", "why", "how", "where", "when"]

    def detect_intent(self, user_input: str) -> str:
        user_input = user_input.strip().lower()

        for kw in self.summary_keywords:
            if kw in user_input: return "summary"

        if user_input.endswith("?"): return "qa"
        if any(user_input.startswith(qw + " ") for qw in self.question_words): return "qa"

        return "qa"