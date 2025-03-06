from transformers import pipeline
from transformers import logging
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.set_verbosity_error()

class BERTQA:

    def __init__(self, model_name="distilbert/distilbert-base-cased-distilled-squad", conf_threshold=0.8):
        self.qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)
        self.conf_threshold = conf_threshold

    def answer_question(self, question: str, summary: str) -> str:
        try:
            response = self.qa_pipeline({"question": question, "context": summary})
            answer, score = response["answer"], response["score"]
        except Exception: 
            return self.__answer_fallback()
        
        is_unsatisfying_answer = score < self.conf_threshold or not answer.strip()
        answer = self.__answer_fallback() if is_unsatisfying_answer else answer

        return answer

    def __answer_fallback(self) -> str: 
        return """
        I'm sorry, but I don't have enough information to answer that right now.
        Could you provide more details or clarify your question?
        """