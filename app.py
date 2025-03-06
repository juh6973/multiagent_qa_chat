from src.dpr import DPR
from src.summarizer import Summarizer
from src.bert_qa import BERTQA
from src.intent_handler import IntentHandler
from src.message_handler import MessageHandler


def main():
    message_handler = MessageHandler()
    print(message_handler.get("start"))

    dpr = DPR()
    summarizer = Summarizer()
    bert = BERTQA(conf_threshold=0.2)
    intent_handler = IntentHandler()

    print(message_handler.get("welcome"))
    print(message_handler.get("instructions"))

    while True:
        query = input(message_handler.get("user"))
        if query.strip().lower() in ("quit", "exit"): 
            print(f"{message_handler.get('chatbot')}{message_handler.get('goodbye')}")
            break

        intent = intent_handler.detect_intent(query)

        results = dpr.retrieve_best_passage(query, top_k=5)
        references = "\n".join([text for text, _ in results])
        response = summarizer.summarize(references)
        if intent == "qa": response = bert.answer_question(query, response)

        print(f"{message_handler.get('chatbot')}{response}\n")

if __name__ == "__main__":
    main()