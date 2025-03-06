import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import logging

logging.set_verbosity_error()

class Summarizer:

    def __init__(self, repo="Juh6973/t5-small-summarizer-cnn-dailymail"):
        self.tokenizer = T5Tokenizer.from_pretrained(repo)
        self.model = T5ForConditionalGeneration.from_pretrained(repo)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40, num_beams: int = 4) -> str:
        """Generate a summary for the given text."""
        prompt = self.__get_prompt() + text
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True)
        inputs = inputs.to(self.device)
        
        # Generate summary
        summary_ids = self.model.generate(
            inputs, 
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True
        )
        
        # Decode and return summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def __get_prompt(self) -> str:
        """Creates the prompt prefix for T5."""
        return "summarize: "