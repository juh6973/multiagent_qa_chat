import faiss
import torch
import numpy as np
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import logging

logging.set_verbosity_error()


class DPR:

    def __init__(self):
        self.qencoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.qtokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.passages = self.__get_passages()
        self.faiss_index = self.__get_index()
        

    def retrieve_best_passage(self, query, top_k=3) -> list[tuple[str, float]]:
        query_inputs = self.qtokenizer(query, return_tensors="pt")
        with torch.no_grad(): query_embedding = self.qencoder(**query_inputs).pooler_output.cpu().numpy()

        D, I = self.faiss_index.search(query_embedding, k=top_k)
        results = [(self.passages[I[0][i]], D[0][i]) for i in range(top_k)]
        
        return results
    
    @staticmethod
    def __get_passages(passage_path="./dpr/passages.txt") -> list[str]:
        """Gets passages"""
        with open(passage_path, "r") as f:
            return [line.strip() for line in f]
        
    @staticmethod
    def __get_index(index_path="./dpr/faiss_index.bin"): return faiss.read_index(index_path)
