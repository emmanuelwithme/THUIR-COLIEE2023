import pickle
import torch
from typing import List, Dict

class EmbeddingsData:
    def __init__(self, ids: List[str], embeddings: torch.Tensor):
        self.ids = ids
        self.embeddings = embeddings
        self.id2vec: Dict[str, torch.Tensor] = {
            id: vec for id, vec in zip(ids, embeddings)
        }

    def save(self, path: str):
        """將 EmbeddingsData 儲存成 .pkl 檔"""
        with open(path, 'wb') as f:
            pickle.dump({
                'ids': self.ids,
                'embeddings': self.embeddings
            }, f)

    @classmethod
    def load(cls, path: str):
        """從 .pkl 檔讀取並建立 EmbeddingsData"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(data['ids'], data['embeddings'])