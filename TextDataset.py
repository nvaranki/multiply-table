from typing import List

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, data):
        # Vocabulary (token-to-index mapping)
        words = list(set(" ".join(data).split()))
        words.sort()
        self.vocab = {word: i for i, word in enumerate(words)}
        self.vocab_size = len(self.vocab)
        # tokenized data
        self.data = [self.__tokenize_and_convert_to_indices(sentence)
            for sentence in data]

    # Tokenize and convert to indices
    def __tokenize_and_convert_to_indices(self, sentence):
        return [self.vocab[word] for word in sentence.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

    @staticmethod
    def generate(r1: tuple, r2: tuple, op: List[str] = " * ", eq: List[str] = " = ") -> List[str]:
        rl = list()
        # TODO "six times two equals twelve"
        for o in op:
            for e in eq:
                for a in range(r1[0],r1[1]+1):
                    for b in range(r2[0],r2[1]+1):
                        rl.append( str(a) + o + str(b) + e + str(a*b) )
        return rl
