import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, data):
        # Vocabulary (token-to-index mapping)
        self.vocab = {word: i for i, word in enumerate(set(" ".join(data).split()))}
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
