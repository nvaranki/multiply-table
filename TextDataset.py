import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, data):
        # Vocabulary (token-to-index mapping)
        self.vocab = {word: i for i, word in enumerate(set(" ".join(data).split()))}
        self.vocab_size = len(self.vocab)
        # tokenized data
        self.data = [TextDataset.__tokenize_and_convert_to_indices(
            sentence, self.vocab) for sentence in data]

    # Tokenize and convert to indices
    @staticmethod
    def __tokenize_and_convert_to_indices(sentence, vocab):
        return [vocab[word] for word in sentence.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)
