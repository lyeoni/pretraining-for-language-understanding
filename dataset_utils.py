import torch
from torch.utils.data import Dataset

class Corpus(Dataset):
    def __init__(self, corpus_path, tokenizer):
        self.corpus = []
        self.tokenizer = tokenizer

        with open(corpus_path, 'r', encoding='utf8') as reader:
            for li, line in enumerate(reader):
                self.corpus.append(line.strip())

    def __getitem__(self, index):
        tokens_indices = self.tokenizer.tokenize_and_transform(self.corpus[index])
        tokens_indices = torch.tensor(tokens_indices)
        return tokens_indices[:-1], tokens_indices[1:]

    def __len__(self):
        return len(self.corpus)