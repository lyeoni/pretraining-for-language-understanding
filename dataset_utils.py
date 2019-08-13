import torch
from torch.utils.data import Dataset

class Corpus(Dataset):
    def __init__(self, corpus_path, tokenizer, model_type, cuda):
        self.corpus = []
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.cuda = cuda

        with open(corpus_path, 'r', encoding='utf8') as reader:
            for li, line in enumerate(reader):
                self.corpus.append(line.strip())

    def __getitem__(self, index):
        """
        Return inputs, targets tensors used for model training.
        The size of returned tensor is as follows.

        if self.model_type == 'LSTM':
            |inputs|    = (batch_size, max_seq_len-1)
            |targets|   = (batch_size, max_seq_len-1)
        
        elif self.model_type == 'BiLSTM':
            |inputs|    = (batch_size, max_seq_len)
            |targets|   = (batch_size, max_seq_len-2)
        """
        
        tokens_indices = self.tokenizer.tokenize_and_transform(self.corpus[index])
        tokens_indices = torch.tensor(tokens_indices)
        if self.cuda:
            tokens_indices = tokens_indices.cuda()

        if self.model_type == 'LSTM':
            return tokens_indices[:-1], tokens_indices[1:]
        elif self.model_type == 'BiLSTM':
            return tokens_indices, tokens_indices[1:-1]

    def __len__(self):
        return len(self.corpus)