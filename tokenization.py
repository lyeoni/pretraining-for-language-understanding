class Vocab(object):
    def __init__(self, list_of_tokens, embedding_vec=None, unk_token=None,
                 bos_token=None, eos_token=None, pad_token=None, min_freq=1, lower=True):
        self.list_of_tokens = list_of_tokens
        self.embedding_vec = embedding_vec
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.min_freq = min_freq
        self.lower = lower
        self.stoi, self.itos, self.freqs = {}, {}, {}

        # vocab
        for sti, special_token in enumerate([self.unk_token, self.bos_token, self.eos_token, self.pad_token]):
             if special_token: 
                 self.stoi[special_token] = sti
                 self.itos[sti] = special_token

    def build(self):
        # if the token doesn't appear in the vocabulary at least once
        for ti, token in enumerate(self.list_of_tokens):
            # lowercase the token
            if self.lower:
                token = token.lower()

            if token not in self.stoi.keys():
                self.itos[self.__len__()] = token
                self.stoi[token] = self.__len__()
                self.freqs[token] = 1
            else:
                self.freqs[token] += 1
        
        # sort by frequency in 'descending' order
        self.freqs = dict(sorted(self.freqs.items(), key=lambda x: x[1], reverse=True))
        # minimum frequency required for a token
        for token, freq in self.freqs.items():
            if freq < self.min_freq:
                del self.itos[self.stoi[token]]
                del self.stoi[token]

    def __len__(self):
        return len(self.stoi)

class Tokenizer(object):
    def __init__(self, tokenization_fn, vocab=None, max_seq_length=128):
        self.tokenization_fn = tokenization_fn
        self.vocab = vocab
        self.max_seq_length = max_seq_length

    def tokenize(self, text):
        tokens = self.tokenization_fn(text)
        if self.vocab:
            # lowercase the token
            if self.vocab.lower:
                tokens = [token.lower() for token in tokens]
            
            # add beginning of sentence token
            if self.vocab.bos_token:
                tokens = [self.vocab.bos_token] + tokens 
            
            # add end of sentence token
            if self.vocab.eos_token:    
                tokens = tokens + [self.vocab.eos_token]
            
            # add padding token
            if self.vocab.pad_token and len(tokens) < self.max_seq_length:
                tokens += [self.vocab.pad_token] * (self.max_seq_length-len(tokens))
            elif self.vocab.pad_token and len(tokens) >= self.max_seq_length:
                tokens = tokens[:self.max_seq_length]

        return tokens
    
    def transform(self, tokens):
        if self.vocab:
            return [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi[self.vocab.unk_token] for token in tokens]
    
    def inverse_transform(self, indices):
        if self.vocab:
            return [self.vocab.itos[index] for index in indices]

    def tokenize_and_transform(self, text):
        if self.vocab:
            return self.transform(self.tokenize(text))
    