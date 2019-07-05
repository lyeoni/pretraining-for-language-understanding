class Vocab(object):
    def __init__(self, list_of_tokens, unk_token='<unk>', pad_token='<pad>',
                 bos_token='<bos>', eos_token='<eos>'):
        self.list_of_tokens = list_of_tokens
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # vocab
        self.stoi, self.itos, self.freq = {}, {}, {}
        for sti, special_token in enumerate([self.unk_token, self.pad_token, self.bos_token, self.eos_token]):
             if special_token: 
                 self.stoi[special_token] = sti
                 self.itos[sti] = special_token

    def build(self):
        for ti, token in enumerate(self.list_of_tokens):
            # if the token doesn't appear in the vocabulary at least once
            if token not in self.stoi.keys():
                self.itos[self.__len__()] = token
                self.stoi[token] = self.__len__()
                self.freq[token] = 1
            else:
                self.freq[token] += 1
        
        # sort by frequency in 'descending' order
        self.freq = dict(sorted(self.freq.items(), key=lambda x: x[1], reverse=True))

    def __len__(self):
        return len(self.stoi)

class Tokenizer(object):
    def __init__(self, tokenization_fn, vocab=None):
        self.tokenization_fn = tokenization_fn
        self.vocab = vocab
        
    def tokenize(self, text):
        tokens = self.tokenization_fn(text)
        return tokens
    
    def transform(self, tokens):
        if self.vocab:
            return [self.vocab.stoi[token] for token in tokens]