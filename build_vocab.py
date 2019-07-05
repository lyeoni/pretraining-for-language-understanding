import argparse
import pickle
from tokenizer import Vocab, Tokenizer

def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--corpus', required=True)
    p.add_argument('--vocab', required=True)
    p.add_argument('--is_tokenized', action='store_true')
    p.add_argument('--tokenizer', default='mecab')

    config = p.parse_args()
    return config

def main(config):
    list_of_tokens = []
    if config.is_tokenized:
        # read tokens
        with open(config.corpus, 'r', encoding='utf8') as reader:
            for li, line in enumerate(reader):
                list_of_tokens += line.strip().split()
    else:
        # select tokenizer
        if config.tokenizer=='mecab':
            from konlpy.tag import Mecab
            tokenizer = Tokenizer(tokenization_fn=Mecab().morphs)

        # tokenization & read tokens
        with open(config.corpus, 'r', encoding='utf8') as reader:
            for li, line in enumerate(reader):
                list_of_tokens += tokenizer.tokenize(line.strip())

    # build vocabulary                
    vocab = Vocab(list_of_tokens=list_of_tokens)
    vocab.build()
    print('Vocabulary size: ', len(vocab))

    # save vocabulary
    with open(config.vocab, 'wb') as writer:
        pickle.dump(vocab, writer)
    print('Vocabulary saved to ', config.vocab)

if __name__=='__main__':
    config = argparser()
    main(config)