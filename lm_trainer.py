import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tokenization import Tokenizer, Vocab
from dataset_utils import Corpus
from torch.utils.data import DataLoader
from models import LSTMLM
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def argparser():
    p = argparse.ArgumentParser()

    # Required parameters
    p.add_argument('--train_corpus', required=True, type=str)
    p.add_argument('--test_corpus', required=True, type=str)
    p.add_argument('--vocab', required=True, type=str)
    p.add_argument('--model_type', required=True, type=str,
                   help='Model type selected in the list: LSTM')

    # Input parameters
    p.add_argument('--is_tokenized', action='store_true',
                   help='Whether the corpus is already tokenized')
    p.add_argument('--tokenizer', default='mecab', type=str,
                   help='Tokenizer used for input corpus tokenization')
    p.add_argument('--max_seq_len', default=32, type=int,
                   help='The maximum total input sequence length after tokenization')

    # Train parameters
    p.add_argument('--epochs', default=10, type=int,
                   help='Total number of training epochs to perform')
    p.add_argument('--batch_size', default=16, type=int,
                   help='Batch size for training')
    p.add_argument('--shuffle', action='store_true',
                   help='Wheter to shuffle input data')

    # Model parameters
    p.add_argument('--embedding_size', default=256, type=int,
                   help='Word embedding vector dimension')
    p.add_argument('--hidden_size', default=512, type=int,
                   help='Hidden size of LSTM')
    p.add_argument('--n_layers', default=3, type=int,
                   help='Number of layers in LSTM')
    p.add_argument('--dropout_p', default=.2, type=float,
                   help='Dropout rate used for dropout layer in LSTM')
    p.add_argument('--is_bidirectional', action='store_true',
                   help='Whether to use bidirectional LSTM')

    config = p.parse_args()
    return config

def train():
    model.train()
    total_loss = 0

    for bi, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        # |inputs|, |targets| = (batch_size, seq_len)

        # Take feed-forward
        outputs = model(inputs)
        # |outputs| = (batch_size, seq_len, vocab_len)
        
        loss = 0
        for i in range(config.max_seq_len-1):
            loss += criterion(outputs[:,i], targets[:,i])
        else:
            loss /= config.batch_size
        
            total_loss += loss.item()

        loss.backward()
        optimizer.step()
        
        if bi % (len(train_loader)//300) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format(
                epoch, bi*len(inputs), len(train_loader.dataset), 100.*bi/len(train_loader),
                loss.item()))

    print('====> Train Epoch: {} Average loss: {:.4f}\t'.format(
        epoch, total_loss / len(train_loader)))
    
    # Save model
    torch.save(model.state_dict(), 'rnn_lm{}.pth'.format(epoch))

if __name__=='__main__':
    config = argparser()
    print(config)

    # Load vocabulary
    import pickle
    with open(config.vocab, 'rb') as reader:
        vocab = pickle.load(reader)

    # Select tokenizer
    if config.tokenizer=='mecab':
        from konlpy.tag import Mecab
        tokenizer = Tokenizer(tokenization_fn=Mecab().morphs,
                              vocab=vocab, max_seq_length=config.max_seq_len)
    
    # Build dataloader
    train_corpus = Corpus(corpus_path=config.train_corpus, tokenizer=tokenizer)   
    train_loader = DataLoader(dataset=train_corpus,
                              batch_size=config.batch_size,
                              shuffle=config.shuffle)
    test_corpus = Corpus(corpus_path=config.test_corpus, tokenizer=tokenizer)   
    test_loader = DataLoader(dataset=test_corpus, batch_size=config.batch_size)

    # Build model
    if config.model_type=='LSTM':
        model = LSTMLM(input_size=len(vocab),
                       embedding_size=config.embedding_size,
                       hidden_size=config.hidden_size,
                       output_size=len(vocab),
                       n_layers=config.n_layers,
                       dropout_p=config.dropout_p,
                       is_bidirectional=config.is_bidirectional,
                       device=device).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss(ignore_index=vocab.stoi[vocab.pad_token], reduction='sum')
    print('=========MODEL=========\n',model)

    # Train
    for epoch in range(1, config.epochs+1):
        train()