import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from tokenization import Tokenizer, Vocab
from dataset_utils import Corpus
from models import LSTMLM, BiLSTMLM

def argparser():
    p = argparse.ArgumentParser()

    # Required parameters
    p.add_argument('--train_corpus', default=None, type=str, required=True)
    p.add_argument('--vocab', default=None, type=str, required=True)
    p.add_argument('--model_type', default=None, type=str, required=True,
                   help='Model type selected in the list: LSTM, BiLSTM')

    p.add_argument('--test_corpus', default=None, type=str)

    # Input parameters
    p.add_argument('--is_tokenized', action='store_true',
                   help='Whether the corpus is already tokenized')
    p.add_argument('--tokenizer', default='mecab', type=str,
                   help='Tokenizer used for input corpus tokenization')
    p.add_argument('--max_seq_len', default=32, type=int,
                   help='The maximum total input sequence length after tokenization')

    # Train parameters
    p.add_argument('--multi_gpu', action='store_true',
                   help='Whether to training with multiple GPU')
    p.add_argument('--cuda', default=True, type=bool,
                   help='Whether CUDA is currently available')
    p.add_argument('--epochs', default=10, type=int,
                   help='Total number of training epochs to perform')
    p.add_argument('--batch_size', default=512, type=int,
                   help='Batch size for training')
    p.add_argument('--clip_value', default=10, type=int,
                   help='Maximum allowed value of the gradients. The gradients are clipped in the range')
    p.add_argument('--shuffle', default=True, type=bool, 
                   help='Whether to reshuffle at every epoch')
        
    # Model parameters
    p.add_argument('--embedding_size', default=256, type=int,
                   help='Word embedding vector dimension')
    p.add_argument('--hidden_size', default=1024, type=int,
                   help='Hidden size of LSTM')
    p.add_argument('--n_layers', default=3, type=int,
                   help='Number of layers in LSTM')
    p.add_argument('--dropout_p', default=.2, type=float,
                   help='Dropout rate used for dropout layer in LSTM')

    config = p.parse_args()
    return config

def train():
    n_batches, n_samples = len(train_loader), len(train_loader.dataset)

    model.train()
    total_loss, total_ppl = 0, 0
    for iter_, batch in enumerate(tqdm(train_loader)):
        inputs, targets = batch
        # |inputs|, |targets| = (batch_size, seq_len), (batch_size, seq_len)
        
        preds = model(inputs)
        # |preds| = (batch_size, seq_len, len(vocab))
        
        if config.multi_gpu:
            # If the model run parallelly using DataParallelModel,the output tensor size is as follows.
            # |preds| = [(batch_size/n_gpus, seq_len, len(vocab))] * n_gpus
            n_gpus = len(preds)
            preds = [pred.view(-1, len(vocab)).contiguous() for pred in preds]
            # |preds| = [(batch_size/n_gpus*seq_len, len(vocab))] * n_gpus
        else:
            preds = preds.view(-1, len(vocab)).contiguous()    
            # |preds| = (batch_size*seq_len, len(vocab))

        targets = targets.view(-1).contiguous()
        # |targets| = (batch_size*seq_len)
        
        loss = loss_fn(preds, targets)
        total_loss += loss.item()
        total_ppl += np.exp(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.clip_value)
        optimizer.step()
        
        if iter_ % (n_batches//300) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.4f} \tPerplexity: {:.5f}'.format(
                    epoch, iter_, n_batches, 100.*iter_/n_batches, loss.item(), np.exp(loss.item())))

    print('====> Train Epoch: {} Average loss: {:.4f} \tPerplexity: {:.5f}'.format(
            epoch, total_loss/n_batches, total_ppl/n_batches))
    
    # Save model
    torch.save(model.state_dict(), '{}_lm{}.pth'.format(config.model_type.lower(), epoch))

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
    train_loader = DataLoader(dataset=Corpus(corpus_path=config.train_corpus,
                                             tokenizer=tokenizer,
                                             model_type=config.model_type,
                                             cuda=config.cuda),
                              batch_size=config.batch_size,
                              shuffle=config.shuffle,
                              drop_last=True)
    if config.test_corpus:
        test_loader = DataLoader(dataset=Corpus(corpus_path=config.test_corpus,
                                                tokenizer=tokenizer,
                                                model_type=config.model_type,
                                                cuda=config.cuda),
                                 batch_size=config.batch_size,
                                 shuffle=config.shuffle,
                                 drop_last=True)

    # Build model
    if config.model_type=='LSTM':
        model = LSTMLM(input_size=len(vocab),
                       embedding_size=config.embedding_size,
                       hidden_size=config.hidden_size,
                       output_size=len(vocab),
                       n_layers=config.n_layers,
                       dropout_p=config.dropout_p)
    elif config.model_type=='BiLSTM':
        model = BiLSTMLM(input_size=len(vocab),
                         embedding_size=config.embedding_size,
                         hidden_size=config.hidden_size,
                         output_size=len(vocab),
                         n_layers=config.n_layers,
                         dropout_p=config.dropout_p)
        
    loss_fn = nn.NLLLoss(ignore_index=vocab.stoi[vocab.pad_token])
    optimizer = optim.Adam(model.parameters())
    
    if config.cuda:
        if config.multi_gpu:
            from parallel import DataParallelModel, DataParallelCriterion
            model = DataParallelModel(model).cuda()
            loss_fn = DataParallelCriterion(loss_fn).cuda()
        else:
            model = model.cuda()
            loss_fn = loss_fn.cuda()
    print('=========MODEL=========\n',model)

    # Train
    for epoch in range(1, config.epochs+1):
        train()