# Pre-training For Language-Understanding
[![LICENSE](https://img.shields.io/github/license/lyeoni/pretraining-for-language-understanding?style=flat-square)](https://github.com/lyeoni/pretraining-for-language-understanding/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/lyeoni/pretraining-for-language-understanding?style=flat-square&color=yellow)](https://github.com/lyeoni/pretraining-for-language-understanding/issues)
[![GitHub stars](https://img.shields.io/github/stars/lyeoni/pretraining-for-language-understanding?style=flat-square&color=important)](https://github.com/lyeoni/pretraining-for-language-understanding/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/lyeoni/pretraining-for-language-understanding?style=flat-square&color=blueviolet)](https://github.com/lyeoni/pretraining-for-language-understanding/network/members)

Generalized Pre-training for Language Understanding

## Overview
### Language Model
A Lanugage Model (LM) captures **the distribution over all possible sentences**.
- Input : a sentence
- Output : the probability of the input sentence

While language modeling is a typical _unsupervised learning_ on large corpus, we turn this into a _sequence of supervised learning_ in this repo.

#### Autoregressive Language Model
<br>
<p align="center">
<img width="500" src="https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig2-Anim-160908-r01.gif" align="middle">
</p>

Autoregressive language model captures the distribution over the next token is based on all the previous token. In other words, it looks at the previous token, and predicts the next token. The objective of Autoregressive language model is expressed in a formula as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;input\;&space;sentence&space;:&space;x&space;=&space;(x_{1},&space;x_{2},...,&space;x_{t})" title="input\; sentence : x = (x_{1}, x_{2},..., x_{t})" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;likelihood&space;:&space;p(X)&space;=&space;p(x_{1})p(x_{2}|x_{1})\cdots&space;p(x_{t}|x_{1},...x_{t-1})&space;=&space;\prod_{t=1}^{T}p(x_{t}|&space;x_{<t})" title="likelihood : p(X) = p(x_{1})p(x_{2}|x_{1})\cdots p(x_{t}|x_{1},...x_{t-1}) = \prod_{t=1}^{T}p(x_{t}| x_{<t})" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;objtective:&space;\underset{\theta}{max}\;&space;logp_{\theta}(x)&space;=&space;\underset{\theta}{max}\;&space;\sum_{t=1}^{T}log(p(x_{t}|x_{<t}))" title="objtective: \underset{\theta}{max}\; logp_{\theta}(x) = \underset{\theta}{max}\; \sum_{t=1}^{T}log(p(x_{t}|x_{<t}))" />
</p>

Because Autoregressive language model should be forward or backward, only one-way uni-directional context information can be used. Therefore, it's difficult to understand the context in both directions simultaneously.

RNNLM, ELMo, GPT are typically the case, and **Unidirectional/Bidirectional LSTM language models are covered in this repo.**

cf. Bidirectional LSTM LM, ELMo use context in both directions. However, only shallow understanding is possible beacuase it use contexts that are independently learned in each direction.

## 1. Build Corpus

### Wikipedia
Wikipedia regularly distributes the entire document. You can download Korean Wikipedia dump [here](https://dumps.wikimedia.org/kowiki/) (and English Wikipedia dump [here](https://dumps.wikimedia.org/enwiki/)).
Wikipedia recommends using `pages-articles.xml.bz2`, which includes only the latest version of the entire document, and is approximately 600 MB compressed (for English, `pages-articles-multistream.xml.bz2`).

You can use `wikipedia_ko.sh` script to download the dump on the latest Korean Wikipedia document. For English, use `wikipedia_en.sh`

example:
```
$ cd build_corpus
$ chmod 777 wikipedia_ko.sh
$ ./wikipedia_ko.sh
```

The downloaded dump using above shell script is in XML format, and we need to parse XML to text file. The Python script `WikiExtractor.py` in [attardi/wikiextractor](https://github.com/attardi/wikiextractor) repo, extracts and cleans text from the dump.

example:
```
$ git clone https://github.com/attardi/wikiextractor
$ python wikiextractor/WikiExtractor.py kowiki-latest-pages-articles.xml

$ head -n 4 text/AA/wiki_02
<doc id="577" url="https://ko.wikipedia.org/wiki?curid=577" title="천문학">
천문학

천문학(天文學, )은 별이나 행성, 혜성, 은하와 같은 천체와, 지구 대기의 ..
</doc>
```

The extracted text is saved as text file of a certain size. To combine these, use `build_corpus.py`. The output `corpus.txt` contains _4,277,241 sentences, 55,568,030 words_.

example:
```
$ python build_corpus.py > corpus.txt
$ wc corpus.txt 
4277241  55568030 596460787 corpus.txt
```

Now, you need to split the corpus to train-set and test-set.

```
$ cat corpus.txt | shuf > corpus.shuf.txt
$ head -n 855448 corpus.shuf.txt > corpus.test.txt
$ tail -n 3421793 corpus.shuf.txt > corpus.train.txt
$ wc -l corpus.train.txt corpus.test.txt
  3421793 corpus.train.txt
   855448 corpus.test.txt
  4277241 합계
```
 
## 2. Preprocessing

### Build Vocab
Our corpus `corpus.txt` has 55,568,030 words, and 608,221 unique words. If the minimum frequency needed to include a token in the vocabulary is set to 3, the vocabulary contains **_297,773_** unique words.

Here we use the train corpus `corpus.train.txt` to build vocabulary.
The vocabulary built by train corpus contains **_557,627_** unique words, and **_271,503_** unique words that appear at least three times.

example:
```
$ python build_vocab.py --corpus build_corpus/corpus.train.txt --vocab vocab.train.pkl --min_freq 3 --lower
Namespace(bos_token='<bos>', corpus='build_corpus/corpus.train.txt', eos_token='<eos>', is_tokenized=False, lower=True, min_freq=3, pad_token='<pad>', tokenizer='mecab', unk_token='<unk>', vocab='vocab.train.pkl')
Vocabulary size:  271503
Vocabulary saved to vocab.train.pkl
```

Since the vocabulary file is too large(~1.3GB) to upload on this repo, I uploaded it to Google Drive.
- `vocab.train.pkl` : [[download]](https://drive.google.com/file/d/195kdXPQtiG0eqppH-L2VKoHAcgqCSR1l/view?usp=sharing)

## 3. Training

```
$ python lm_trainer.py -h
usage: lm_trainer.py [-h] --train_corpus TRAIN_CORPUS --vocab VOCAB
                     --model_type MODEL_TYPE [--test_corpus TEST_CORPUS]
                     [--is_tokenized] [--tokenizer TOKENIZER]
                     [--max_seq_len MAX_SEQ_LEN] [--multi_gpu] [--cuda CUDA]
                     [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                     [--clip_value CLIP_VALUE] [--shuffle SHUFFLE]
                     [--embedding_size EMBEDDING_SIZE]
                     [--hidden_size HIDDEN_SIZE] [--n_layers N_LAYERS]
                     [--dropout_p DROPOUT_P]

optional arguments:
  -h, --help            show this help message and exit
  --train_corpus TRAIN_CORPUS
  --vocab VOCAB
  --model_type MODEL_TYPE
                        Model type selected in the list: LSTM, BiLSTM
  --test_corpus TEST_CORPUS
  --is_tokenized        Whether the corpus is already tokenized
  --tokenizer TOKENIZER
                        Tokenizer used for input corpus tokenization
  --max_seq_len MAX_SEQ_LEN
                        The maximum total input sequence length after
                        tokenization
  --multi_gpu           Whether to training with multiple GPU
  --cuda CUDA           Whether CUDA is currently available
  --epochs EPOCHS       Total number of training epochs to perform
  --batch_size BATCH_SIZE
                        Batch size for training
  --clip_value CLIP_VALUE
                        Maximum allowed value of the gradients. The gradients
                        are clipped in the range
  --shuffle SHUFFLE     Whether to reshuffle at every epoch
  --embedding_size EMBEDDING_SIZE
                        Word embedding vector dimension
  --hidden_size HIDDEN_SIZE
                        Hidden size of LSTM
  --n_layers N_LAYERS   Number of layers in LSTM
  --dropout_p DROPOUT_P
                        Dropout rate used for dropout layer in LSTM
```

example:
```
$ python lm_trainer.py --train_corpus build_corpus/corpus.train.txt --vocab vocab.train.pkl --model_type LSTM --batch_size 16
```

You can select your own parameter values via argument inputs. 

### Training with multiple GPU

Training a model with single GPU is not only very slow, it also limits adjusting batch size, model size, and so on.
To accelerate model training with multiple GPU and use large model, what you have to do is to include `--multi_gpu` flag like belows. For more details, please check [here](https://github.com/lyeoni/pretraining-for-language-understanding/blob/master/parallel.py).

#### Training Unidiretional LSTM Language Model
This example code trains unidirectional-LSTM model on the Wikipedia corpus using parallel training on 8 * V100 GPUs.

```
$ python lm_trainer.py --train_corpus build_corpus/corpus.train.txt --vocab vocab.train.pkl --model_type LSTM --multi_gpu
Namespace(batch_size=512, clip_value=10, cuda=True, dropout_p=0.2, embedding_size=256, epochs=10, hidden_size=1024, is_tokenized=False, max_seq_len=32, model_type='LSTM', multi_gpu=True, n_layers=3, shuffle=True, test_corpus=None, tokenizer='mecab', train_corpus='build_corpus/corpus.train.txt', vocab='vocab.train.pkl')
=========MODEL=========
 DataParallelModel(
  (module): LSTMLM(
    (embedding): Embedding(271503, 256)
    (lstm): LSTM(256, 1024, num_layers=3, batch_first=True, dropout=0.2)
    (fc): Linear(in_features=1024, out_features=512, bias=True)
    (fc2): Linear(in_features=512, out_features=271503, bias=True)
    (softmax): LogSoftmax()
  )
)
```

#### Training Bidirectional LSTM Language Model
This example code trains Bidirectional-LSTM model on the Wikipedia corpus using parallel training on 8 * V100 GPUs.

```
$ python lm_trainer.py --train_corpus build_corpus/corpus.train.txt --vocab vocab.train.pkl --model_type BiLSTM --n_layers 1 --multi_gpu
Namespace(batch_size=512, clip_value=10, cuda=True, dropout_p=0.2, embedding_size=256, epochs=10, hidden_size=1024, is_tokenized=False, max_seq_len=32, model_type='BiLSTM', multi_gpu=True, n_layers=1, shuffle=True, test_corpus=None, tokenizer='mecab', train_corpus='build_corpus/corpus.train.txt', vocab='vocab.train.pkl')
=========MODEL=========
 DataParallelModel(
  (module): BiLSTMLM(
    (embedding): Embedding(271503, 256)
    (lstm): LSTM(256, 1024, batch_first=True, dropout=0.2, bidirectional=True)
    (fc): Linear(in_features=2048, out_features=1024, bias=True)
    (fc2): Linear(in_features=1024, out_features=512, bias=True)
    (fc3): Linear(in_features=512, out_features=271503, bias=True)
    (softmax): LogSoftmax()
  )
)
```

## 4. Evaluation

### Perplexity

A language model captures the distribution over all possible sentences. And, the best language model is one that the best predicts an unseen sentence. Perplexty is a very common measurement of how well a probability distribution predicts unseen sentences.

**_Perplexity_** : _inverse probability of the given sentence, normalized by the number of words (by taking geometric mean)_ 

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\fn_jvn&space;PP(W)&space;=&space;P(w_{1},&space;w_{2}...w_{n})^{-\frac{1}{n}}&space;=\sqrt[n]{\frac{1}{P(w_{1}w_{2}...w_{N})}}" title="PP(W) = P(w_{1}, w_{2}...w_{n})^{-\frac{1}{n}} =\sqrt[n]{\frac{1}{P(w_{1}w_{2}...w_{N})}}" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\fn_jvn&space;Chain\;&space;rule:\;&space;PP(W)&space;=&space;\sqrt[n]{\prod_{i=1}^{N}\frac{1}{P(w_{i}|w_{1}...w_{i-1})}}" title="Chain\; rule:\; PP(W) = \sqrt[n]{\prod_{i=1}^{N}\frac{1}{P(w_{i}|w_{1}...w_{i-1})}}" />
</p>

As you can see from the above equation, perplexity is defined as the exponentiated negative average log-likelihood. In other words, maximizing probability is the same as minimizing perplexity.

### Results

And now, perplexity is the metric that we're going to be using.
A low perplexity indicates that the probability distribution is good at predicting the sentence. 

|Model|Loss|Perplexity|
|-|-:|-:|
|Unidirectional-LSTM|3.496|33.037|
|Bidirectional-LSTM|||

<br>


## Reference

### General
- [Google DeepMind] [WaveNet: A Generative Model for Raw Audio](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
- [Dan Jurafsky] [CS 124: From Languages to Information at Stanford](https://web.stanford.edu/class/cs124/lec/languagemodeling2019.pdf)
- [attardi/wikiextractor] [WikiExtractor](https://github.com/attardi/wikiextractor)

### Models

#### Unidirectiaonl LSTM LM
- [DSKSD] [6. Recurrent Neural Networks and Language Models](https://nbviewer.jupyter.org/github/DSKSD/DeepNLP-models-Pytorch/blob/master/notebooks/06.RNN-Language-Model.ipynb)
- [yunjey/pytorch-tutorial] [Language Model (RNN-LM)](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py)
- [pytorch/examples] [Word-level language modeling RNN](https://github.com/pytorch/examples/tree/master/word_language_model)

#### Bidirectional LSTM LM
- [Mousa, Amr, and Björn Schuller] [Contextual Bidirectional Long Short-Term Memory Recurrent Neural Network Language Models:A Generative Approach to Sentiment Analysis](https://www.aclweb.org/anthology/E17-1096)
- [Motoki Wu] [The Bidirectional Language Model](https://medium.com/@plusepsilon/the-bidirectional-language-model-1f3961d1fb27)

### Multi GPU Training
- [matthew l][PyTorch Multi-GPU 제대로 학습하기](https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b)
- [zhanghang1989/PyTorch-Encoding] [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
, [Issue: How to use the DataParallelCriterion, DataParallelModel](https://github.com/zhanghang1989/PyTorch-Encoding/issues/54)
