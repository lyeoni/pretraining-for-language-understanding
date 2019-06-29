# Pre-training For Language-Understanding
Generalized Pre-training for Language Understanding 

## Corpus

### Wikipedia 
Wikipedia regularly distributes the entire document. 
You can download English Wikipedia dump [here](https://dumps.wikimedia.org/enwiki/)(and Korean Wikipedia dump [here](https://dumps.wikimedia.org/kowiki/)).
Wikipedia recommends using `pages-articles-multistream.xml.bz2`, which includes only the latest version of the entire document, and is approximately 14 GB compressed
(for korean, `pages-articles.xml.bz2`).

You can use `wikipedia_en.sh` script to download the dumps on the latest English Wikipedia document. For Korean, use `wikipedia_ko.sh`

example:
```
$ ./wikipedia_en.sh
```