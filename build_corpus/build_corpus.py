import os, sys
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

WIKI='./text'

if __name__=='__main__':
    for path in tqdm(os.listdir(WIKI)):
        for sub_path in os.listdir(os.path.join(WIKI, path)):
            _corpus = os.path.join(WIKI, path, sub_path)
            with open(_corpus, 'r', encoding='utf8') as reader:
                for li, line in enumerate(reader):
                    if line.find('<doc')!=-1: # beginning of document
                        is_title=True
                        continue
                    if line.find('</doc')!=-1: # end of document
                         continue
                    if is_title: # string containing white-space only
                        is_title=False
                        continue
                    if line.strip() !='':
                        sentences = sent_tokenize(line.strip())
                        for sentence in sentences:
                            sys.stdout.write(sentence+'\n')