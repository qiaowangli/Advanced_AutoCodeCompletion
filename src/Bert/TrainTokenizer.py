from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing

import numpy as np
import os

def makeSenteces(rawInput):
    with open("Sentences.txt","w") as f:
        for l in rawInput:
            f.write(' '.join(l) + '\n')

def traintokenizer():
    tokenizer = BertWordPieceTokenizer(handle_chinese_chars=False,
                                strip_accents=False,
                                lowercase=False)
    tokenizer.train(files=["Sentences.txt"])
    os.system("mkdir CodeTokenizer")
    tokenizer.save_model("./CodeTokenizer")

def cleanVocab(rawInput):
    buffer = [l[0] for l in rawInput]
    keepers = []
    for word in buffer:
        if word not in keepers:
            keepers.append(word)
    with open("./CodeTokenizer/vocab.txt",'r') as f:
        oldVocab = f.read().split("\n")
    finalVocab = oldVocab[:129] + keepers + oldVocab[129:]
    finalVocab = list(dict.fromkeys(finalVocab))
    with open("./CodeTokenizer/vocab.txt",'w') as f:
        f.write('\n'.join(finalVocab))


def compileTokenizer(rawInput):
    makeSenteces(rawInput)
    traintokenizer()
    cleanVocab(rawInput)

def main():
    compileTokenizer()

if __name__ == '__main__':
    main()
