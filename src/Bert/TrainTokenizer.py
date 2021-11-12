from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing

import numpy as np
import pickle
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
    return tokenizer

def compileTokenizer(rawInput):
    makeSenteces(rawInput)
    tokenizer = traintokenizer()
    return tokenizer

def main():
    compileTokenizer()

if __name__ == '__main__':
    main()
