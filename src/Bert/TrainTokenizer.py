from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing

import numpy as np
import pickle
import os

def mainSenteces(rawInput):
    with open("Sentences","w") as f:
        for l in rawInput:
            f.write(' '.join(l) + '\n')

def trainTokenazer():
    tokenizer = BertWordPieceTokenizer(handle_chinese_chars=False,
                                strip_accents=False,
                                lowercase=False)
    tokenizer.train(files=["Sentences.txt"])
    tokenizer.enable_truncation(max_length=512)
    os.system("mkdir CodeTokenizer")
    tokenizer.save_model("./CodeTokenizer")

def main():
    trainTokenazer()


if __name__ == '__main__':
    main()
