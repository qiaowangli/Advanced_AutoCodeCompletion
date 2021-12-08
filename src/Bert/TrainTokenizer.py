from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
from alive_progress import alive_bar

import numpy as np
import os
import pickle

def makeSenteces(rawInput):
    print("Compiling sequnces into sentences")
    with open("Sentences.txt","w") as f:
        for l in rawInput:
            f.write(' '.join(l) + '\n')
            yield

def traintokenizer():
    print("Training tokenizer")
    tokenizer = BertWordPieceTokenizer(handle_chinese_chars=False,
                                strip_accents=False,
                                lowercase=False)
    tokenizer.train(files=["Sentences.txt"])
    try:
        os.mkdir("CodeTokenizer")
    except:
        pass
    tokenizer.save_model("./CodeTokenizer")

def cleanVocab(rawInput):
    print("Cleaning vocab")
    keepers = [l[0] for l in rawInput]
    keepers = list(dict.fromkeys(keepers))

    with open("words.txt",'w') as f:
        for word in keepers:
            f.write(word + '\n')

    with open("./CodeTokenizer/vocab.txt",'r') as f:
        oldVocab = f.read().split("\n")
    finalVocab = oldVocab[:129] + keepers + oldVocab[129:]
    finalVocab = list(dict.fromkeys(finalVocab))
    with open("./CodeTokenizer/vocab.txt",'w') as f:
        f.write('\n'.join(finalVocab))

def compileTokenizer(rawInput):
    print('''
        ████████╗██████╗░░█████╗░██╗███╗░░██╗██╗███╗░░██╗░██████╗░
        ╚══██╔══╝██╔══██╗██╔══██╗██║████╗░██║██║████╗░██║██╔════╝░
        ░░░██║░░░██████╔╝███████║██║██╔██╗██║██║██╔██╗██║██║░░██╗░
        ░░░██║░░░██╔══██╗██╔══██║██║██║╚████║██║██║╚████║██║░░╚██╗
        ░░░██║░░░██║░░██║██║░░██║██║██║░╚███║██║██║░╚███║╚██████╔╝
        ░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝╚═╝░░╚══╝╚═╝╚═╝░░╚══╝░╚═════╝░

        ████████╗░█████╗░██╗░░██╗███████╗███╗░░██╗██╗███████╗███████╗██████╗░
        ╚══██╔══╝██╔══██╗██║░██╔╝██╔════╝████╗░██║██║╚════██║██╔════╝██╔══██╗
        ░░░██║░░░██║░░██║█████═╝░█████╗░░██╔██╗██║██║░░███╔═╝█████╗░░██████╔╝
        ░░░██║░░░██║░░██║██╔═██╗░██╔══╝░░██║╚████║██║██╔══╝░░██╔══╝░░██╔══██╗
        ░░░██║░░░╚█████╔╝██║░╚██╗███████╗██║░╚███║██║███████╗███████╗██║░░██║
        ░░░╚═╝░░░░╚════╝░╚═╝░░╚═╝╚══════╝╚═╝░░╚══╝╚═╝╚══════╝╚══════╝╚═╝░░╚═╝
    ''')

    with alive_bar(len(rawInput)) as bar:
        for i in makeSenteces(rawInput):
            bar()
    traintokenizer()
    cleanVocab(rawInput)
    os.system("clear")

def main():
    compileTokenizer()

if __name__ == '__main__':
    main()
