import TrainTokenizer
import TrainModel
import FineTuneModel

import pickle

import torch
import os
import numpy as np
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline

def trainModel(data):
    TrainTokenizer.compileTokenizer(data)
    model = TrainModel.compileModel(data)
    return model

def getVector(model,sentence = "summary CallExpression ExpressionStatement BlockStatement FunctionDeclaration Program"):
    x = tokenizer(sentence)["input_ids"]
    input_ids = torch.tensor(x).unsqueeze(0)
    output = model(input_ids)
    return output

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def TopK(x, k):
    a = dict([(i, j) for i, j in enumerate(x)])
    sorted_a = dict(sorted(a.items(), key = lambda kv:kv[1], reverse=True))
    indices = list(sorted_a.keys())[:k]
    values = list(sorted_a.values())[:k]
    #return (indices, values)
    return indices

def findFirstMask(model,sentence = "summary CallExpression ExpressionStatement BlockStatement FunctionDeclaration Program",topn = 5):
    sentence = sentence.split()
    correct = sentence[0]
    sentence[0] = "[MASK]"
    sentence = ' '.join(sentence)
    tokenizer = BertTokenizer.from_pretrained('./CodeTokenizer')
    x = tokenizer(sentence)["input_ids"]
    x = torch.tensor(x).unsqueeze(0)
    output = model(x)
    maskLogits = output[0][0][1]
    npLogits = (maskLogits.detach().numpy())
    top5 = TopK(npLogits,topn)
    with open("./CodeTokenizer/vocab.txt","r") as f:
        lines = f.readlines()
    results = [lines[i].strip() for i in top5]
    print("Correct output:",correct)
    for i in range(len(results)):
        print("Number " + str(i) + ": " + results[i])
    return results

def loadModel():
    model = pickle.load(open("model.p"),"rb")
    tokenizer = pickle.load(open("tokenizer.p"),"rb")
    return model, tokenizer

def trainBertModel(data):
    model = trainModel(data)
    tokenizer = BertTokenizer.from_pretrained('./CodeTokenizer')
    model = FineTuneModel.fineTune(model)
    model.eval()

    pickle.dump(model,open("model.p","wb"))
    pickle.dump(tokenizer,open("tokenizer.p","wb"))

    return model

def main():
    '''
    the data is a list of sequences
    a sequnce is a list of words (node name)
    right now it's loading from a saved datastructre, it would be parsed in in the future
    '''
    data = pickle.load(open("data.p","rb"))
    # for l in data:
    #     print(l)
    model = trainBertModel(data)
    os.system("clear")
    findFirstMask(model)

if __name__ == '__main__':
    main()
