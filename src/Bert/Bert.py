import TrainTokenizer
import TrainModel
import FineTuneModel

import pickle
import sys
import torch
import os
import numpy as np
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline

def FindMsk(model,sentence,words,topn=5,echo=True):
    #reference https://gist.github.com/yuchenlin/a2f42d3c4378ed7b83de65c7a2222eb2
    tokenizer = BertTokenizer.from_pretrained('./CodeTokenizer')

    sentence = "[CLS] %s [SEP]"%sentence
    tokenized_text = tokenizer.tokenize(sentence)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, 50, sorted=True)

    results = []
    weights = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        results.append(predicted_token)
        weights.append(float(token_weight))

    reList = []
    for i in range(len(results)):
        if(topn == 0):
            break
        word = results[i]
        #specials = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
        #if word not in specials and word[0] != '#' and len(word) != 1:
        if word in words:
            if(echo):
                print("[MASK]: '%s'"%word, " | weights:", weights[i])
            reList.append(word)
            topn -= 1
    return reList

def validate(model,words,testIndex=0,topn=5):
    input = []
    with open("Sentences.txt","r") as f:
        input = f.read().split('\n')

    total = len(input)
    correctPrediction = 0
    for s in input:
        sl = s.split(' ')
        correct = sl[testIndex]
        sl[testIndex] = "[MASK]"
        s = ' '.join(sl)
        predictions = FindMsk(model,s,words,topn=topn,echo=True)
        if(correct in predictions):
            correctPrediction += 1
    return float(correctPrediction/total)

def loadModel():
    model = pickle.load(open("model.p","rb"))
    tokenizer = pickle.load(open("tokenizer.p","rb"))
    return model, tokenizer

def trainBertModel(data):
    TrainTokenizer.compileTokenizer(data)
    model = TrainModel.makeModel()
    model = TrainModel.compileModel(model,data,MAX_LEN=512,epochCount=16)
    tokenizer = BertTokenizer.from_pretrained('./CodeTokenizer')
    model = FineTuneModel.fineTune(model,MAX_LEN=512,epochCount=16)
    model.eval()

    pickle.dump(model,open("model.p","wb"))
    pickle.dump(tokenizer,open("tokenizer.p","wb"))

    return model

def testBertModel():
    model, tokenizer = loadModel()
    model.eval()
    words = pickle.load(open("words.p","rb"))
    testIndex = 0
    r = validate(model,words,testIndex,topn=10)
    print(r)

def main():
    '''
    the data is a list of sequences
    a sequnce is a list of words (node name)
    right now it's loading from a saved datastructre, it would be parsed in in the future
    '''
    data = pickle.load(open("fullList.p","rb"))
    model = trainBertModel(data)

if __name__ == '__main__':
    if(sys.argv[1] == "1"):
        main()
    if(sys.argv[1] == "2"):
        testBertModel()
