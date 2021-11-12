import TrainTokenizer
import pickle
import TrainModel
import torch
import os

from transformers import BertTokenizer

def trainModel(data):
    TrainTokenizer.compileTokenizer(data)
    model = TrainModel.compileModel(data)
    return model

def getVector(model,sentence = "summary CallExpression ExpressionStatement BlockStatement FunctionDeclaration Program"):
    tokenizer = BertTokenizer.from_pretrained('./CodeTokenizer')
    x = tokenizer(sentence)["input_ids"]
    input_ids = torch.tensor(x).unsqueeze(0)
    output = model(input_ids)
    return output[0]

def main():
    '''
    the data is a list of sequences
    a sequnce is a list of words (node name)
    right now it's loading from a saved datastructre, it would be parsed in in the future

    alot of messy output that doesn't effect the performance and can only be
    removed by editing the huggingface source code.
    '''
    data = pickle.load(open("data.p","rb"))
    model = trainModel(data)
    vector = getVector(model,"summary CallExpression ExpressionStatement BlockStatement FunctionDeclaration Program")
    os.system("clear")
    print(vector)
    print(vector.size())

if __name__ == '__main__':
    main()
