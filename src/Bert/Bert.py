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
    return output

def main():
    data = pickle.load(open("data.p","rb"))
    model = trainModel(data)
    vector = getVector(model,"summary CallExpression ExpressionStatement BlockStatement FunctionDeclaration Program")
    os.system("clear")
    print(vector)


if __name__ == '__main__':
    main()
