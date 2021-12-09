import TrainTokenizer
import TrainModel
import FineTuneModel
import ValidationFunctions

import pickle
import sys
import torch
import os
import numpy as np
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline

def loadModel():
    model = pickle.load(open("model.p","rb"))
    tokenizer = pickle.load(open("tokenizer.p","rb"))
    return model, tokenizer

def trainBertModel(fileName):
    data = pickle.load(open(fileName,"rb"))

    TrainTokenizer.compileTokenizer(data)
    tokenizer = BertTokenizer.from_pretrained('./CodeTokenizer')
    model = TrainModel.trainModel(data)
    model = FineTuneModel.fineTune(model)

    pickle.dump(model,open("model.p","wb"))
    pickle.dump(tokenizer,open("tokenizer.p","wb"))
    print("Trained and saved model and tokenizer")

def validateBertModel():
    model, tokenizer = loadModel()
    model.eval()
    result = ValidationFunctions.validateBertModel(model,tokenizer)
    result = round(result,2)
    print("Final accuracy: " + str(result) + "%")


def main():
    os.system("clear")
    print('''
        ██████╗░███████╗██████╗░████████╗  ░█████╗░░█████╗░███╗░░██╗████████╗██████╗░░█████╗░██╗░░░░░
        ██╔══██╗██╔════╝██╔══██╗╚══██╔══╝  ██╔══██╗██╔══██╗████╗░██║╚══██╔══╝██╔══██╗██╔══██╗██║░░░░░
        ██████╦╝█████╗░░██████╔╝░░░██║░░░  ██║░░╚═╝██║░░██║██╔██╗██║░░░██║░░░██████╔╝██║░░██║██║░░░░░
        ██╔══██╗██╔══╝░░██╔══██╗░░░██║░░░  ██║░░██╗██║░░██║██║╚████║░░░██║░░░██╔══██╗██║░░██║██║░░░░░
        ██████╦╝███████╗██║░░██║░░░██║░░░  ╚█████╔╝╚█████╔╝██║░╚███║░░░██║░░░██║░░██║╚█████╔╝███████╗
        ╚═════╝░╚══════╝╚═╝░░╚═╝░░░╚═╝░░░  ░╚════╝░░╚════╝░╚═╝░░╚══╝░░░╚═╝░░░╚═╝░░╚═╝░╚════╝░╚══════╝
    ''')
    print("1) Train model")
    print("2) Evaluate model")
    print("3) Quit")
    choice = input("Enter your mode: ")
    if(choice == "1"):
        defaultFile = "data.p"
        fileName = input("Input sequnce file name (default "+ defaultFile +"): ")
        if(fileName == ''):
            fileName = defaultFile
        os.system("clear")
        trainBertModel(fileName)
    elif(choice == "2"):
        os.system("clear")
        validateBertModel()
    elif(choice == "3"):
        exit(0)

    else:
        print("Error command")

if __name__ == '__main__':
    main()
