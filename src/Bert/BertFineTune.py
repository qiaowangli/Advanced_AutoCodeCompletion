# import torch
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split

from transformers import BertTokenizer

# from pytorch_pretrained_bert import BertTokenizer, BertConfig
# from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
# from tqdm import tqdm, trange
# import io
import numpy as np
import pickle
import os

import json
from minio import Minio

def getData():
    #TODO: adjust input method (instead of pickle) when the real data come
    #Compile tokenized data by adding the flags
    raw = pickle.load(open("data.p","rb"))
    tokens = [["[CLS]"] + l + ["[SEP]"] for l in raw]

    return tokens

def train(tokens):
    MAX_LEN = 15
    #MAX_LEN = int(sum([len(x) for x in tokens]) / len(tokens))
    tokenizer = BertTokenizer.from_pretrained('./CodeTokenizer')
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokens]

    for i in input_ids:
        print(i)

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []
    for seq in input_ids:
        attention_masks.append([float(i>0) for i in seq])

    segment_masks = [0 for i in range(len(input_ids))]


def main():
    os.system("clear")
    tokens = getData()
    train(tokens)

if __name__ == '__main__':
    main()
