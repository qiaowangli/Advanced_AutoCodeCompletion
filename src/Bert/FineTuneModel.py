from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertAdam
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
import random
import json
from minio import Minio
import torch
import pickle

class CodeDataSet(Dataset):
    def __init__(self,encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self,idx):
        val = {key: val[idx] for key, val in self.encodings.items()}
        return val

def getSamples(data,sampleRate):
    indexs = [i for i in range(len(data["input_ids"]))]
    random.shuffle(indexs)
    samplesNum = int(len(data["input_ids"])*sampleRate)
    indexs = indexs[:samplesNum]

    input_ids = data["input_ids"][indexs]
    masks = data["attention_mask"][indexs]
    tokens = data["token_type_ids"][indexs]
    slabel = data["next_sentence_label"][indexs]
    return input_ids,masks,tokens,slabel

def maskSeq(ids,maskRate):
    data = ids.clone().detach()
    for x in range(len(data)):
        seq = data[x]
        maskNum = int(len(seq)*maskRate)
        for i in range(maskNum):
            index = random.randint(0,len(seq)-1)
            for i in range(4):
                index = (index+i)%len(seq)
                if(seq[index] not in [0,2,3,4]):
                    data[x][index] = 4
                    break
    return data

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def MakeNSPinput(data):
    '''
    Generate train data for NSP training
    input: data is the list sequences (while seq is a list of words)
    output: first and second setence pair with label 0 for true and 1 for false
    '''
    firstHalf = []
    secondHalf = []
    #0 -> a and b does connect 1 -> they don't
    labels = []
    for i in range(len(data)):
        if(random.random() > 0.4):
            #Correct split
            labels.append(0)
            splitPoint = random.randint(1,len(data[i])-1)
            firstHalf.append(' '.join(data[i][:splitPoint]))
            secondHalf.append(' '.join(data[i][splitPoint:]))
        else:
            #random split
            labels.append(1)
            otherList = random.randint(0,len(data)-1)
            while otherList == i:
                otherList = random.randint(0,len(data)-1)
            splitPoint = random.randint(1,len(data[i])-1)
            firstHalf.append(' '.join(data[i][:splitPoint]))
            splitPoint = random.randint(1,len(data[otherList])-1)
            secondHalf.append(' '.join(data[otherList][splitPoint:]))
    labels = torch.LongTensor([labels]).T
    return firstHalf,secondHalf,labels

def processDataloaders(data,tokenizer,MAX_LEN=15,sampleRate=0.1,maskRate=0.15):
    firstHalf,secondHalf,labels = MakeNSPinput(data)

    tokenized = tokenizer(firstHalf,secondHalf,return_tensors="pt",
                        max_length=MAX_LEN,truncation=True,padding="max_length")
    tokenized["next_sentence_label"] = labels

    train_labels, train_masks, train_tokens,train_slabels = getSamples(tokenized,sampleRate)
    train_input_ids = maskSeq(train_labels,maskRate)
    train_inputs = {"input_ids":train_input_ids,
                    "token_type_ids":train_tokens,
                    "attention_mask":train_masks,
                    "next_sentence_label":train_slabels,
                    "labels":train_labels}

    validation_labels, validation_masks, validation_tokens, validation_slabels = getSamples(tokenized,sampleRate)
    validation_input_ids = maskSeq(validation_labels,maskRate)
    validation_input = {"input_ids":validation_input_ids,
                    "token_type_ids":validation_tokens,
                    "attention_mask":validation_masks,
                    "next_sentence_label":validation_slabels,
                    "labels":validation_labels}

    batch_size = 32 #16 or 32
    train_data = CodeDataSet(train_inputs)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = CodeDataSet(validation_input)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    return train_dataloader, validation_dataloader

def getOptimizer(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=2e-5,
                         warmup=.1)
    return optimizer

def trainWithData(model,loader,epochCount):
    device = torch.device("cpu")
    model.to(device)
    model.train()
    optimizer = getOptimizer(model)

    train_loss_set = []
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for epoch in range(epochCount):
        loop = tqdm(loader,leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            next_sentence_label = batch["next_sentence_label"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids,token_type_ids=None,
                            attention_mask=attention_mask,
                            next_sentence_label=next_sentence_label,
                            labels=labels)
            loss = outputs.loss
            train_loss_set.append(loss.item())
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    return model

def validaMode(model,loader,epochCount):
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    optimizer = getOptimizer(model)
    device = torch.device("cpu")
    model.eval()

    for epoch in range(epochCount):
        loop = tqdm(loader,leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            next_sentence_label = batch["next_sentence_label"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids,token_type_ids=None,
                            attention_mask=attention_mask)

            #logits = logits.detach().cpu().numpy()
            #label_ids = b_labels.to('cpu').numpy()

            logits = logits[0].detach().numpy()
            labels = labels.numpy()

            tmp_eval_accuracy = flat_accuracy(logits, labels)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    return model

def fineTune(model,MAX_LEN=512,epochCount=16):
    with open("Sentences.txt","r") as f:
        data = f.read().split('\n')
    while '' in data:
        data.remove('')
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained('./CodeTokenizer')
    train_dataloader,validation_dataloader = processDataloaders(data,tokenizer,MAX_LEN)
    trainWithData(model,train_dataloader,epochCount)
    validaMode(model,validation_dataloader,epochCount)
    return model

def main():
    train()

if __name__ == '__main__':
    main()
