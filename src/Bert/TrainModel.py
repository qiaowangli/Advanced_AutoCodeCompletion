from transformers import BertConfig, BertForPreTraining
from transformers import BertTokenizer
from transformers import AdamW
from torch.utils.data import Dataset
from tqdm import tqdm

import random
import torch

#Reference = https://towardsdatascience.com/how-to-train-bert-aaad00533168

class CodeDataSet(Dataset):
    def __init__(self,encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self,idx):
        val = {key: val[idx] for key, val in self.encodings.items()}
        return val

def makeModel(tokenizer):
    #TODO: better bert config
    config = BertConfig(tokenizer.vocab_size, hidden_size=300,
                        num_hidden_layers=2, num_attention_heads=2, is_decoder=True,
                        add_cross_attention=True)
    model = BertForPreTraining (config)
    return model

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

def MakeMLMMasking(inputs,maskingRate = 0.15):
    rand = torch.rand(inputs.input_ids.shape)
    #Do not mask CLS, SEP, or PAD
    mask_arr = (rand < 0.15) * (inputs.input_ids != 3) * (inputs.input_ids != 4) * (inputs.input_ids != 0)
    for i in range(inputs.input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        #MASK id
        inputs.input_ids[i,selection] = 5
    return inputs

def TrainWithData(model,loader,epochCount):
    device = torch.device("cpu")
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(),lr=5e-5)

    for epoch in range(epochCount):
        loop = tqdm(loader,leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            next_sentence_label = batch["next_sentence_label"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids,token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            next_sentence_label=next_sentence_label,
                            labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
    return model


def compileModel(data,MAX_LEN = 15,maskingRate = 0.15,epochCount=3):
    #data is the list sequences (while seq is a list of words)
    firstHalf,secondHalf,labels = MakeNSPinput(data)

    tokenizer = BertTokenizer.from_pretrained('./CodeTokenizer')
    inputs = tokenizer(firstHalf,secondHalf,return_tensors="pt",
                        max_length=MAX_LEN,truncation=True,padding="max_length")
    inputs["next_sentence_label"] = labels
    inputs["labels"] = inputs.input_ids.detach().clone()

    inputs = MakeMLMMasking(inputs,maskingRate)
    dataset = CodeDataSet(inputs)

    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    #device = torch.device("cuba") if torch.cuba.is_available() else torch.device("cpu")

    model = makeModel(tokenizer)
    model = TrainWithData(model,loader,epochCount)
    return model

def main():
    compileModel()

if __name__ == '__main__':
    main()
