from transformers import BertConfig, BertLMHeadModel
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline

import torch
import pickle
import numpy as np
import json
import os

#Reference: https://www.kaggle.com/mojammel/train-model-from-scratch-with-huggingface

class CodeDataset(Dataset):
    def __init__(self,input,mask):
        self.inputs = np.array(input)
        self.mask = np.array(mask)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src = self.inputs[idx]
        mask = self.mask[idx]
        return src, mask

def makeModel(tokenizer):
    #TODO: better bert config
    config = BertConfig(tokenizer.vocab_size, hidden_size=300,
                        num_hidden_layers=2, num_attention_heads=2, is_decoder=True,
                        add_cross_attention=True)
    model = BertLMHeadModel(config)
    return model

def eval_model(model,tokenizer):
    code_example = "printStatus CallExpression [MASK] BlockStatement FunctionDeclaration Program"
    fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    outputs = fill_mask(code_example)
    for i in outputs:
        print(i)

def data_collate_fn(features):
    batch = {}
    batch['input_ids'] = torch.tensor([t[0] for t in features])
    batch['attention_mask'] = torch.tensor([t[1] for t in features])
    return batch

def main():
    MAX_LEN = 15
    data = pickle.load(open("data.p","rb"))
    sentences = []
    with open("Sentences.txt","r") as f:
        sentences = f.read().split("\n")

    tokenizer = BertTokenizer.from_pretrained('./CodeTokenizer')
    #tokenizedData = [tokenizer(l,pad_to_max_length=True,max_length=MAX_LEN,add_special_tokens =True) for l in sentences]
    tokenizedData = tokenizer(sentences,padding="longest",max_length=MAX_LEN)
    input = [d for d in tokenizedData["input_ids"]]
    mask = [d for d in tokenizedData["attention_mask"]]
    dataset = CodeDataset(input, mask)

    #data_collate_fn(tokenizer,data)
    model = makeModel(tokenizer)




    training_args = TrainingArguments(
        output_dir="./bert",
        overwrite_output_dir=True,
        num_train_epochs=10,
        do_train=True,
        per_gpu_train_batch_size=32,
        save_steps=500,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collate_fn
    )

    trainer.train()



if __name__ == '__main__':
    main()
