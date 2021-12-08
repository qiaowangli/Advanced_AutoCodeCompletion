from transformers import BertTokenizer
import torch
import os

def FindMsk(model,tokenizer,sentence,words,topn=5,echo=True):
    #reference https://gist.github.com/yuchenlin/a2f42d3c4378ed7b83de65c7a2222eb2
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
        if word in words:
            if(echo):
                print("[MASK]: '%s'"%word, " | weights:", weights[i])
            reList.append(word)
            topn -= 1
    if(echo):
        print()
    return reList

def validate(model,tokenizer,testIndex,topn,echo):
    input = []
    with open("Sentences.txt","r") as f:
        input = f.read().split('\n')
    with open("words.txt",'r') as f:
        words = f.read().split('\n')

    total = len(input)
    correctPrediction = 0
    for s in input:
        sl = s.split(' ')
        correct = sl[testIndex]
        sl[testIndex] = "[MASK]"
        s = ' '.join(sl)
        predictions = FindMsk(model,tokenizer,s,words,topn,echo)
        if(correct in predictions):
            correctPrediction += 1
    return float(correctPrediction/total) * 100


def validateBertModel(model,tokenizer):
    os.system("clear")
    print('''
        ██████╗░███████╗██████╗░████████╗  ███████╗██╗░░░██╗░█████╗░██╗░░░░░██╗░░░██╗░█████╗░████████╗██╗░█████╗░███╗░░██╗
        ██╔══██╗██╔════╝██╔══██╗╚══██╔══╝  ██╔════╝██║░░░██║██╔══██╗██║░░░░░██║░░░██║██╔══██╗╚══██╔══╝██║██╔══██╗████╗░██║
        ██████╦╝█████╗░░██████╔╝░░░██║░░░  █████╗░░╚██╗░██╔╝███████║██║░░░░░██║░░░██║███████║░░░██║░░░██║██║░░██║██╔██╗██║
        ██╔══██╗██╔══╝░░██╔══██╗░░░██║░░░  ██╔══╝░░░╚████╔╝░██╔══██║██║░░░░░██║░░░██║██╔══██║░░░██║░░░██║██║░░██║██║╚████║
        ██████╦╝███████╗██║░░██║░░░██║░░░  ███████╗░░╚██╔╝░░██║░░██║███████╗╚██████╔╝██║░░██║░░░██║░░░██║╚█████╔╝██║░╚███║
        ╚═════╝░╚══════╝╚═╝░░╚═╝░░░╚═╝░░░  ╚══════╝░░░╚═╝░░░╚═╝░░╚═╝╚══════╝░╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚═╝░╚════╝░╚═╝░░╚══╝
    ''')
    testIndex = input("Input the index of the masked word (default 0): ")
    try:
        testIndex = int(testIndex)
    except:
        testIndex = 0

    topn = input("Input n for topn accuracy test (default 5): ")
    try:
        topn = int(topn)
    except:
        topn = 5

    echo = input("Print output? (y/n) (default n): ")
    echo = True if echo == 'y' else False

    return validate(model,tokenizer,testIndex,topn,echo)
