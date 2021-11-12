import TrainTokenizer
import pickle
import TrainModel

def main():
    data = pickle.load(open("data.p","rb"))
    tokenizer = TrainTokenizer.compileTokenizer(data)
    model = TrainModel.compileModel(data)

if __name__ == '__main__':
    main()
