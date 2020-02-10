import Model as model
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.optim import lr_scheduler
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataset import PreProcess,TxtCNNDataset,collate_fn,createVocab,load_bin_vec
import nltk



def train(model:nn.Module, epochs:int, data_loader,test_loader,savepath):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        scheduler.step()
        avg_loss = 0.0
        train_loss = []
        for batchid, (inputs, targets, maxlen ) in enumerate(data_loader):
            optimizer.zero_grad()
            inputs, targets= inputs.to(device), targets.to(device)
            outputs = model(inputs,maxlen)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            train_loss += [loss.item()]
            if batchid % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch + 1, batchid + 1, avg_loss / 50))
                avg_loss = 0.0
            torch.cuda.empty_cache()


        val_loss, val_acc = evaluate(model, test_loader)

        print('Epoch:{:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
              format( epoch, val_loss, val_acc))

        with open('./loss_mobile.txt', 'w+') as f:
            f.write('Epoch:{:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}\n'.
                    format(epoch, val_loss, val_acc))


        save_model(model,savepath)



def evaluate(model:nn.Module,test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels,maxlength) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats,maxlength)

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        loss = criterion(outputs, labels)
        #         c_loss = criterion_closs(feature, labels.long())
        #         loss = l_loss

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()] * feats.size()[0])
        del feats
        del labels

    return np.mean(test_loss), accuracy/total


def predict(model:nn.Module, test_loader):
    model.eval()
    predict = []
    for batch_num, (feats, ids, maxlen) in enumerate(test_loader):
        feats = feats.to(device)
        #         ID = ID.to(device)
        outputs= model(feats,maxlen)

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        predict.append(pred_labels.cpu().numpy())
        # ids.append(ID)

        del feats
        # del ID
    return predict


def save_model(model, path):
    torch.save(model.state_dict(), path)




def load_model(model,path):
    pretrained = torch.load(path)
    model.load_state_dict(pretrained)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TextCNN")
    parser.add_argument('-lr', type=float,default=0.001,help="learning rate initial")
    parser.add_argument('-epoch', type=int,default=100,help="number of epochs")
    parser.add_argument('-batchsize', type=int,default=64,help="batch size")
    parser.add_argument('-dropout', type=float,default=0.5,help="dropout rate")
    parser.add_argument('-filternumber', type=int,default=100,help="number of each size of filter")
    parser.add_argument('-device', type=str,default='cpu'if not torch.cuda.is_available() else 'cuda'
                        ,help="device")
    parser.add_argument('-savedpath', type=str,default='./save1.pt',help="path to save model")
    parser.add_argument('-weightdecay', type=float,default=0.0001,help="necessary rate for L2 regularizer")
    parser.add_argument('-pretrainedpath', type=str,default='./',help="path of pretrained word2vec")
    parser.add_argument('-gamma', type=float,default=0.1,help="gamma rate for scheduler")
    parser.add_argument('-step', type=int,default=10,help="step size for scheduler")
    parser.add_argument('-class', type=int,default=16,help="class number output")
    parser.add_argument('-embedding', type=int,default=300,help="embeddings channel")
    args = parser.parse_args()

# preprocess
    nltk.download('punkt')

    file = "../topicclass/topicclass_train.txt"

    with open(file) as f:
        lines = f.readlines()
        x = []
        y = []
        for i in lines:
            target, text = i.split("|||")
            y.append(target.strip())


    cl = list(set(y))
    vocab = set()
    readf = "./topicclass/topicclass_train.txt"
    vocab = createVocab(readf, vocab)
    word_vec = load_bin_vec('./GoogleNews-vectors-negative300.bin', vocab)
    trainx, trainy = PreProcess(file, cl,word_vec)

    dataset = TxtCNNDataset(trainx, trainy)


    train_loader = DataLoader(dataset, batch_size=64, shuffle=True,
                              collate_fn=collate_fn)
    file1 = "../topicclass/topicclass_valid.txt"
    devx, devy = PreProcess(file1, cl)
    testset = TxtCNNDataset(devx, devy)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=True,
                             collate_fn=collate_fn)

    file2 = "./topicclass/topicclass_test.txt"
    testx, testy = PreProcess(file2, cl,word_vec)
    finalset = TxtCNNDataset(testx, testy)
    final_loader = DataLoader(finalset, batch_size=64, shuffle=False,
                              collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learningRate = 1e-2
    weightDecay = 5e-3
    model = model.TextCNN(16, 300, 100, 0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train(model,args.epoch,train_loader, test_loader,args.savedpath)
    labels = predict(model,test_loader)
    list2 = []
    for i in labels:
        for j in i:
            list2.append(j)
    result = [cl[i] for i in list2]
    index = [i for i in range(len(list2))]
    data = pd.DataFrame(result, index=index)
    data.sort_index(inplace=True)
    data.to_csv("./testresult1.csv")