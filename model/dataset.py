import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import nltk
from torch.optim import lr_scheduler
import string
from torch.autograd import Variable

def PreProcess(file,cl,word_vec):
    with open (file) as f:
        lines = f.readlines()
        x = []
        y = []
        for i in lines:
            target,text = i.split("|||")
            sentence = text.strip()
            sentence = sentence.translate(str.maketrans('','',string.punctuation))
    #         stop words need to be processed
            sentence = nltk.word_tokenize(sentence)
            sent = np.zeros((len(sentence),300))
            for j in range(len(sentence)):
#                 if sentence[j] not in vectors.vocab:
                if sentence[j] not in word_vec.keys():
                    word = np.zeros(300).astype(np.float32)
                else:
                    word = word_vec[sentence[j]].astype(np.float32)
                sent[j] = word
            x.append(sent)
            if target.strip() in cl:
                y.append(cl.index(target.strip()))
            else: y.append(100)
        return x, y


class TxtCNNDataset(Dataset):
    def __init__(self, texts, targets):
        self.seqs = texts
        self.labels = targets

    def __getitem__(self, i):
        line = self.seqs[i]
        label = self.labels[i]
        return self.seqs[i], self.labels[i]

    #         return line.to(DEVICE),label.to(DEVICE)
    def __len__(self):
        return len(self.seqs)


def collate_fn(seq_list):
    inputs, targets = zip(*seq_list)
    maxlen = max([len(seq) for seq in inputs])
    a = [np.pad(row, ((0, maxlen - len(row)), (0, 0)), 'constant').astype(np.float32) for row in inputs]
    a = np.asarray(a).astype(np.float32)
    data = torch.tensor(a)
    #     data = torch.DoubleTensor(np.asarray([np.pad(row,((0,maxlen - len(row)),(0,0)),'constant')for row in inputs]).astype(np.float32))
    targets = torch.DoubleTensor(np.asarray([i for i in targets]))

    # print(type(data))
    data = data.unsqueeze(1)
    # print(data.shape)
    return data, targets.long(), maxlen


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, 'rb') as f:
        header = f.readline()
        # print (header)
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size): #Change this to full on a bigger machine
            #print line
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
            if (line%100000==0):
                print(line)
    return word_vecs


def createVocab(readf, vocab):
    with open (readf) as f:
        lines = f.readlines()
        for i in lines:
            target,text = i.split("|||")
            sentence = text.strip()
            sentence = sentence.translate(str.maketrans('','',string.punctuation))
            sentence = nltk.word_tokenize(sentence)
            for j in sentence:
                vocab.add(j)
    return vocab
