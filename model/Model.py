from torch import nn
import torch
import numpy as np


class TextCNN(nn.Module):
    def __init__(self, class_size, embeddings, filternum, dropout):
        super(TextCNN, self).__init__()
        # super().__init__()
        self.filternum = filternum
        self.inchannel = embeddings
        self.outchannel = class_size
        self.config = 59
        #         self.conv = nn.Conv2d(1,self.filternum,kernel_size=(3,self.inchannel))
        #         self.bn = nn.BatchNorm2d(self.filternum)
        # #         self.relu = nn.ReLU(inplace=True),
        #         self.max = nn.MaxPool2d((config - 3+1,1))
        self.NetworkUnit3 = nn.Sequential(
            nn.Conv2d(1, self.filternum, kernel_size=(3, self.inchannel)),
            nn.BatchNorm2d(self.filternum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((self.config - 3 + 1, 1))
        )

        self.NetworkUnit4 = nn.Sequential(
            nn.Conv2d(1, self.filternum, kernel_size=(4, self.inchannel)),
            nn.BatchNorm2d(self.filternum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((self.config - 4 + 1, 1))

        )

        self.NetworkUnit5 = nn.Sequential(
            nn.Conv2d(1, self.filternum, kernel_size=(5, self.inchannel)),
            nn.BatchNorm2d(self.filternum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((self.config - 5 + 1, 1))

        )
        self.dp = nn.Dropout(dropout)

        self.linear = nn.Linear(3 * filternum, self.outchannel)

    def forward(self, x, maxlength):
        # length per batch is different
        self.length_perbatch = maxlength
        self.config = maxlength
        #         print(self.length_perbatch)
        batchsize = x.shape[0]

        #         x1 = self.conv(x)
        #         print(x1.shape)
        #         x1 = self.bn(self.conv(x))
        #         print(x1.shape)
        #         x1 = self.max(F.relu(x1))
        #         print(x1.shape)

        x1 = self.NetworkUnit3(x)
        x2 = self.NetworkUnit4(x)
        x3 = self.NetworkUnit5(x)
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batchsize, 1, -1)
        #         print(x.shape)
        x = self.linear(self.dp(x))
        x = x.view(-1, self.outchannel)
        return x



