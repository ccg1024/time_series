# -*- coding: utf-8 -*-
"""
training time series model, and compare the accuracy.
"""
import sys
sys.path.append("/home/william/CodePlace/Python/pytorch/time_series")
from model import seq2seq
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


total_data = torch.load('./datas/traindata.pt')

one_sample = total_data[0]

train_data = one_sample[:900]
test_data = one_sample[900:]

windows = 10
train_target = one_sample[10:900]
test_target = one_sample[910:]

encoder = seq2seq.Encoder(1, 51)
decoder = seq2seq.Decoder(1, 51)
model = seq2seq.Seq2seq(encoder, decoder, 51, 1)

# test = torch.from_numpy(train_data[:windows])
# test.view(10, 1, -1).shape  数据应该构造成这样，10个数据看成10个序列

criterion = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=0.001)


epoch = 10
final_predict = []
for i in range(epoch):
    for j in range(len(train_target)):
        opt.zero_grad()
        X = train_data[j:windows+j]
        X = torch.from_numpy(X).view(10, 1, -1)
        y = torch.tensor([train_target[j]]).view(1, 1, -1)
        # print(X)
        # print(y)
        # print(X.shape, y.shape)
        predict = model(X, y)
        if i == epoch - 1:
            final_predict.append(predict.item())
        loss = criterion(predict.view(1, 1, -1), y)
        print('loss: ', loss.item())
        loss.backward()
        opt.step()

# print(final_predict[0].item())
# print(train_target[0])

plt.plot(np.arange(len(train_target)), train_target, 'b')
plt.plot(np.arange(len(train_target)), final_predict, 'r')
plt.show()
