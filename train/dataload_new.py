"""
training new seq2seq model.
using moving windows
@data: flights.csv
@date: 2022/7/22
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import seq2seq

file_dat = pd.read_csv("~/CodePlace/Python/pytorch/datasets/seaborn-data/master/flights.csv")
min_val = file_dat.iloc[:, 2].min()
max_val = file_dat.iloc[:, 2].max()
one_hot_data = pd.get_dummies(file_dat.iloc[:, 1:])


def normalization(sample_value, min_val, max_val):
    val_rang = max_val - min_val;
    return (sample_value - min_val) / val_rang


def renormalization(prediction_val, min_val, max_val):
    val_range = max_val - min_val
    return prediction_val * val_range + min_val


total_len = len(one_hot_data)
np_dat = np.asarray(one_hot_data)
windows = 6

# model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("current training device: ", device)

s2s = seq2seq.Seq2Seq(12, 50, 1, 1, 1, device=device)
epochs = 10
loss_fn = nn.MSELoss()
optimizer = optim.Adam(s2s.parameters(), lr=0.001)

for epoch in range(epochs):
    hn = cn = None
    print("training epoch: ", epoch)
    for i in range(total_len - windows):
        # moving windows
        input_windows = np_dat[i:i + windows]
        target_windows = np_dat[i + windows:i + windows + 1]

        encoder_input = input_windows[:, 1:]
        decoder_input = input_windows[:, 0]

        decoder_input = normalization(decoder_input, min_val, max_val)

        encoder_input = torch.Tensor(encoder_input)
        encoder_input = encoder_input.reshape((6, 1, 12)).to(device)

        decoder_input = torch.Tensor(decoder_input)
        decoder_input = decoder_input.reshape((6, 1, 1)).to(device)
        # print(encoder_input)
        # print(decoder_input)

        # target
        target_val = target_windows[:, 0]
        target_val = normalization(target_val, min_val, max_val)
        target_val = torch.Tensor(target_val).reshape((1, 1))
        # print(target_val)

        # train
        prediction, hn, cn = s2s(encoder_input, decoder_input, hn, cn)

        # prediction = renormalization(prediction, min_val, max_val)

        optimizer.zero_grad()
        loss = loss_fn(prediction.to("cpu"), target_val)
        loss.backward()
        optimizer.step()
        print(loss.item())
