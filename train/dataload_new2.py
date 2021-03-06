"""
most is same for dataload_new.py
not using moving windows, just one sequence to one target.
@data: flights.csv
@date: 2022/7/26

just run in terminal python -m train.dataload_new2

there some new error, in plt.savefig(path), the path can not be a absolute path.
may like pyright config file, just search current workspace. So if need run in
pycharm, need change some file path.

Interestingly, the pd.read_csv() can work fine with absolute path.
."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import seq2seq

import matplotlib.pyplot as plt


# prepare data.
file_dat = pd.read_csv("~/CodePlace/Python/pytorch/datasets/seaborn-data/master/flights.csv")
min_val = file_dat.iloc[:, 2].min()
max_val = file_dat.iloc[:, 2].max()
one_hot_data = pd.get_dummies(file_dat.iloc[:, 1:])


def normalization(sample_value, min_val, max_val):
    val_rang = max_val - min_val
    return (sample_value - min_val) / val_rang


def renormalization(sample_value, min_val, max_val):
    val_rang = max_val - min_val
    return (sample_value * val_rang) + min_val


total_len = len(one_hot_data)
np_dat = np.asarray(one_hot_data)

test_split = total_len - int(total_len * 0.2)

train_dat = np_dat[:test_split]
test_dat = np_dat[test_split:]

# model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("current training device: ", device)

s2s = seq2seq.Seq2Seq(12, 50, 1, 1, 1, layer_num=1, device=device)
epochs = 1
print("Total number of iterations: ", epochs)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(s2s.parameters(), lr=0.001)

last_train = []
for epoch in range(epochs):
    hn = cn = None
    train_error = []
    print("training epoch: ", epoch)
    for i in range(test_split - 1):
        input_window = train_dat[i:i+1]
        target_window = train_dat[i+1:i+2]

        encoder_input = input_window[:, 1:]
        decoder_input = input_window[:, 0]

        decoder_input = normalization(decoder_input, min_val, max_val)

        encoder_input = torch.Tensor(encoder_input)
        encoder_input = encoder_input.reshape((1, 1, 12)).to(device)

        decoder_input = torch.Tensor(decoder_input)
        decoder_input = decoder_input.reshape((1, 1, 1)).to(device)

        target_val = target_window[:, 0]
        target_val = normalization(target_val, min_val, max_val)
        target_val = torch.Tensor(target_val).reshape((1, 1))

        prediction, hn, cn = s2s(encoder_input, decoder_input, hn, cn)

        optimizer.zero_grad()
        loss = loss_fn(prediction.to("cpu"), target_val)
        loss.backward()
        optimizer.step()

        if epoch == epochs - 1:
            last_train.append(prediction.item())

        # print(loss.item())
        train_error.append(loss.item())
    print("average loss: ", (np.asarray(train_error) / len(train_error)).sum())


last_train = np.asarray(last_train)
np_train_dat = np.asarray(train_dat[1:, 0])

# Scale transformation does not make a difference in graphics
plt.plot(np.arange(len(last_train)), last_train, color="black", label="predic")
norm_np_train = normalization(np_train_dat, min_val, max_val)
plt.plot(np.arange(len(norm_np_train)), norm_np_train, color="r", label='true')
plt.legend()
plt.savefig("./tests/nom_predict.png")
plt.close()

renom_predit = renormalization(last_train, min_val, max_val)
plt.plot(np.arange(len(renom_predit)), renom_predit, color='black', label='predict')
plt.plot(np.arange(len(np_train_dat)), np_train_dat, color='r', label='true')
plt.legend()
plt.savefig("./tests/renom_predict.png")

# test
test_input_data = train_dat[-1:]
test_input_data = np.concatenate((test_input_data, test_dat[0:]), axis=0)
test_len = len(test_input_data)
test_cord = []
test_error = []
s2s.eval()
for i in range(test_len-1):
    input_window = test_input_data[i:i+1]
    target_window = test_input_data[i+1:i+2]

    encoder_input = input_window[:, 1:]
    decoder_input = input_window[:, 0]

    decoder_input = normalization(decoder_input, min_val, max_val)

    encoder_input = torch.Tensor(encoder_input).reshape((1, 1, 12)).to(device)

    decoder_input = torch.Tensor(decoder_input).reshape((1, 1, 1)).to(device)

    target_val = target_window[:, 0]
    target_val = normalization(target_val, min_val, max_val)
    target_val = torch.Tensor(target_val).reshape((1, 1))

    with torch.no_grad():
        prediction, hn, cn = s2s(encoder_input, decoder_input, hn, cn)
        loss = loss_fn(prediction.to("cpu"), target_val)
        test_cord.append(prediction.item())
        test_error.append(loss.item())

print("average test error: ", (np.asarray(test_error) / len(test_error)).sum())


test_cord = np.asarray(test_cord)
future_cord = np.asarray(test_dat[:, 0])
test_cord = renormalization(test_cord, min_val, max_val)

plt.close()
plt.plot(np.arange(len(test_cord)), test_cord, color='black', label="predict")
plt.plot(np.arange(len(future_cord)), future_cord, color='r', label='true')
plt.legend()
plt.savefig("./tests/renom_test.png")

