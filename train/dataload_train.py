"""
training model by patched data object.
"""
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from model import seq2seq


class CustomDataset(Dataset):
    """
    A simple framework for dataset object. the function parameter maybe need to add.
    and the content of function need to be finished.
    """
    def __init__(self, file_path, transform=None, target_transform=None):
        """
        :param transform: a modify function or a Lambda expression. To change the data to final version.
        :param target_transform: just like `transform`, modify the target data.
        """
        # make some prepare for raw data,
        # example: reshape the data.
        raw_file = pd.read_csv(file_path)
        time_series = raw_file.iloc[:, 2]
        self.min_val = time_series.min()
        self.max_val = time_series.max()
        self.train_x, self.train_y = self.process(time_series)
        self.transform = transform
        self.target_transform = target_transform

    def get_y(self):
        return self.train_y

    def __len__(self):
        # return the total len of training data.
        return len(self.train_y)

    def __getitem__(self, idx):
        # return one sample of training data.
        # the return form: x, y.
        X = self.train_x[idx]
        y = self.train_y[idx]
        if self.transform:
            pass
        if self.target_transform:
            pass
        return X, y

    def process(self, time_series):
        # min-max normalization
        time_series = (time_series - self.min_val) / (self.max_val-self.min_val)
        time_series = torch.tensor(time_series)
        total_len = len(time_series)
        total_tensor = torch.tensor([])
        total_target = []
        gap = 7
        for i in range(total_len-gap-1):
            temp = time_series[i:i+gap]
            total_target.append(time_series[i+gap].item())
            total_tensor = torch.cat((total_tensor, temp), 0)
        total_tensor = total_tensor.view(-1, 7, 1)
        total_target = torch.tensor(total_target).view(-1, 1)
        # print(total_tensor.shape)
        # print(total_target.shape)
        # shape: (136, 7, 1) (136, 1)
        return total_tensor.type(torch.float), total_target.type(torch.float)


epochs = 10
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create dataset obj
# the DataLoader obj is a iterator. return a batched X, y.
# the shape of X, y is: (batch_size, seq, feature)
training_data = CustomDataset('~/seaborn-data/flights.csv')
train_dataloader = DataLoader(training_data, batch_size=batch_size)

# create model
encoder = seq2seq.Encoder(input_size=1, hidden_size=64).to(device)
decoder = seq2seq.Decoder(input_size=1, hidden_size=64).to(device)
model = seq2seq.Seq2seq(encoder=encoder, decoder=decoder, hidden_size=64, batch_size=batch_size).to(device)

# create loss and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# set teaching force rate
teach_force = 0.5

# begin iteration
# remember last train
last_predict = torch.tensor([])
for epoch in range(epochs):
    print('epoch ', epoch)
    for X, y in train_dataloader:
        X = X.transpose(0, 1).to(device)
        y = y.view(batch_size, -1, 1).transpose(0, 1).to(device)
        target_tensor = y if random.random() < teach_force else torch.zeros(y.shape).to(device)
        predict = model(X, target_tensor, device)

        if epoch == epochs - 1:
            last_predict = torch.cat((last_predict, predict.squeeze(1)))

        optimizer.zero_grad()
        loss = loss_fn(predict, y.squeeze(0))
        loss.backward()
        optimizer.step()
        print('\r', loss.item(), end='')
    print()

last_predict = last_predict.detach().numpy()

plt.plot(np.arange(len(training_data)), training_data.get_y())
plt.plot(np.arange(len(training_data)), last_predict)
plt.show()
