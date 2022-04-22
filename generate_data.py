# -*- coding: utf-8 -*-
"""
 文件描述:
 generate some simple time series data.
 @作者: ccg
 @日期: 2022 04 21 10:03
 @路径: time_series-PyCharm-generate_data
"""


import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float32')
torch.save(data, open('datas/traindata.pt', 'wb'))

plt.figure(figsize=(10, 8))
plt.title('Sine wave')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.arange(x.shape[1]), data[0, :], 'r', linewidth=2.0)
plt.show()
