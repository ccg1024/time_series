import pandas as pd
import numpy as np

pd_file = pd.read_csv("~/CodePlace/Python/pytorch/datasets/seaborn-data/master/flights.csv")
one_hot_dat = pd.get_dummies(pd_file.iloc[:, 1:])

gap = 6
total_len = len(one_hot_dat)

