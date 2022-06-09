"""
@author: ccg
@data: 2022/6/5

DESCRIPTION:
-----------
This file is a underlying model.

In the whole training process, maybe the hidden state and the cell state just need
initial once for each iteration. Or just need account for some other reason to
choose which time need to initial the hidden state and the cell state.
"""


from typing import Optional, Tuple
import torch
from torch.functional import Tensor
import torch.nn as nn


class LSTM(nn.Module):
    """
    The tensor shape of this calss is (seq_len, batch_size, features)
    """
    def __init__(self,n_input, n_output, n_hidden=51, n_layer=1) -> None:
        """
        :param n_input:  the feature number of input tensor.
        :param n_output: the feature number of output tensor.
        :param n_input:  the feature number of hidden layer.
        :param n_input:  the number of lstm layer.
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=n_hidden, num_layers=n_layer)
        self.fc = nn.Linear(n_hidden, n_output)

    def initHidden(self, batch_size):
        return torch.zeros(self.n_layer, batch_size, self.n_hidden)

    def forward(self, x: Tensor, hx: Optional[Tuple[Tensor, Tensor]]=None) -> Tensor:
        assert hx is not None, "need initHidden parameters"
        _, (h_t, _) = self.lstm(x, hx)
        # the shape of h_t is (n_layer, batch_size, hidden_feature)
        # so we need the hidden state of last layer.
        return self.fc(h_t[-1])

    def forward_(self, x, future=0):
        """
        older one, and should not be used.
        """
        outputs = []
        n_samples = x.size(0)

        h_t = torch.zeros(2, n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(2, n_samples, self.n_hidden, dtype=torch.float32)

        for input_t in x.split(1, dim=1):
            _, (h_t, c_t) = self.lstm(input_t.unsqueeze(0).to(torch.float32), (h_t, c_t))
            output = self.fc(h_t[-1]).squeeze(0)
            outputs.append(output)

        # predict
        for _ in range(future):
            _, (h_t, c_t) = self.lstm(outputs[-1].unsqueeze(0), (h_t, c_t))
            output = self.fc(h_t[-1]).squeeze(0)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs

