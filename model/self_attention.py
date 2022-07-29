"""
This file is to understand the theory of self-attention mechanism, and code the implementation

The weight W_Q, W_K, W_V generate q, k, v of each input vector, it is init by nn.Linear(input_dim, output_dim).
The `input_dim` is equal to the dimension of input vector `x` (asuming just one sequence),
and the `outpu_dim` is equal to the dimension of q, the k, v is just lik q.

by contrast, this file is very as same as the file:
scienceComput/pytorch/py/attention.py -> This file is copy from youtube.

EXAMPLE
-------

>>> x = torch.Tensor([1, 2, 3, 4])
>>> W_q = nn.Linear(4, 4)
>>> W_q(x)
[2.1, 2.0, 1.1, 1.2]  # a sigle q.
"""


import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class SelfAttention(nn.Module):
    # TODO: maybe need add other input dim.
    def __init__(self, input_dim: int, q_dim: int, k_dim: int = -1, v_dim: int = -1) -> None:
        """
        usually, the dimension of q is equal to v.

        ARGS
        ____

        : param input_dim : the dimension of input vector. Like the example above.
        : param q_dim     : the dimension of q vector.
        : param k_dim     : the dimension of k vector. Default is -1, means equal to the dimension of q.
        : param v_dim     : the dimension of v vector. Default is -1, means equal to the dimension of q.
        """
        super().__init__()
        self.q_dim = q_dim
        self.k_dim = q_dim if k_dim == -1 else k_dim
        self.v_dim = q_dim if v_dim == -1 else v_dim

        # Notice here, the bias need to set false.
        # Just need the weight matrix
        self.W_Q = nn.Linear(input_dim, self.q_dim, bias=False)
        self.W_K = nn.Linear(input_dim, self.k_dim, bias=False)
        self.W_V = nn.Linear(input_dim, self.v_dim, bias=False)

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x_q: Tensor, x_k: Tensor = None, x_v: Tensor = None):
        """
        According to the paper, the input data used to calculate q, k, v can bifferent. So add the
        redundant parameters: x_k, x_v. But in Transformer model, this three input is same.

        The pytorch code is set the seconde position to batch_size: (seq, batch_size, feature), in
        my code, setting the first position to batch_size: (batch_size, seq, data)

        ARGS
        ____

        :param x_q: used to calculate vector q.
        :param x_k: used to calculate vector k.
        :param x_v: used to calculate vector v.
        """
        x_k = x_q if x_k is None else x_k
        x_v = x_q if x_v is None else x_v

        Q = self.W_Q(x_q)
        K = self.W_K(x_k)
        V = self.W_V(x_v)

        """
        calculate QK^T
        the torch.bmm is matrix times, and it is batch first.
        this function is require the input shape is 3-D.
        torch.matmul() is also a matrix times, but do not have rule for input shape.
        """
        middle_state = torch.bmm(Q, K.transpose(1, 2))
        middle_state = self.soft_max(middle_state / np.sqrt(self.q_dim))
        return torch.bmm(middle_state, V)
