"""
 implement lstm manually.
 the reference site: https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
 
 data: 2022/5/17
"""
import math
import torch
import torch.nn as nn


class NaiveCustomLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # i_g
        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # f_g
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # c_g
        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # o_g
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, init_states=None):
        """
        assume x.shape represents (batch_size, sequence_len, input_size)
        """
        
        assert x.dim() == 3, "the tensor shape must equal to (batch_size, sequence_len, input_size)"

        batch_size, sequence_size, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch_size, self.hidden_size).to(x.device),
                torch.zeros(batch_size, self.hidden_size).to(x.device)
            )
        else:
            h_t, c_t = init_states

        for t in range(sequence_size):
            x_t = x[:, t, :]

            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        
        # reshape hidden_seq
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


# the LSTM with Peephole optimization
class CustomPeepholeLSTM(nn.Module):
    """
    there are some confusion things in `forward` function. To see these issues
    in commendations place.
    """
    def __init__(self, input_size, hidden_size, peephole=False) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.peephole = peephole

        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))

        self.init_weights()
        assert False, "this class do not finished, using other class. Like `NativePeepholeLSTM`"

    def init_weights(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, init_states=None):
        assert x.dim() == 3, "the tensor shape must equal to (batch_size, sequence_len, input_size)"

        batch_size, sequence_size, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch_size, self.hidden_size).to(x.device),
                torch.zeros(batch_size, self.hidden_size).to(x.device)
            )
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(sequence_size):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication

            if self.peephole:
                gates = x_t @ self.W + c_t @ self.U + self.bias
            else:
                gates = x_t @ self.W + h_t @ self.U + self.bias
                # g_t = torch.tanh(gates[:, HS*2:HS*3])

            i_t, f_t, o_t = (
                torch.sigmoid(gates[:, :HS]),
                torch.sigmoid(gates[:, HS:HS*2]),
                torch.sigmoid(gates[:, HS:HS*3])
            )

            if self.peephole:
                c_t = f_t * c_t + i_t * torch.sigmoid(x_t @ self.W + self.bias)[:, HS*2:HS*3]
                h_t = torch.tanh(o_t * c_t)
            else:
                # c_t = f_t * c_t + i_t * g_t got issue
                h_t = o_t * torch.tanh(c_t)

        hidden_seq.append(h_t.unsqueeze(0))
        # hidden_seq = hidden_seq.transpose(0, 1).contiguous() got issue

        return hidden_seq, (h_t, c_t)


class NativePeepholeLSTM(nn.Module):
    """
    simple test, the output shape is as same as the original LSTM.
    """
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # input gate
        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.P_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # for peephole

        # forget gate
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.P_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # for peephole
        
        # candidate param
        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # output gate
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        self.P_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # for peephole
        
        # initial weights of model.
        self.init_weights()

    # default initial function of lstm in pytorch.
    def init_weights(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, X: torch.Tensor, init_states=None):
        """
        :param X:           the input tensor with shape (batch_size, seq_len, features).
        :param init_states: the initial value of hidden state and cell state.
        """
        assert X.dim() == 3, "the input tensor dimension must equal to 3."

        batch_size, sequence_len, _ = X.size()
        hidden_outputs = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch_size, self.hidden_size).to(X.device),
                torch.zeros(batch_size, self.hidden_size).to(X.device)
            )
        else:
            h_t, c_t = init_states

        for t in range(sequence_len):
            x_t = X[:, t, :]

            i_t = torch.sigmoid(h_t @ self.W_i + x_t @ self.U_i + c_t @ self.P_i + self.b_i)
            f_t = torch.sigmoid(h_t @ self.W_f + x_t @ self.U_f + c_t @ self.P_f + self.b_f)
            g_t = torch.tanh(h_t @ self.W_c + x_t @ self.U_c + self.b_c)
            c_t = f_t * c_t + i_t * g_t
            o_t = torch.sigmoid(h_t @ self.W_o + x_t @ self.U_o + c_t @ self.P_o + self.b_o)
            h_t = o_t * torch.tanh(c_t)

            hidden_outputs.append(h_t.unsqueeze(0))

        # reshape
        hidden_outputs = torch.cat(hidden_outputs, dim=0).transpose(0, 1).contiguous()

        return hidden_outputs, (h_t, c_t)
