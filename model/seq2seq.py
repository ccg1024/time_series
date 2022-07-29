# -*- coding: utf-8 -*-
"""
in the part, the code is for basic seq2seq model based LSTM.
implement by pytorch, the below will show the parameters of LSTM.

torch.nn.LSTM
-------------
input_size    - the number of expected features of input x.
hidden_size   - the number of features in hidden state h.
num_layers    - number of recurrent layers. Default=1
bias          - Default=True
batch_first   - if True, (batch, seq, feature) instead of (seq, batch, feature)
dropout       - Default=0
bidirectional - if True, become a Bi-LSTM.
proj_size     - if > 0,  will use LSTM with projections of corresponding size. Default=0

:return
outputs - tensor of shape (L, D*H_out) for unbatched input, (L, N, D*H_out) for batch input
        - containing the output feature (h_t) from last layer of the LSTM, for each t. Just
        - means it contain all h_t from h_1 to h_t of last LSTM layer.
h_n     - tensor of shape (D*num_layers, H_out) for unbatched input, (D*num_layers, N, H_out)
        - containing the final hidden state of each LSTM layer.
c_n     - just like h_n, containing the final cell state.

EXAMPLE
-------
>>> lstm = torch.nn.LSTM(10, 20, 2, batch_first=True)
>>> input = torch.randn.(3, 5, 10)
>>> h0 = torch.zeros(3, 2, 20)
>>> c0 = torch.zeros(3, 2, 20)
>>> output, (hn, cn) = lstm(input, (h0, c0))

the code just use 2 layer LSTM for each encoder and decoder.
why need initial multilayer hadden state -> the h_t is the output of a lstm layer
so, the first layer of lstm will put its h_t as a output, and the seconde lstm layer
will use this output as its input, and the seconde lstm layer also need a h_0.
"""

import torch
import torch.nn as nn
from torch import Tensor
from self_attention import SelfAttention


class Encoder(nn.Module):
    """
    the default shape is (seq, batch, feature)
    """

    def __init__(self, input_size: int, hidden_size: int, batch_size: int = 1, layer_num: int = 2,
                 dropout: float = 0.) -> None:
        """

        :param input_size:  the feature number of X, equal to X.shape[-1].
        :param hidden_size: the feature number of hidden layer.
        :param layer_num:   the layer number of LSTM.
        :param dropout:     dropout rate.
        """
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.layer_num,
                            dropout=dropout)

    def forward(self, input_tensor: Tensor, short_term: Tensor = None, long_term: Tensor = None, device: str = 'cpu'):
        """

        : param input_tensor   : training input, shape (seq, batch, features)
        : param short_term     : the hidden states, shape (layer_num, batch_size, hidden_features)
        : param long_term      : the cell states, shape (layer_num, batch_size, hidden_features)
        : return output_tensor : each seq outputs of last lstm layer.
        : return (hn, cn)      : last seq's hidden states and cell states of last lstm layer
        """
        # for test
        if short_term == None or long_term == None:
            short_term = long_term = self.initHidden(self.batch_size, device)

        output_tensor, (hn, cn) = self.lstm(input_tensor, (short_term, long_term))
        return output_tensor, (hn, cn)

    def initHidden(self, batch_size: int, device: str = 'cpu'):
        return torch.zeros(self.layer_num, batch_size, self.hidden_size).to(device)


class Decoder(nn.Module):
    """
    Note: the input_size is equal to target_tensor!
    """

    def __init__(self, input_size: int, hidden_size: int, layer_num: int = 2, dropout: float = 0.) -> None:
        """

        :param input_size:  the feature number of y, equal to y.shape[-1]
        :param hidden_size: the feature number of hidden layer.
        :param layer_num:   the layer number of LSTM.
        :param dropout:     dropout rate.
        """
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.layer_num,
                            dropout=dropout)

    def forward(self, input_tensor: Tensor, short_term: Tensor, long_term: Tensor):
        output_tensor, (hn, cn) = self.lstm(input_tensor, (short_term, long_term))
        return output_tensor, (hn, cn)


class Seq2seq__(nn.Module):
    """
    need remake. See class Seq2Seq
    """
    def __init__(self, encoder, decoder, hidden_size, batch_size):
        """

        :param encoder:     encoder instance.
        :param decoder:     decoder instance.
        :param hidden_size: the hidden size of decoder.
        :param batch_size:  batch_size of input data.
        """
        super(Seq2seq__, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, input_tensor: Tensor, target_tensor: Tensor, hn: Tensor = None, cn: Tensor = None,
                device='cpu'):
        if hn == None or cn == None:
            hn = cn = self.encoder.initHidden(self.batch_size).to(device)
        # h0 = self.encoder.initHidden(self.batch_size).to(device)
        _, (encoder_hn, encoder_cn) = self.encoder(input_tensor, hn, cn)
        _, (decoder_h, _) = self.decoder(target_tensor, encoder_hn, encoder_cn)
        self.linear = self.linear.to(device)
        output_tensor = self.linear(decoder_h[-1])

        return output_tensor


class Seq2Seq(nn.Module):
    """
    reconstruct seq2seq model.
    """
    def __init__(self, input_size: int, hidden_size: int, target_size: int, output_size: int, batch_size: int,
                 layer_num: int = 2, dropout_en: float = 0., dropout_de: float = 0., device: str = 'cpu') -> None:
        """

        : param intpu_size  : number of x features.
        : param hidden_size : number of lstm layer hidden features.
        : param target_size : number of y features.
        : param batch_size  : training batch size.
        : param layer_num   : number of lstm layer in encoder and decoder.
        : param dropout_en  : encoder dropout rate.
        : param dropout_de  : decoder dropout rate.
        : param device      : training device, cpu or cuda.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.dropout_en = dropout_en
        self.dropout_de = dropout_de
        self.device = device

        # create model
        self.encoder = Encoder(self.input_size, self.hidden_size, self.batch_size, self.layer_num,
                               self.dropout_en).to(device)
        self.decoder = Decoder(self.target_size, self.hidden_size, self.layer_num, self.dropout_de).to(device)
        self.linear = nn.Linear(self.hidden_size, self.output_size).to(device)

    def forward(self, input_tensor: Tensor, target_tensor: Tensor, short_term: Tensor = None,
                long_term: Tensor = None, att: bool = False) -> Tensor:
        """

        :param input_tenser: dataset input X.
        :param target_tensor: dataset input y.
        :param short_term: hidden states of encoder.
        :param long_term: cell states of encoder.
        """
        encoder_outputs, (encoder_hn, encoder_cn) = self.encoder(input_tensor, short_term, long_term, device=self.device)

        # TODO: add attention code.
        if att :
            # the encoder_outputs shape: (seq, batch, feature)
            # reshape to (batch, seq, feature)
            encoder_outputs = encoder_outputs.transpose(0, 1)

        _, (decoder_hn, _) = self.decoder(target_tensor, encoder_hn, encoder_cn)
        outputs = self.linear(decoder_hn[-1])
        return outputs, encoder_hn.detach(), encoder_cn.detach()

