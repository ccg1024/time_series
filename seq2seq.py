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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    the default shape is (batch, seq, feature)
    """

    def __init__(self, input_size, hidden_size, layer_num=2, dropout=0.) -> None:
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.layer_num,
                            dropout=dropout)

    def forward(self, input_tensor, short_term, long_term):
        output_tensor, (hn, cn) = self.lstm(input_tensor, (short_term, long_term))
        return output_tensor, (hn, cn)

    def initHidden(self, batch_size):
        return torch.zeros(self.layer_num, batch_size, self.hidden_size, device=device)


class Decoder(nn.Module):
    """
    Note: the input_size is equal to target_tensor!
    """

    def __init__(self, input_size, hidden_size, layer_num=2, dropout=0.) -> None:
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.layer_num,
                            dropout=dropout)

    def forward(self, input_tensor, short_term, long_term):
        output_tensor, (hn, cn) = self.lstm(input_tensor, (short_term, long_term))
        return output_tensor, (hn, cn)


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, batch_size):
        super(Seq2seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, input_tensor, target_tensor):
        h0 = self.encoder.initHidden(self.batch_size)
        encode_output, (encoder_hn, encoder_cn) = self.encoder(input_tensor, (h0, h0))
        _, (decoder_h, _) = self.decoder(target_tensor, (encoder_hn, encoder_cn))

        output_tensor = self.linear(decoder_h)

        return output_tensor
