import torch
from model import seq2seq

# test_encoder = seq2seq.Encoder(1, 4, 1)

test_input = torch.Tensor([[1], [4], [2]])
test_input = test_input.reshape((3, 1, 1))

# outputs, (hn, cn) = test_encoder(test_input)

# test_decoder = seq2seq.Decoder(1, 4)
test_target = torch.Tensor([[12]])
test_target = test_target.reshape((1, 1, 1))

# d_outputs, (dhn, dcn) = test_decoder(test_target, hn, cn)

# test_seq = seq2seq.Seq2seq(test_encoder.to("cuda"), test_decoder.to("cuda"), 4, 1)
s2s = seq2seq.Seq2Seq(1, 5, 1, 1, 1, device='cuda')
print(s2s(test_input.to("cuda"), test_target.to("cuda")))

