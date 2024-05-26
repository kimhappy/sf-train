import torch
import torch.nn as nn

class _Model(nn.Module):
    def __init__(
        self       ,
        input_size ,
        output_size,
        hidden_size):
        super(_Model, self).__init__()
        self.rec    = nn.LSTM  (input_size , hidden_size, batch_first = True)
        self.lin    = nn.Linear(hidden_size, output_size)
        self.hidden = None

    def forward(self, x):
        rec_out, self.hidden = self.rec(x, self.hidden)
        lin_out              = self.lin(rec_out)
        return x[ ..., 0 ] + lin_out[ ..., 0 ]

    def detach_hidden(self):
        self.hidden = tuple([h.clone().detach() for h in self.hidden])

    def params(self):
        weight_i = self.state_dict()[ 'rec.weight_ih_l0' ].flatten()
        weight_h = self.state_dict()[ 'rec.weight_hh_l0' ].flatten()
        bias_i   = self.state_dict()[ 'rec.bias_ih_l0'   ].flatten()
        bias_h   = self.state_dict()[ 'rec.bias_hh_l0'   ].flatten()
        weight   = self.state_dict()[ 'lin.weight'       ].flatten()
        bias     = self.state_dict()[ 'lin.bias'         ].flatten()
        return torch.cat([weight_i, weight_h, bias_i, bias_h, weight, bias]).cpu().numpy()
