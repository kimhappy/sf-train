import torch.nn as nn

class _Model(nn.Module):
    def __init__(
        self       ,
        input_size ,
        output_size,
        hidden_size):
        super(_Model, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.rec         = nn.LSTM  (input_size , hidden_size, batch_first = True)
        self.lin         = nn.Linear(hidden_size, output_size)
        self.hidden      = None

    def forward(self, x):
        rec_out, self.hidden = self.rec(x, self.hidden)
        lin_out              = self.lin(rec_out)
        return x[ ..., 0 ] + lin_out[ ..., 0 ]

    def detach_hidden(self):
        self.hidden = tuple([h.clone().detach() for h in self.hidden])
