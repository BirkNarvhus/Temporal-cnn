import torch
import torch.nn as nn


class Rnn (nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, classes, device):
        super(Rnn, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,  batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.rnn(x, h0)
        out = self.linear(x[:, -1, :])

        return nn.functional.log_softmax(out)



