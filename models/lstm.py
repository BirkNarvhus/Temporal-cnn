import torch
import torch.nn as nn


class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, classes, device):
        super(Lstm, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.directions = 2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout,
                            bias=True)
        self.linear = nn.Linear(hidden_size * self.directions, classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.directions, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.directions, x.size(0), self.hidden_size).to(self.device)
        x, hidden = self.lstm(x, (h0, c0))

        final_state = \
            hidden[0].view(1, self.num_layers, self.directions, x.size(0),
                           self.hidden_size)[:, -1]

        final_state = final_state.squeeze()
        h_1, h_2 = final_state[0], final_state[1]
        final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states
        out = self.linear(final_hidden_state)

        return out
