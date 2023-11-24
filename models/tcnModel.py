import torch.nn as nn
from models.temporalCNN import TemporalConvNet


class TcnModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TcnModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y = self.tcn(x)
        o = self.linear(y[:, :, -1])
        return nn.functional.log_softmax(o, dim=1)


