import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bi=False):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bi, batch_first=True)
        self.out = nn.Linear(hidden_size * (1 + bi), output_size)

    def forward(self, inp_seq):
        out, _ = self.lstm(inp_seq)
        out = out[:, -1, :]
        out = self.out(out)

        out = F.softmax(out, dim=1)
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bi=False):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=bi, batch_first=True)

        self.out = nn.Linear(hidden_size * (1 + bi), output_size)

    def forward(self, inp_seq):
        out, _ = self.lstm(inp_seq)
        out = out[:, -1, :]
        out = self.out(out)

        out = F.softmax(out)
        return out


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
