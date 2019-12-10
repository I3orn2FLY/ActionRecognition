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

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        # self.fc1 = nn.Linear(side * side, 500)
        # self.fc2 = nn.Linear(500, 500)
        # self.fc3 = nn.Linear(500, 2000)
        # self.fc4 = nn.Linear(2000, nb_class)
        #
        self.net = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, img):
        return self.net(img)

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



