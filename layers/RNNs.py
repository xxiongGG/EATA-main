import torch
from torch import nn
from torch.autograd import Variable


class RNN_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_objects):
        super(RNN_Net, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.num_objects = num_objects
        self.rnn = nn.RNN(input_size, hidden_size, self.layer_dim, batch_first=True).to(self.device)
        self.fc = nn.Linear(self.hidden_dim, self.num_objects).to(self.device)

    def forward(self, x):
        x.to(self.device)
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        out, _ = self.rnn(x, h0)
        res = torch.sigmoid(self.fc(out))
        return res


class LSTM_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_objects):
        super(LSTM_Net, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.num_objects = num_objects
        self.lstm = nn.LSTM(input_size, hidden_size, self.layer_dim, True).to(self.device)
        self.fc = nn.Linear(self.hidden_dim, self.num_objects).to(self.device)



    def forward(self, x):
        x.to(self.device)
        out, _ = self.lstm(x)
        res = torch.sigmoid(self.fc(out))
        return res


class GRU_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_objects):
        super(GRU_Net, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.num_objects = num_objects
        self.gru = nn.GRU(input_size, hidden_size, self.layer_dim, batch_first=True).to(self.device)
        self.fc = nn.Linear(self.hidden_dim, self.num_objects).to(self.device)


    def forward(self, x):
        x.to(self.device)
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        out, _ = self.gru(x, h0)
        res = torch.sigmoid(self.fc(out))
        return res
