import torch
import torch.nn as nn

class LSTMLM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, 
                 n_layers, dropout_p):

        super(LSTMLM, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout_p,
                            bidirectional=False,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        # |input| = (batch_size, max_seq_len-1)

        embeds = self.embedding(input)
        # |embeds| = (batch_size, max_seq_len-1, embedding_size)

        lstm_out, hidden = self.lstm(embeds)
        # If bidirectional=True, num_directions is 2, else it is 1.
        # |lstm_out| = (batch_size, max_seq_len-1, num_directions*hidden_size)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size)

        fc_out = self.fc(lstm_out)
        # |fc_out| = (batch_size, max_seq_len-1, hidden_size//2)
        output = self.softmax(self.fc2(fc_out))
        # |output| = (batch_size, max_seq_len-1, output_size)

        return output

class BiLSTMLM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size,
                 n_layers, dropout_p):

        super(BiLSTMLM, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout_p,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        # |input| = (batch_size, max_seq_len)
        
        embeds = self.embedding(input)
        # |embeds| = (batch_size, max_seq_len, embedding_size)
        
        lstm_out, hidden = self.lstm(embeds)
        # If bidirectional=True, num_directions is 2, else it is 1.
        # |lstm_out| = (batch_size, max_seq_len, num_directions*hidden_size)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size)
        
        forward_out = lstm_out[:, :-2, :self.hidden_size]
        backward_out = lstm_out[:, 2:, self.hidden_size:]
        # |forward_out| = (batch_size, max_seq_len-2, hidden_size)
        # |backward_out| = (batch_size, max_seq_len-2, hidden_size)
        
        context = torch.cat((forward_out, backward_out), dim=-1)
        # |context| = (batch_size, max_seq_len-2, 2*hidden_size)
        
        fc_out = self.fc2(self.fc(context))
        # |fc_out| = (batch_size, max_seq_len-2, hidden_size//2)
        output = self.softmax(self.fc3(fc_out))
        # |output| = (batch_size, max_seq_len-2, output_size)
        
        return output