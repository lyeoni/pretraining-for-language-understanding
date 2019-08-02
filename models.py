import torch
import torch.nn as nn

class LSTMLM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, 
                 n_layers, dropout_p, is_bidirectional):

        super(LSTMLM, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.is_bidirectional = is_bidirectional

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout_p,
                            bidirectional=is_bidirectional,
                            batch_first=True)
        if is_bidirectional:
            self.fc = nn.Linear(hidden_size*2, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
        else:            
            self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        # |input| = (batch_size, seq_len)
        
        embeds = self.embedding(input)
        # |emb| = (batch_size, seq_len, embedding_size)
        
        lstm_out, hidden = self.lstm(embeds)
        # |hidden[0]|, |hidden[1]| = (n_layers*n_directions, batch_size, hidden_size)
        # |lstm_out| = (batch_size, seq_len, hidden_size*n_directions)
        
        if self.is_bidirectional:
            fc_out = self.fc2(self.fc(lstm_out))
        else:
            fc_out = self.fc(lstm_out)
        # |fc_out| = (batch_size, seq_len, hidden_size)
        output = self.softmax(fc_out)
        # |output| = (batch_size, seq_len, vocab_len)
        
        return output