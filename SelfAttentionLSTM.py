import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_weights = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (64 ** 0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output + x


class SelfAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, attention_dim, dropout_rate=0.5, num_directions=1):
        super(SelfAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, )
        self.self_attention = SelfAttention(hidden_dim * num_directions, attention_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        #self.projection_layer = nn.Linear(hidden_dim * num_directions + input_dim, input_dim)  # Added projection layer

    def forward(self, x, hidden):
        lstm_output, (h_n, c_n) = self.lstm(x, hidden)
        lstm_output = self.dropout(lstm_output)  # Apply dropout to LSTM output
        attention_output = self.self_attention(lstm_output)
        #combined_output = torch.cat((x, attention_output), dim=-1)  # Combine input x and attention_output
        #projected_output = self.projection_layer(combined_output)  # Project combined_output back to the original input_dim
        return attention_output, (h_n, c_n)