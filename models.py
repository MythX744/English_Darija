import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
import random


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # Store num_layers even if we don't use it for base LSTM

        # Input gates with improved initialization
        self.i2i_g = nn.Linear(input_size, hidden_size)
        self.i2f_g = nn.Linear(input_size, hidden_size)
        self.i2o_g = nn.Linear(input_size, hidden_size)

        # Hidden gates
        self.h2i_g = nn.Linear(hidden_size, hidden_size)
        self.h2f_g = nn.Linear(hidden_size, hidden_size)
        self.h2o_g = nn.Linear(hidden_size, hidden_size)

        # Cell state gates
        self.i2g = nn.Linear(input_size, hidden_size)
        self.h2g = nn.Linear(hidden_size, hidden_size)

        # Layer normalization for better stability
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Initialize forget gate bias to 1 to help with gradient flow
        self.i2f_g.bias.data.fill_(1.0)
        self.h2f_g.bias.data.fill_(1.0)

    def forward(self, input_seq, states=None):
        batch_size = input_seq.size(0)
        seq_len = input_seq.size(1)

        # Initialize states if not provided
        if states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)
        else:
            h_t, c_t = states
            h_t = h_t.squeeze(1)
            c_t = c_t.squeeze(1)

        hidden_seq = []

        for t in range(seq_len):
            x_t = input_seq[:, t]

            # Calculate gates with improved gradient flow
            input_gate = torch.sigmoid(self.h2i_g(h_t) + self.i2i_g(x_t))
            forget_gate = torch.sigmoid(self.h2f_g(h_t) + self.i2f_g(x_t))
            output_gate = torch.sigmoid(self.h2o_g(h_t) + self.i2o_g(x_t))

            # Calculate cell state candidate
            g = torch.tanh(self.h2g(h_t) + self.i2g(x_t))

            # Update cell state with layer normalization
            c_t = forget_gate * c_t + input_gate * g
            c_t = self.layer_norm(c_t)

            # Calculate hidden state
            h_t = output_gate * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(1))

        # Combine hidden states
        hidden_seq = torch.cat(hidden_seq, dim=1)

        return hidden_seq, (h_t.unsqueeze(1), c_t.unsqueeze(1))


class TranslationModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, hidden_size, cell_type=LSTM, num_layers=2):
        super(TranslationModel, self).__init__()

        # Embeddings
        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=0)
        self.decoder_embedding = nn.Embedding(output_vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)

        # LSTM cells
        self.encoder = cell_type(embedding_dim, hidden_size, num_layers)
        self.decoder = cell_type(embedding_dim, hidden_size, num_layers)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, batch, teacher_forcing_ratio=0.5):
        source = batch['eng']
        target = batch['darija']

        batch_size = source.size(0)
        max_len = target.size(1) - 1

        # Encoding
        enc_embedded = self.dropout(self.encoder_embedding(source))
        encoder_outputs, encoder_states = self.encoder(enc_embedded)

        # Initialize decoder input
        decoder_input = target[:, 0].unsqueeze(1)  # START token
        decoder_states = encoder_states
        outputs = torch.zeros(batch_size, max_len, self.fc.out_features).to(source.device)

        # Decoding
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        for t in range(max_len):
            dec_embedded = self.dropout(self.decoder_embedding(decoder_input))
            decoder_output, decoder_states = self.decoder(dec_embedded, decoder_states)

            decoder_output = self.layer_norm(decoder_output)
            prediction = self.fc(decoder_output)
            outputs[:, t:t + 1] = prediction

            decoder_input = target[:, t + 1].unsqueeze(1) if use_teacher_forcing else prediction.argmax(2)

        return outputs


class Peehole(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Peehole, self).__init__()
        self.hidden_size = hidden_size

        # Input gates
        self.i2i_g = nn.Linear(input_size, hidden_size)
        self.i2f_g = nn.Linear(input_size, hidden_size)
        self.i2o_g = nn.Linear(input_size, hidden_size)

        # Hidden gates
        self.h2i_g = nn.Linear(hidden_size, hidden_size)
        self.h2f_g = nn.Linear(hidden_size, hidden_size)
        self.h2o_g = nn.Linear(hidden_size, hidden_size)

        # Cell state gates
        self.i2g = nn.Linear(input_size, hidden_size)
        self.h2g = nn.Linear(hidden_size, hidden_size)

        # Peephole connections
        self.c2i_g = nn.Linear(hidden_size, hidden_size)
        self.c2f_g = nn.Linear(hidden_size, hidden_size)
        self.c2o_g = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.hidden2out = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_seq, states=None):
        batch_size = input_seq.size(0)
        seq_len = input_seq.size(1)

        # Initialize states
        if states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)
        else:
            h_t, c_t = states
            h_t = h_t.squeeze(1)
            c_t = c_t.squeeze(1)

        hidden_seq = []

        for t in range(seq_len):
            x_t = input_seq[:, t]

            # Calculate gates with peephole connections
            input_gate = F.sigmoid(
                self.h2i_g(h_t) +
                self.i2i_g(x_t) +
                self.c2i_g(c_t)
            )
            forget_gate = F.sigmoid(
                self.h2f_g(h_t) +
                self.i2f_g(x_t) +
                self.c2f_g(c_t)
            )

            # Cell state candidate
            g = F.tanh(self.h2g(h_t) + self.i2g(x_t))

            # Update cell state
            c_t = forget_gate * c_t + input_gate * g

            # Output gate with updated cell state
            output_gate = F.sigmoid(
                self.h2o_g(h_t) +
                self.i2o_g(x_t) +
                self.c2o_g(c_t)
            )

            # Update hidden state
            h_t = output_gate * F.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(1))

        # Combine all hidden states
        hidden_seq = torch.cat(hidden_seq, dim=1)

        # Return final states in the correct format
        final_h = h_t.unsqueeze(1)
        final_c = c_t.unsqueeze(1)

        return hidden_seq, (final_h, final_c)

    def initHidden(self, batch_size=1):
        return torch.zeros(batch_size, 1, self.hidden_size).cuda()

    def initCellState(self, batch_size=1):
        return torch.zeros(batch_size, 1, self.hidden_size).cuda()


class WMC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(WMC, self).__init__()
        self.hidden_size = hidden_size

        # Input gates
        self.i2i_g = nn.Linear(input_size, hidden_size)
        self.i2f_g = nn.Linear(input_size, hidden_size)
        self.i2o_g = nn.Linear(input_size, hidden_size)

        # Hidden gates
        self.h2i_g = nn.Linear(hidden_size, hidden_size)
        self.h2f_g = nn.Linear(hidden_size, hidden_size)
        self.h2o_g = nn.Linear(hidden_size, hidden_size)

        # Cell state gates
        self.i2g = nn.Linear(input_size, hidden_size)
        self.h2g = nn.Linear(hidden_size, hidden_size)

        # Working memory connections
        self.c2i_g = nn.Linear(hidden_size, hidden_size)
        self.c2f_g = nn.Linear(hidden_size, hidden_size)
        self.c2o_g = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.hidden2out = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_seq, states=None):
        batch_size = input_seq.size(0)
        seq_len = input_seq.size(1)

        # Initialize states if not provided
        if states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)
        else:
            h_t, c_t = states
            h_t = h_t.squeeze(1)
            c_t = c_t.squeeze(1)

        # Process each timestep
        hidden_seq = []

        for t in range(seq_len):
            x_t = input_seq[:, t]

            # Calculate gates with working memory
            input_gate = F.sigmoid(
                self.h2i_g(h_t) +
                self.i2i_g(x_t) +
                F.tanh(self.c2i_g(c_t))
            )
            forget_gate = F.sigmoid(
                self.h2f_g(h_t) +
                self.i2f_g(x_t) +
                F.tanh(self.c2f_g(c_t))
            )

            # Cell state candidate
            g = F.tanh(self.h2g(h_t) + self.i2g(x_t))

            # Update cell state
            c_t = forget_gate * c_t + input_gate * g

            # Output gate with working memory
            output_gate = F.sigmoid(
                self.h2o_g(h_t) +
                self.i2o_g(x_t) +
                F.tanh(self.c2o_g(c_t))
            )

            # Update hidden state
            h_t = output_gate * F.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(1))

        # Combine all hidden states
        hidden_seq = torch.cat(hidden_seq, dim=1)

        # Return final states in the correct format
        final_h = h_t.unsqueeze(1)
        final_c = c_t.unsqueeze(1)

        return hidden_seq, (final_h, final_c)

    def initHidden(self, batch_size=1):
        return torch.zeros(batch_size, 1, self.hidden_size).cuda()

    def initCellState(self, batch_size=1):
        return torch.zeros(batch_size, 1, self.hidden_size).cuda()


class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, input_seq, states=None):
        outputs, (hidden, cell) = self.lstm(input_seq, states)
        return outputs, (hidden, cell)
