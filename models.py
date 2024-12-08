import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset


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


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_size, cell_type=LSTM, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=0)
        self.cell_type = cell_type

        if cell_type == StackedLSTM:
            self.lstm = cell_type(embedding_dim, hidden_size, num_layers)
            self.num_layers = num_layers  # Add this line
            self.hidden_size = hidden_size  # Add this line
        else:
            self.lstm = cell_type(embedding_dim, hidden_size, num_layers)

    def forward(self, src):
        embedded = self.embedding(src)

        # Initialize hidden state for StackedLSTM
        if self.cell_type == StackedLSTM:
            batch_size = src.size(0)
            hidden = self.lstm.init_hidden(batch_size)
            outputs, states = self.lstm(embedded, hidden)
        else:
            outputs, states = self.lstm(embedded)

        return outputs, states


class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_size, cell_type=LSTM, num_layers=1):
        super().__init__()
        # Convert Darija words to vectors
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim, padding_idx=0)

        # Process embedded sequences (hidden_size matches encoder)
        if cell_type == StackedLSTM:
            self.lstm = cell_type(embedding_dim, hidden_size, num_layers)
        else:
            self.lstm = cell_type(embedding_dim, hidden_size, num_layers)

        # Convert LSTM outputs to word predictions
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_vocab_size)
        )

    def forward(self, input, states):
        # input shape: [batch_size, 1]
        embedded = self.embedding(input)  # shape: [batch_size, 1, embedding_dim]
        lstm_output, states = self.lstm(embedded, states)  # lstm_output: [batch_size, 1, hidden_size]
        prediction = self.output_layer(lstm_output[:, -1])  # shape: [batch_size, output_vocab_size]
        return prediction, states


class TranslationModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, hidden_size, cell_type=LSTM, num_layers=1):
        super().__init__()
        self.encoder = Encoder(input_vocab_size, embedding_dim, hidden_size, cell_type)
        self.decoder = Decoder(output_vocab_size, embedding_dim, hidden_size, cell_type)
        self.max_len = 15

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        max_len = trg.shape[1] if trg is not None else self.max_len
        outputs = torch.zeros(batch_size, max_len, self.decoder.output_layer[-1].out_features).to(src.device)

        # Encode input sequence
        _, encoder_states = self.encoder(src)

        # Initialize decoder with START token
        decoder_input = torch.tensor([[2]] * batch_size).cuda()  # 2 is START token index
        decoder_states = encoder_states

        # Generate translation one word at a time
        for t in range(max_len):
            prediction, decoder_states = self.decoder(decoder_input, decoder_states)
            outputs[:, t] = prediction

            # Teacher forcing
            if t < max_len - 1:
                use_teacher_forcing = (torch.rand(1).item() < teacher_forcing_ratio) and (trg is not None)
                decoder_input = trg[:, t:t + 1] if use_teacher_forcing else prediction.argmax(1).unsqueeze(1)

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
        super(StackedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, input, hidden=None):
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(input.size(0))
        output, hidden = self.lstm(input, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device='cuda')
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device='cuda')
        return (h0, c0)


'''
class TranslationModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, word_darija, cell_type=LSTM, num_layers=1,
                 dropout_rate=0.5):
        super(TranslationModel, self).__init__()
        self.hidden_size = hidden_size
        self.word_darija = word_darija
        self.num_layers = num_layers
        self.max_len = 50

        # Improved embeddings with positional encoding
        self.encoder_embedding = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size, padding_idx=0),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_size)
        )

        self.decoder_embedding = nn.Sequential(
            nn.Embedding(output_size, hidden_size, padding_idx=0),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_size)
        )

        # LSTM layers
        self.encoder = cell_type(hidden_size, hidden_size)
        self.decoder = cell_type(hidden_size, hidden_size)

        # Output projection with residual connection
        self.out = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        max_len = trg.shape[1] if trg is not None else self.max_len

        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, max_len, len(self.word_darija)).to(src.device)

        # Encoder
        encoder_embedded = self.encoder_embedding(src)
        encoder_outputs, encoder_states = self.encoder(encoder_embedded)

        # Initialize decoder input
        decoder_input = torch.tensor([[self.word_darija['<START>']] * batch_size]).T.to(src.device)
        decoder_states = encoder_states

        # Improved teacher forcing schedule
        teacher_force = torch.rand(max_len) < teacher_forcing_ratio

        # Decoder
        for t in range(max_len):
            # Embed decoder input
            decoder_embedded = self.decoder_embedding(decoder_input)

            # Forward through decoder
            decoder_output, decoder_states = self.decoder(decoder_embedded, decoder_states)

            # Project to vocabulary size
            prediction = self.out(decoder_output[:, -1])
            outputs[:, t] = prediction

            # Teacher forcing with schedule
            if t < max_len - 1:
                decoder_input = trg[:, t:t + 1] if teacher_force[t] and trg is not None else prediction.argmax(
                    1).unsqueeze(1)

        return outputs
'''
