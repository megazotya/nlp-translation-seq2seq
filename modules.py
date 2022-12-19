import random
import torch
from torch import nn

def softmax(x, temperature): # when Temperature = 1, this is regular SoftMax
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=0)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):

        #src = [src sent len, batch size]

        # Compute an embedding from the src data and apply dropout to it
        embedded = self.dropout(self.embedding(src))

        #embedded = [src sent len, batch size, emb dim]

        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)

        outputs, (hidden, _) = self.rnn(embedded)

        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer
        if self.bidirectional:
            hidden = hidden.reshape(self.n_layers, 2, -1, self.hid_dim) # [n layers, n directions, batch size, hid dim]
            hidden = hidden.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim) # transpose => [n layers, batch size, n directions, hid dim]
            #hidden => [n layers, batch size, n directions * hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, temperature):
        super().__init__()

        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        self.temperature = temperature

    def forward(self, hidden, encoder_outputs):

        # hidden = [1, batch size, dec_hid_dim]
        # encoder_outputs = [src len, batch size, enc_hid_dim]

        src_len = encoder_outputs.shape[0]

        #repeat decoder hidden state src_len times
        hidden = hidden.repeat(src_len, 1, 1)

        #hidden = [src len, batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))

        #energy = [src len, batch size, dec hid dim]

        attention = self.v(energy).squeeze(2)

        #attention= [src len, batch size]

        return softmax(attention, temperature=self.temperature)
        #return attention [src len, batch size]

class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers,
                 dropout, attention):

        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        # use GRU
        self.rnn = nn.GRU(
            input_size = emb_dim + enc_hid_dim,
            hidden_size = dec_hid_dim,
            num_layers = n_layers,
            dropout = dropout
        )

        # linear layer to get next word
        self.out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]

        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]

        input = input.unsqueeze(0) # because only one word, no words sequence

        #input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        #embedded = [1, batch size, emb dim]

        if hidden.shape[0] != 1:
            #when number of layers for hidden > 1
            #we'll get the hidden state of the last layer
            hidden = hidden[-1, :, :].unsqueeze(0)

            #hidden = [1, batch size, hid dim]

        # get weighted sum of encoder_outputs
        a = self.attention(hidden, encoder_outputs)
        weighted = (a.unsqueeze(2) *  encoder_outputs).sum(dim = 0)
        #weighted = [batch size, enc_hid_dim]

        # concatenate weighted sum and embedded, break through the GRU
        rnn_input = torch.cat((weighted.unsqueeze(0), embedded), dim = 2)
        # [1, batch size, emb dim + enc_hid_dim]

        # get predictions
        output, hidden = self.rnn(rnn_input, hidden)
        #output = [1, batch size, dec_hid_dim]

        #prediction = [batch size, output dim]

        prediction = self.out(torch.cat((output.squeeze(0), weighted, embedded.squeeze(0)), dim = -1))

        return prediction, hidden


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        if encoder.bidirectional:
            assert encoder.hid_dim * 2 == decoder.dec_hid_dim, \
                "Hidden dimensions of encoder and decoder must be equal!"
        else:
            assert encoder.hid_dim == decoder.dec_hid_dim, \
                "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):

        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden = self.encoder(src)

        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(1, trg_len):

            #insert input token embedding, previous hidden state
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, enc_states)

            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #get the highest predicted token from our predictions
            top1 = output.argmax(-1)
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
