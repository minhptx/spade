import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.base import BaseModule


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim,
                          enc_hid_dim,
                          bidirectional=True,
                          batch_first=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, lengths):

        # src = [batch size, src len]
        embedded = self.dropout(self.embedding(src))

        # embedded = [batch size, src len, emb dim]

        packed_input = pack_padded_sequence(embedded,
                                            lengths,
                                            batch_first=True,
                                            enforce_sorted=False)

        self.rnn.flatten_parameters()
        packed_outputs, hidden = self.rnn(packed_input)

        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        # outputs = [batch size, src len, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, src_len, enc hid dim * 2]

        src_len = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, encoder_outputs=None):

        # input = [batch size]
        # hidden = [batch size, dec hid dim]

        inputs = inputs.unsqueeze(1)

        # input = [batch size, 1]

        embedded = self.dropout(self.embedding(inputs))

        # embedded = [batch size, 1, emb dim]

        self.rnn.flatten_parameters()
        output, hidden = self.rnn(embedded, hidden.unsqueeze(0))

        # output = [batch size, 1, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)

        prediction = self.fc_out(torch.cat((output, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), hidden.squeeze(0)


class AttentionalDecoder(Decoder):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.output_dim = output_dim

        self.attn = Attention(enc_hid_dim, dec_hid_dim)

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim,
                          dec_hid_dim,
                          batch_first=True)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim,
                                output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, encoder_outputs=None):
        if encoder_outputs is None:
            return super().forward(inputs, hidden, encoder_outputs)

        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, src_len, enc hid dim * 2]

        inputs = inputs.unsqueeze(1)

        # input = [batch size, 1]

        embedded = self.dropout(self.embedding(inputs))

        # embedded = [batch size, 1, emb dim]

        attention = self.attention(hidden, encoder_outputs)

        # attention = [batch size, src len]

        attention = attention.unsqueeze(1)

        # attention = [batch size, src len, 1]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(attention, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [batch size, 1, (enc hid dim * 2) + emb dim]

        self.rnn.flatten_parameters()
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [batch size, 1, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        prediction = self.fc_out(torch.cat((output, weighted, embedded),
                                           dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), weighted


class Seq2Seq(BaseModule):
    def __init__(self, hparams, padding_index):
        super().__init__()

        self.hparams = hparams

        self.padding_index = padding_index

        self.encoder = Encoder(
            hparams.vocab_size,
            hparams.enc_emb_dim,
            hparams.enc_hid_dim,
            hparams.dec_hid_dim,
            hparams.dropout_p,
        )

        input_size = hparams.vocab_size

        self.decoder = Decoder(input_size, hparams.dec_emb_dim,
                               hparams.enc_hid_dim, hparams.dec_hid_dim,
                               hparams.dropout_p)

    def encode(self, inputs, lengths):
        encoder_outputs, hidden = self.encoder(inputs, lengths)

        return hidden

    def decode(self, hidden, teacher_forcing_ratio=0.5):
        outputs = []

        input1 = torch.ones(hidden.shape[0], 1) * self.sos_index

        outputs.append(input1)

        for t in range(1, self.hparams.max_length):

            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input1, hidden, None)

            # place predictions in a tensor holding predictions for each token

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            outputs.append(top1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input1 = top1

        return outputs

    def forward(self, inputs, lengths, sm, teacher_forcing_ratio=0.5):

        # inputs = [batch size, src len]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = inputs.shape[0]
        trg_len = inputs.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len,
                              trg_vocab_size).type_as(lengths)
        attentions = torch.zeros(batch_size, trg_len,
                                 self.hparams.dec_hid_dim * 2).type_as(lengths)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(inputs, lengths)

        # first input to the decoder is the <sos> tokens
        input1 = inputs[:, 0]
        outputs[:, 0] = outputs.scatter(1, input1.unsqueeze(1), 1)

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden, attention = self.decoder(input1, hidden,
                                                     encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = output
            attentions[:, t] = attention

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input1 = inputs[:, t] if teacher_force else top1

        return outputs, attentions

    def training_step(self, batch, batch_idx):
        inp, lengths, example = batch

        output, _ = self.forward(inp, lengths,
                                 self.hparams.teacher_forcing_ratio)

        output_dim = output.shape[-1]

        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = inp[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        prediction = output.argmax(1)

        loss = F.cross_entropy(output, trg, ignore_index=self.padding_index)

        mask = trg.ne(self.padding_index)

        acc = prediction.eq(trg).masked_select(mask).sum().float() / mask.sum()

        logs = {"train_loss": loss, "train_acc": acc}

        return {"loss": loss, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        inp, lengths, example = batch

        output, _ = self.forward(inp, lengths,
                                 self.hparams.teacher_forcing_ratio)

        output_dim = output.shape[-1]

        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = inp[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        prediction = output.argmax(1)

        loss = F.cross_entropy(output, trg, ignore_index=self.padding_index)

        mask = trg.ne(self.padding_index)
        acc = prediction.eq(trg).masked_select(mask).sum().float() / mask.sum()

        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs: list):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        logs = {"metrics/val_loss": avg_loss, "metrics/val_acc": avg_acc}
        return {"avg_val_loss": avg_loss, "log": logs, "progress_bar": logs}

    def training_epoch_end(self, outputs: list):
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["train_acc"] for x in outputs]).mean()
        logs = {"metrics/train_loss": avg_loss, "metrics/train_acc": avg_acc}
        return {"avg_train_loss": avg_loss, "log": logs, "progress_bar": logs}

    def configure_optimizers(self):
        return [optim.Adam(self.parameters(), lr=self.hparams.lr)]
