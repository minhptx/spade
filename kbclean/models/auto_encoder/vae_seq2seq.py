import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.auto_encoder.seq2seq import Seq2Seq


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector
    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance
        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        latent_mean = self.hidden_to_mean(cell_output)
        latent_logvar = self.hidden_to_logvar(cell_output)

        return latent_mean, latent_logvar

    @staticmethod
    def reparametrize(mu, logvar):
        std = logvar.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + std * eps


class VSeq2Seq(Seq2Seq):
    def __init__(self, hparams, sos_index, padding_index):
        super().__init__(hparams, padding_index)
        self.hidden2latent = Lambda(hidden_size=hparams.enc_hid_dim,
                                    latent_length=hparams.latent_dim)

        self.sos_index = sos_index

        self.latent2hidden = nn.Linear(hparams.latent_dim, hparams.enc_hid_dim)

    def encode(self, inputs, lengths):
        encoder_outputs, hidden = self.encoder(inputs, lengths)

        latent_mean, latent_logvar = self.hidden2latent(hidden)

        return torch.cat([latent_mean, latent_logvar], dim=1)

    def decode(self, latents, teacher_forcing_ratio=0.5):
        latent_mean, latent_logvar = latents[:, :self.hparams.
                                             latent_dim], latents[:,
                                                                  self.hparams.
                                                                  latent_dim:]
        latent_distribs = Lambda.reparametrize(latent_mean, latent_logvar)
        hidden = self.latent2hidden(latent_distribs)

        outputs = []
        input1 = torch.ones(
            latents.shape[0]).type_as(latents).long() * self.sos_index

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

        return torch.stack(outputs, dim=1)

    def forward(self, inputs, lengths, teacher_forcing_ratio=0.5):
        batch_size = inputs.shape[0]
        trg_len = inputs.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = [
            torch.zeros(batch_size, trg_vocab_size).type_as(inputs).float()
        ]
        attentions = []

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(inputs, lengths)

        latent_mean, latent_logvar = self.hidden2latent(hidden)

        latent = Lambda.reparametrize(latent_mean, latent_logvar)

        hidden = self.latent2hidden(latent)

        # first input to the decoder is the <sos> tokens

        input1 = inputs[:, 0]
        outputs[0].scatter_(1, input1.unsqueeze(1), 1)

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden, attention = self.decoder(input1, hidden,
                                                     encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs.append(output)
            attentions.append(attention)

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input1 = inputs[:, t] if teacher_force else top1

        return torch.stack(outputs, dim=1), (torch.stack(attentions, dim=1),
                                             latent_mean, latent_logvar)

    def training_step(self, batch, batch_idx):
        inp, lengths, _ = batch

        output, (_, mean,
                 logvar) = self.forward(inp, lengths,
                                        self.hparams.teacher_forcing_ratio)

        output_dim = output.shape[-1]

        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = inp[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        prediction = output.argmax(1)

        recon_loss = F.cross_entropy(output,
                                     trg,
                                     ignore_index=self.padding_index)
        kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss.sum(1).mean(0, True)
        loss = kl_loss + recon_loss

        mask = trg.ne(self.padding_index)
        acc = prediction.eq(trg).masked_select(mask).sum().float() / mask.sum()

        logs = {"train_loss": loss, "train_acc": acc, "train_kl_loss": kl_loss}

        return {"loss": loss, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        inp, lengths, _ = batch

        output, (_, mean,
                 logvar) = self.forward(inp, lengths,
                                        self.hparams.teacher_forcing_ratio)

        output_dim = output.shape[-1]

        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = inp[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        prediction = output.argmax(1)

        recon_loss = F.cross_entropy(output,
                                     trg,
                                     ignore_index=self.padding_index)
        kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss.sum(1).mean(0, True)
        loss = kl_loss + recon_loss

        mask = trg.ne(self.padding_index)
        acc = prediction.eq(trg).masked_select(mask).sum().float() / mask.sum()

        return {"val_loss": loss, "val_acc": acc}
