import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.base import BaseModule


class RNNLanguageModel(BaseModule):
    def __init__(self, hparams, char_encoder):
        super(RNNLanguageModel, self).__init__()

        self.hparams = hparams
        self.padding_index = char_encoder.padding_index
        self.char_encoder = char_encoder

        self.dropout = nn.Dropout(p=0.5)
        self.embedding = nn.Embedding(hparams.vocab_size, hparams.emb_dim)
        self.encoder = nn.LSTM(hparams.emb_dim,
                               hparams.hid_dim,
                               batch_first=True)
        self.decoder = nn.Linear(hparams.hid_dim, hparams.vocab_size)

    def forward(self, inputs, lengths):
        embedded = self.dropout(self.embedding(inputs))

        packed = pack_padded_sequence(embedded,
                                      lengths,
                                      batch_first=True,
                                      enforce_sorted=False)
        self.encoder.flatten_parameters()
        packed_output, hidden = self.encoder(packed)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    def predict(self, inputs, lengths):
        outputs = self.forward(inputs[:, :-1], lengths - 1)
        outputs = torch.log_softmax(outputs, dim=2)
        probs = torch.gather(outputs, 2, inputs[:, 1:].unsqueeze(2))
        return torch.mean(probs, dim=1)

    def sample(self, size=10):
        outputs = torch.zeros(size, self.hparams.max_length).long().cuda()
        lengths = []
        for i in range(size):
            input1 = torch.ones(1, 1).long().cuda() * self.padding_index
            for j in range(self.hparams.max_length):
                prob = self.forward(input1, torch.tensor([1]))
                topi = torch.argmax(prob, dim=2)
                if topi[0, 0].item() == self.char_encoder.eos_index:
                    lengths.append(j + 1)
                    break
                outputs[i, j] = topi[0, 0]
        return self.char_encoder.batch_decode(outputs, lengths)

    def training_step(self, batch, batch_idx):
        inputs, lengths, _ = batch

        x = inputs[:, :-1]
        y = inputs[:, 1:]

        y_hat = self.forward(x, lengths - 1)
        output_size = y_hat.shape[-1]

        y = y[:, :y_hat.shape[1]].reshape(-1)
        y_hat = y_hat.reshape(-1, output_size)

        loss = F.cross_entropy(y_hat, y, ignore_index=self.padding_index)
        y_pred = torch.argmax(y_hat, dim=1)

        mask = y.ne(self.padding_index)
        acc = y_pred.eq(y).masked_select(mask).sum().float() / mask.sum()

        logs = {"train_loss": loss, "train_acc": acc}

        return {"loss": loss, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        inputs, lengths, _ = batch

        x = inputs[:, :-1]
        y = inputs[:, 1:]

        y_hat = self.forward(x, lengths - 1)
        output_size = y_hat.shape[-1]

        y = y[:, :y_hat.shape[1]].reshape(-1)
        y_hat = y_hat.reshape(-1, output_size)

        loss = F.cross_entropy(y_hat, y, ignore_index=self.padding_index)
        y_pred = torch.argmax(y_hat, dim=1)

        mask = y.ne(self.padding_index)
        acc = y_pred.eq(y).masked_select(mask).sum().float() / mask.sum()

        return {"val_loss": loss, "val_acc": acc}

    def on_epoch_end(self):
        sents = self.sample()
        self.logger.experiment.add_text("Generated", "  \n".join(sents),
                                        self.global_step)

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)]
