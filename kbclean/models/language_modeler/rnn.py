import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNLanguageModel(LightningModule):
    def __init__(self, hparams):
        super(RNNLanguageModel, self).__init__()

        self.hparams = hparams

        self.dropout = nn.Dropout(p=0.5)
        self.embedding = nn.Embedding(hparams.vocab_size, hparams.emb_dim)
        self.encoder = nn.LSTM(hparams.emb_dim, hparams.hid_dim)
        self.decoder = nn.Linear(hparams.hid_dim, hparams.vocab_size)

    def forward(self, inputs, lengths):
        embedded = self.dropout(self.embedding)

        packed = pack_padded_sequence(embedded, lengths)
        packed_output, hidden = self.encoder(packed)

        output, _ = pad_packed_sequence(packed)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1),
                            decoded.size(1)), hidden

    def training_step(self, batch, batch_idx):
        x = batch[:, :-1]
        y = batch[:, 1:]

        y_hat = self.forward(x)
        output_size = y_hat.shape[-1]

        y_hat = y_hat.view(-1, output_size)
        y = y.view(-1)

        loss = F.cross_entropy(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)

        acc = (y_pred == y).sum() / x.shape[0]

        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x = batch[:, :-1]
        y = batch[:, 1:]

        y_hat = self.forward(x)
        output_size = y_hat.shape[-1]

        y_hat = y_hat.view(-1, output_size)
        y = y.view(-1)

        loss = F.cross_entropy(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)

        acc = (y_pred == y).sum() / x.shape[0]

        return {"val_loss": loss, "val_acc": acc}
