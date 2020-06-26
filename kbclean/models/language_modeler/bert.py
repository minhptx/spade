import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from transformers import BertConfig, BertForMaskedLM

from models.base import BaseModule


class BertLanguageModel(BaseModule):
    def __init__(self, hparams, tokenizer):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.bert = BertForMaskedLM(BertConfig(vocab_size=hparams["vocab_size"]))

    def forward(self, inputs):
        return self.bert.forward(inputs, masked_lm_labels=inputs)

    def encode(self, inputs):
        encoded_batch = self.tokenizer.encode_batch(inputs)
        return pad_sequence([torch.tensor(x.ids) for x in encoded_batch], batch_first=True)

    def predict(self, inputs):
        encoded_inputs = self.encode(inputs)
        loss, logits = self.bert.forward(encoded_inputs, masked_lm_labels=encoded_inputs)[:2]
        outputs = torch.log_softmax(logits, dim=2)
        probs = torch.gather(outputs, 2, encoded_inputs.unsqueeze(2))

        return torch.exp(torch.mean(probs, dim=1))

    def training_step(self, batch, batch_idx):
        loss, scores = self.forward(batch)[:2]

        preds = torch.flatten(torch.argmax(scores, dim=2))
        true_labels = torch.flatten(batch)

        acc = (preds == true_labels).float().mean()
        loss = loss.float() / batch.shape[0]

        logs = {"train_loss": loss, "train_acc": acc}

        return {"loss": loss, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        loss, scores = self.forward(batch)[:2]

        preds = torch.flatten(torch.argmax(scores, dim=2))
        true_labels = torch.flatten(batch)

        acc = (preds == true_labels).float().mean()

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.Adam(self.parameters(), lr=self.hparams.lr)]


class LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model, cfg, trunc_and_reshape=True):
        super(LMHead, self).__init__()
        self.n_embd = cfg.n_embd
        embed_shape = model.embed.weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight  # Tied weights
        self.trunc_and_reshape = trunc_and_reshape  # XD

    def forward(self, h):
        # Truncated Language modeling logits (we remove the last token)
        h_trunc = (
            h[:, :-1].contiguous().view(-1, self.n_embd) if self.trunc_and_reshape else h
        )  # XD
        lm_logits = self.decoder(h_trunc)
        return lm_logits
