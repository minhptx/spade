import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.base import BaseModule


class RGANDiscriminator(BaseModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.model = nn.Sequential(
            nn.Linear(hparams.input_dim, hparams.gen_hid_dim),
            nn.ReLU(),
            nn.Linear(hparams.gen_hid_dim, hparams.latent_dim),
            nn.ReLU(),
            nn.Linear(hparams.latent_dim, 2),
        )

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x = batch
        y = torch.zeros(x.shape[0]).type_as(x).long()
        y_hat = torch.log_softmax(self.forward(x), dim=1)
        loss = F.nll_loss(y_hat, y.type_as(y_hat).long())
        y_pred = torch.argmax(y_hat, dim=1)
        acc = (y_pred == y).sum().float() / y.shape[0]
        logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        x = batch
        y = torch.zeros(x.shape[0]).type_as(x).long()
        y_hat = torch.log_softmax(self.forward(x), dim=1)
        loss = F.nll_loss(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        acc = (y_pred == y).sum().float() / y.shape[0]
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.Adam(self.parameters(), lr=self.hparams.lr)]


class OneClassGAN(BaseModule):
    def __init__(self,
                 hparams,
                 r_discriminator,
                 seq2seq=None,
                 character_encoder=None):
        super().__init__()

        self.hparams = hparams

        self.seq2seq = seq2seq

        self.generator = nn.Sequential(
            nn.Linear(hparams.latent_dim, hparams.gen_hid_dim),
            nn.ReLU(),
            nn.Linear(hparams.gen_hid_dim, hparams.input_dim),
            nn.Tanh(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(hparams.input_dim, hparams.gen_hid_dim),
            nn.ReLU(),
            nn.Linear(hparams.gen_hid_dim, hparams.latent_dim),
            nn.ReLU(),
            nn.Linear(hparams.latent_dim, 2),
        )

        self.r_discriminator = r_discriminator
        self.character_encoder = character_encoder

        self.x_gen = None

    def forward(self, inputs):
        return self.generator(inputs)

    @staticmethod
    def pull_away_term(s1, s2):
        """Calculate Pulling-away Term(PT)."""

        n = s1.size(0)
        s1 = F.normalize(s1, p=2, dim=1)
        s2 = F.normalize(s2, p=2, dim=1)

        repeated_s1 = s1.unsqueeze(1).repeat(1, s2.size(0), 1)
        repeated_s2 = s2.unsqueeze(0).repeat(s1.size(0), 1, 1)

        f_pt = repeated_s1.mul(repeated_s2).sum(-1).pow(2)
        f_pt = torch.tril(f_pt, -1).sum().mul(2).div((n * (n - 1)))

        # f_PT = (S1.mul(S2).sum(-1).pow(2).sum(-1)-1).sum(-1).div(n*(n-1))
        return f_pt

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch

        if optimizer_idx == 0:
            # train discriminator
            z = (torch.rand(x.shape[0], self.hparams.latent_dim).type_as(x) -
                 0.5) * 2

            self.x_gen = self.generator(z)
            y_fake_prob = torch.softmax(self.discriminator(self.x_gen), dim=1)
            y_real_prob = torch.softmax(self.discriminator(x), dim=1)

            dis_fake_loss = F.cross_entropy(
                y_fake_prob,
                torch.ones(x.shape[0]).type_as(x).long())
            dis_real_loss = F.cross_entropy(
                y_real_prob,
                torch.zeros(x.shape[0]).type_as(x).long())

            ent_real_loss = -(y_real_prob *
                              y_real_prob.log()).sum(dim=1).mean()

            loss = dis_fake_loss + dis_real_loss + 1.85 * ent_real_loss
            logs = {"d_loss": loss}

            return {"loss": loss, "log": logs, "progress_bar": logs}

        elif optimizer_idx == 1:
            # train generator
            y_real_hat = self.discriminator(x)
            y_fake_hat = self.discriminator(self.x_gen.detach())
            r_y_fake_hat = self.r_discriminator(self.x_gen.detach())
            r_y_fake_prob = torch.softmax(r_y_fake_hat, dim=1)

            fm_loss = F.mse_loss(y_real_hat, y_fake_hat)
            pt_term = OneClassGAN.pull_away_term(r_y_fake_hat, r_y_fake_hat)

            threshold = (torch.max(r_y_fake_prob[:, -1]) +
                         torch.min(r_y_fake_prob[:, -1])) / 2
            indicator = torch.sign(r_y_fake_prob[:, -1] - threshold)
            mask = torch.where(indicator < 0, torch.zeros_like(indicator),
                               indicator)
            gen_ev = torch.mean(r_y_fake_prob[:, -1].log() * mask)

            loss = fm_loss + gen_ev + pt_term

            logs = {"g_loss": loss}

            return {"loss": loss, "log": logs, "progress_bar": logs}

    def on_epoch_end(self):
        if self.seq2seq is not None:
            z = (torch.rand(10, self.hparams.latent_dim).to(self.device) -
                 0.5) * 2

            x_gen = self.generator(z)
            outputs = self.seq2seq.decode(x_gen)
            self.logger.experiment.add_text(
                "generated", "  \n".join(
                    self.character_encoder.batch_decode(
                        outputs, [10 for _ in outputs])), self.global_step)

    def configure_optimizers(self):
        return (
            [
                optim.AdamW(self.generator.parameters(), lr=self.hparams.lr),
                optim.AdamW(self.discriminator.parameters(), lr=1e-2),
            ],
            [],
        )
