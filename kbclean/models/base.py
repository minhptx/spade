import torch
from pytorch_lightning import LightningModule


class BaseModule(LightningModule):
    def on_train_start(self):
        self.logger.log_hyperparams_metrics(
            self.hparams,
            {
                "val_loss": 100,
                "val_acc": 0,
                "train_loss": 100,
                "train_acc": 0
            },
        )

    def training_epoch_end(self, outputs: list):
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["train_acc"] for x in outputs]).mean()
        logs = {
            "train_loss": avg_loss,
            "train_acc": avg_acc,
        }
        return {"avg_train_loss": avg_loss, "log": logs, "progress_bar": logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss, "val_acc": avg_acc}
        return {"avg_val_loss": avg_loss, "log": logs, "progress_bar": logs}
