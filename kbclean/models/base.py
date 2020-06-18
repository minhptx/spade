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

        return super().on_train_start()
