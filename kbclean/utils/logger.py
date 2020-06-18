from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard.summary import hparams


class MetricsTensorBoardLogger(TensorBoardLogger):
    def __init__(self, *args, **kwargs):
        super(MetricsTensorBoardLogger, self).__init__(*args, **kwargs)

    def log_hyperparams(self, *args, **kwargs):
        pass

    def log_hyperparams_metrics(self, params: dict, metrics: dict) -> None:
        params = self._convert_params(params)
        exp, ssi, sei = hparams(params, metrics)
        writer = self.experiment._get_file_writer()
        writer.add_summary(exp)
        writer.add_summary(ssi)
        writer.add_summary(sei)
