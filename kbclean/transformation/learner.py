from argparse import Namespace

from pytorch_lightning import Trainer
from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertForSequenceClassification, BertModel

from kbclean.transformation.noisy_channel import NoisyChannel, noisy_channel
from kbclean.utils.data.dataset.transform import TransformDataset
from kbclean.utils.data.helpers import diff_dfs, str2regex
from kbclean.utils.data.split import split_train_test_dls
from kbclean.utils.logger import MetricsTensorBoardLogger


def collate_fn(batch):
    return list(zip(batch))


class BertClassification(BertModel):
    def __init__(self, hparams, tokenizer):
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(hparams, tokenizer)

        self.model = BertForSequenceClassification(
            BertConfig(
                vocab_size=hparams.vocab_size,
                hidden_size=hparams.hidden_size,
                num_hidden_layers=hparams.num_hidden_layers,
                intermediate_size=hparams.intermediate_size,
                hidden_dropout_prob=hparams.hidden_dropout_prob,
                num_attention_heads=hparams.num_attention_heads,
            )
        )


class TransLearner:
    def __init__(self, hparams, tokenizer):
        self.hparams = hparams
        self.tokenizer = tokenizer

    def generate_transform_data(self, training_df_pairs):
        string_pairs = []
        for df1, df2 in training_df_pairs:
            diff_df = diff_dfs(df1, df2)
            string_pairs.extend(diff_df[["from", "to"]].values.tolist())
        return string_pairs

    def generate_training_data(self, string_pairs):
        noisy_channel = NoisyChannel(string_pairs)
        training_data = []
        for (error_str, cleaned_str), _ in noisy_channel.transform_dist.items():
            training_data.append([error_str, f"{error_str}=>{cleaned_str}"])
        return training_data

    def fit(self, training_df_pairs):
        string_pairs = self.generate_transform_data(training_df_pairs)
        training_data = self.generate_training_data(string_pairs)

        dataset = TransformDataset(training_data)

        train_dataloader, val_dataloader = split_train_test_dls(
            dataset, collate_fn, batch_size=self.hparams.batch_size
        )

        self.model = BertClassification(self.hparams, self.tokenizer)
        self.trainer = Trainer(
            gpus=[0, 1, 2, 3],
            distributed_backend="ddp",
            logger=MetricsTensorBoardLogger("tt_logs", "deep"),
            max_epochs=2,
        )
        self.trainer.fit(
            self.model,
            train_dataloader=train_dataloader,
            val_dataloaders=[val_dataloader],
        )

    def save(self):
        self.trainer.save_checkpoint(f"{self.hparams.save_path}/model.ckpt")
