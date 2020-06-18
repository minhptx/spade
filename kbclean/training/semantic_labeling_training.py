import argparse

import rapidjson as json
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from ml.multi_classfication import Sherlock
from utils.data.dataset import SlicedTensorDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--trainX", metavar="N", type=str, default="data/train/sherlock/tensors.pt"
    )
    parser.add_argument(
        "--trainY", metavar="N", type=str, default="data/train/sherlock/label.pt"
    )
    parser.add_argument(
        "--labels", metavar="N", type=str, default="data/other/t2d/properties.json"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name_to_tensor = torch.load(args.trainX)

    labels = json.load(open(args.labels, "r"))

    label_tensor = torch.load(args.trainY).long()

    num_class = len(labels)
    input_sizes = [x.shape[1] for x in name_to_tensor.values()]

    label_tensor = label_tensor.long()
    tensors = [x.float() for x in name_to_tensor.values()] + [label_tensor]

    dataset = SlicedTensorDataset(*tensors)
    train_size = int(len(dataset) * 0.7)

    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=2048, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=2048, num_workers=0)

    sherlock = Sherlock(
        input_sizes=input_sizes, num_class=num_class, hidden_size=500, dropout_rate=0.3
    )

    trainer = Trainer(
        gpus=[0, 1, 3],
        amp_level="O1",
        benchmark=False,
        default_save_path="../../checkpoints/sherlock/",
        distributed_backend="dp",
    )
    trainer.fit(
        sherlock, train_dataloader=train_dataloader, val_dataloaders=[val_dataloader]
    )
