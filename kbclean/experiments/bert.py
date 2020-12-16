from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data.dataset import TensorDataset, random_split
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


data_path = Path("data/test/raha/beers")

dirty_df = pd.read_csv(data_path / "dirty.csv", keep_default_na=False, dtype=str)
clean_df = pd.read_csv(data_path / "clean.csv", keep_default_na=False, dtype=str)

for col in dirty_df.columns:
    dirty_values = (
        dirty_df[col]
        .iloc[:]
        .values.tolist()
    )

    clean_values = (
        clean_df[col]
        .iloc[:]
        .values.tolist()
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    input_ids = []
    attention_masks = []

    encoded_list = []

    for sent in dirty_values:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
            truncation=True,
        )

        encoded_list.append(encoded_dict)
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict["input_ids"])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict["attention_mask"])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor([int(x == y) for x, y in zip(dirty_values, clean_values)])

    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.05 * len(dataset))
    val_size = len(dataset) - train_size

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    train_dataset, test_dataset = random_split(dataset, [train_size, val_size])


    def dummy_data_collator(features):
        batch = {}
        batch["input_ids"] = torch.stack([f[0] for f in features])
        batch["attention_mask"] = torch.stack([f[1] for f in features])
        batch["labels"] = torch.stack([f[2] for f in features])

        return batch


    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        num_train_epochs=3,  # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
    )


    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
        data_collator=dummy_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

    df = pd.DataFrame(columns=["sentence", "prediction", "labels"])

    for idx, sent in enumerate(dirty_values):
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
            truncation=True,
        )

        outputs = model(**{x: y.cuda() for x, y in encoded_dict.items()}, return_dict=True)
        df = df.append(pd.Series({"sentence": sent, "prediction": torch.argmax(outputs.logits, dim=1), "labels": labels[idx]}), ignore_index=True)
    
    df.to_csv(f"output/lstm/bert/{col}_result.csv", index=None)