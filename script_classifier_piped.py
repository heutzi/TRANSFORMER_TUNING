import argparse
import ast
from datasets import Dataset, load_dataset
import os
from sklearn.metrics import mean_absolute_error
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, Trainer, TrainingArguments

import numpy as np

from modules.classifier_large import DenseClassifierConfig, DenseClassifier


####################
#  CLASSIFICATION  #
####################

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    mae = mean_absolute_error(labels, preds)  # fonctionne pas ici
    return {"mean absolute error": mae}


def build_trainer(
        dataset: Dataset,
        model_name: str,
        num_labels: int
        ) -> Trainer:
    # input_embed_size = dataset["train"]["input_ids"].shape[1]
    input_embed_size = 768
    classifier_config = DenseClassifierConfig(model_name,
                                              input_embed_size,
                                              num_labels,
                                              classifier_dropout=0.2)
    classifier = DenseClassifier(classifier_config)

    batch_size = 128
    logging_steps = len(dataset["train"]) // batch_size
    model_name = "Linear-emotion"  # unused

    # Note : by default the trainer uses an Adam optimizer
    training_args = TrainingArguments(output_dir=model_name,
                                      num_train_epochs=40,
                                      learning_rate=5e-4,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      weight_decay=0.01,
                                      eval_strategy="epoch",
                                      disable_tqdm=False,
                                      logging_steps=logging_steps,
                                      push_to_hub=False,
                                      log_level="error",
                                      save_strategy='no')

    trainer = Trainer(model=classifier,
                      args=training_args,
                      compute_metrics=compute_metrics,
                      train_dataset=emotion_ds["train"],
                      eval_dataset=emotion_ds["test"])
    return classifier, trainer


##########
#  MAIN  #
##########


if __name__ == "__main__":
    desc = "Train model with provided data"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        'model_name',
        metavar='Model',
        type=str,
        help='Model name')

    DATA_DIR = "corpora/"

    args = parser.parse_args()
    model_name = args.model_name

    labels_pred_list = []
    labels_true_lists = []

    def convert_labels(item):
        item["id"] = int(item["id"])
        for dim in dims:
            label = ast.literal_eval(item[dim])
            item[dim] = torch.tensor(label)
        return item

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    dims = ["familiarité", "agréabilité", "utilité", "légitimité"]

    train_dir = f"{DATA_DIR}/trains"
    N_train_examples = [name for name in os.listdir(train_dir)
                        if os.path.isfile(os.path.join(train_dir, name))]

    for dimension in dims:

        os.makedirs(f"results/predictions/{dimension}/", exist_ok=True)

        for i in range(0, N_train_examples):

            #  DATASETS LOADING
            emotion_ds = load_dataset(
                "csv",
                data_files={
                    "train": f"{DATA_DIR}/trains/yoom_avg_train_{i}.csv",
                    "test": f"{DATA_DIR}/tests/yoom_avg_test_{i}.csv"
                },
                delimiter='\t'
                )

            emotion_ds = emotion_ds.map(convert_labels)

            emotion_ds = emotion_ds.map(tokenize,
                                        batched=True,
                                        batch_size=None)

            emotion_ds = emotion_ds.rename_column(dimension, "labels")

            emotion_ds.set_format(
                "torch",
                columns=["labels", "input_ids", "attention_mask"]
                )

            #  TRAINER BUILDING
            num_labels = 3
            classifier, trainer = build_trainer(emotion_ds,
                                                model_name,
                                                num_labels)

            trainer.train()

            output = trainer.predict(emotion_ds["test"])

            labels_pred = np.array(softmax(torch.tensor(output.predictions)))

            output_ds = emotion_ds["test"].select_columns(
                ['id', 'text', 'labels']
                )
            output_ds = output_ds.rename_column("labels", dimension)
            output_ds = output_ds.add_column(
                name="prediction",
                column=list(labels_pred)
                )

            output_ds.to_csv(
                f"results/predictions/{dimension}/predictions_{i}.csv",
                sep='\t'
                )

            labels_pred_list.append(labels_pred)

            labels_true = np.array(emotion_ds["test"]["labels"])
            labels_true_lists.append(labels_true)
