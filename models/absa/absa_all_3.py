# models/absa/absa-all-3.py

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from scipy.special import softmax
import sys
import time
sys.path.append('../..')  # Add parent directory to path
from utils.data_utils_absa import preprocess_for_absa
from utils.metrics_utils import ModelMetrics

def train_absa_all():
    # Initialize metrics
    metrics = ModelMetrics("absa-all-3")

    # Load data
    train_df = pd.read_csv("Dataset/rest16_quad_train_cleaned.tsv", delimiter="\t")
    dev_df   = pd.read_csv("Dataset/rest16_quad_dev_cleaned.tsv", delimiter="\t")
    test_df  = pd.read_csv("Dataset/rest16_quad_test_cleaned.tsv", delimiter="\t")

    # Preprocess
    train_df, dev_df, test_df = preprocess_for_absa(train_df, dev_df, test_df)

    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    def tokenize(example):
        # Using a longer max_length to accommodate the additional sentiment information
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=156)

    train_dataset = train_dataset.map(tokenize, batched=True)
    dev_dataset = dev_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # Include original ABSA columns in the dataset format
    columns_to_format = ["input_ids", "attention_mask", "labels"]
    if "original_aspect" in train_dataset.column_names:
        columns_to_format.extend(["original_aspect", "original_sentiment"])

    for dataset in [train_dataset, dev_dataset, test_dataset]:
        dataset.set_format("torch", columns=columns_to_format)

    num_labels = len(train_df["labels"].unique())
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

    # Define compute_metrics for Trainer to track metrics during training
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = softmax(logits, axis=-1)
        return metrics.compute_metrics(y_true=labels, y_pred=preds, y_prob=probs, is_test=False)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",  # Evaluate after each epoch
        save_strategy="no",
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    # Train and track time
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # Evaluate on test set and track time
    start_time = time.time()
    predictions = trainer.predict(test_dataset)
    prediction_time = time.time() - start_time

    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)

    # Add final test metrics with timing information
    metrics.compute_metrics(
        y_true=labels,
        y_pred=preds,
        y_prob=probs,
        training_time=training_time,
        prediction_time=prediction_time,
        is_test=True
    )

    # Generate plots and save
    metrics.plot_confusion_matrix()
    metrics.plot_learning_curves()
    metrics.plot_roc_curves(labels, probs)
    metrics.save_metrics_to_csv()

    print("\nABSA-All-3 Model Results:")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Prediction Time: {prediction_time:.2f} seconds")
    print("\nMetrics Summary:")
    print(metrics.get_metrics_summary())

    return metrics

if __name__ == "__main__":
    train_absa_all()