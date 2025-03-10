import pandas as pd
import numpy as np
import ast
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score

from MetricCallback import MetricsCallback

# 1. Load the data
train_df = pd.read_csv("Dataset/rest16_quad_train_cleaned.tsv", delimiter="\t")
dev_df   = pd.read_csv("Dataset/rest16_quad_dev_cleaned.tsv", delimiter="\t")
test_df  = pd.read_csv("Dataset/rest16_quad_test_cleaned.tsv", delimiter="\t")

print("Training Data Columns:")
print(train_df.columns)

# 2. Convert the sentiment column values to plain integers.
# For "other_column1", we assume the sentiment is the third token when splitting by whitespace.
def convert_label(x):
    if isinstance(x, str):
        tokens = x.split()
        if len(tokens) >= 3:
            try:
                # Use the third token (index 2) as the sentiment label
                return int(tokens[2])
            except Exception as e:
                raise ValueError(f"Error converting token '{tokens[2]}' to int from string: {x}") from e
        else:
            try:
                return int(x)
            except Exception as e:
                raise ValueError(f"Cannot convert string '{x}' to int.") from e
    elif isinstance(x, list):
        if len(x) >= 3:
            try:
                return int(x[2])
            except Exception as e:
                return int(x[0])
        else:
            return int(x[0])
    else:
        return int(x)

# Apply the conversion to the sentiment column
train_df["other_column1"] = train_df["other_column1"].apply(convert_label)
dev_df["other_column1"]   = dev_df["other_column1"].apply(convert_label)
test_df["other_column1"]  = test_df["other_column1"].apply(convert_label)

# 3. Rename the sentiment column to "labels"
train_df = train_df.rename(columns={"other_column1": "labels"})
dev_df   = dev_df.rename(columns={"other_column1": "labels"})
test_df  = test_df.rename(columns={"other_column1": "labels"})

# 4. Create Hugging Face Datasets from the dataframes
train_dataset = Dataset.from_pandas(train_df)
dev_dataset   = Dataset.from_pandas(dev_df)
test_dataset  = Dataset.from_pandas(test_df)

# 5. Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 6. Define the tokenization function using the text column
def tokenize_function(example):
    return tokenizer(example["text_column"], padding="max_length", truncation=True, max_length=128)

# 7. Tokenize all datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset   = dev_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

# 8. Set the format for PyTorch and select the relevant columns
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 9. Determine the number of unique labels
num_labels = len(train_df["labels"].unique())

# 10. Load DistilBERT for sequence classification with a classification head
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# 11. Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# 12. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,                     # Adjust as needed
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",            # Note: 'evaluation_strategy' is deprecated; consider using 'eval_strategy'
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
)

# 13. Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[MetricsCallback("metrics.json")]
)

# 14. Train the model
trainer.train()

# 15. Save and Evaluate on the test set
trainer.save_model("./saved_model")
results = trainer.evaluate(test_dataset)
print("Test set evaluation results:", results)