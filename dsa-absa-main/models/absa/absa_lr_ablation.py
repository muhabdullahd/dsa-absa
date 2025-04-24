import sys
sys.path.append('../..')  # Let Python find the utils folder

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset
from utils.data_utils_absa import preprocess_for_absa

# Load and process the data
train_df = pd.read_csv("../../Dataset/rest16_quad_train_cleaned.tsv", delimiter="\t")
dev_df = pd.read_csv("../../Dataset/rest16_quad_dev_cleaned.tsv", delimiter="\t")
test_df = pd.read_csv("../../Dataset/rest16_quad_test_cleaned.tsv", delimiter="\t")
train_df, dev_df, test_df = preprocess_for_absa(train_df, dev_df, test_df)

# Set the number of labels based on the data
num_labels = len(train_df["labels"].unique())

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Convert to Dataset and tokenize
train_dataset = Dataset.from_pandas(train_df)
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

# Define schedulers
scheduler_step = StepLR(optimizer, step_size=1, gamma=0.1)  # Decay LR by 0.1 every epoch
scheduler_rop = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=3)  # Cosine annealing over 3 epochs

# Choose a scheduler (uncomment one for each run)
scheduler = scheduler_step
# scheduler = scheduler_rop
# scheduler = scheduler_cosine

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Step the scheduler
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(total_loss)
    else:
        scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}, LR: {current_lr}")