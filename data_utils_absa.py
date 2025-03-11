# data_utils.py
import pandas as pd
import string
import os
import spacy
from textblob import TextBlob
from nltk.tokenize import word_tokenize

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define file paths (ensure Dataset folder exists)
dataset_folder = "Dataset"
train_path = os.path.join(dataset_folder, "rest16_quad_train.tsv")
dev_path = os.path.join(dataset_folder, "rest16_quad_dev.tsv")
test_path = os.path.join(dataset_folder, "rest16_quad_test.tsv")

# Define column names based on expected structure
column_names = ['text', 'absa1', 'absa2', 'absa3']

# Load datasets
df_train = pd.read_csv(train_path, sep="\t", names=column_names, on_bad_lines='skip')
df_dev = pd.read_csv(dev_path, sep="\t", names=column_names, on_bad_lines='skip')
df_test = pd.read_csv(test_path, sep="\t", names=column_names, on_bad_lines='skip')

# Function to clean text: Lowercasing, removing punctuation
def clean_text(text):
    if isinstance(text, str):  # Ensure text is a string
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text

# Apply TextBlob correction (fixes minor typos)
df_train['corrected_text'] = df_train['text'].apply(lambda x: str(TextBlob(x).correct()))
df_dev['corrected_text'] = df_dev['text'].apply(lambda x: str(TextBlob(x).correct()))
df_test['corrected_text'] = df_test['text'].apply(lambda x: str(TextBlob(x).correct()))

# Apply cleaning function (lowercasing, removing punctuation)
df_train['cleaned_text'] = df_train['corrected_text'].apply(clean_text)
df_dev['cleaned_text'] = df_dev['corrected_text'].apply(clean_text)
df_test['cleaned_text'] = df_test['corrected_text'].apply(clean_text)

# Tokenization (spaCy)
df_train['tokens'] = df_train['cleaned_text'].apply(lambda x: [token.text.lower() for token in nlp(x) if token.text not in string.punctuation])
df_dev['tokens'] = df_dev['cleaned_text'].apply(lambda x: [token.text.lower() for token in nlp(x) if token.text not in string.punctuation])
df_test['tokens'] = df_test['cleaned_text'].apply(lambda x: [token.text.lower() for token in nlp(x) if token.text not in string.punctuation])

# Split train data into train (80%) and validation (20%)
train_split = int(0.8 * len(df_train))
df_train_final = df_train.iloc[:train_split].reset_index(drop=True)
df_val = df_train.iloc[train_split:].reset_index(drop=True)

# Save processed files
df_train_final[['tokens', 'absa1', 'absa2', 'absa3']].to_csv(os.path.join(dataset_folder, "rest16_quad_train_cleaned.tsv"), sep="\t", index=False)
df_val[['tokens', 'absa1', 'absa2', 'absa3']].to_csv(os.path.join(dataset_folder, "rest16_quad_val_cleaned.tsv"), sep="\t", index=False)
df_dev[['tokens', 'absa1', 'absa2', 'absa3']].to_csv(os.path.join(dataset_folder, "rest16_quad_dev_cleaned.tsv"), sep="\t", index=False)
df_test[['tokens', 'absa1', 'absa2', 'absa3']].to_csv(os.path.join(dataset_folder, "rest16_quad_test_cleaned.tsv"), sep="\t", index=False)

# Print confirmation
print("Preprocessing complete. Cleaned train, validation, dev, and test sets saved.")