import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
import spacy
from textblob import TextBlob


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define column names
column_names = ['text_column', 'other_column1', 'other_column2', 'other_column3']

# Read TSV file with specified column names and handle bad lines
df_dev = pd.read_csv("Dataset/rest16_quad_dev.tsv", sep="\t", on_bad_lines='skip', names=column_names)

# Step 1: Text Correction using TextBlob (Handles minor typos)
df_dev['corrected_text'] = df_dev['text_column'].apply(lambda x: str(TextBlob(x).correct()))

# Tokenization (Better handling of contractions)
df_dev['tokens'] = df_dev['text_column'].apply(lambda x: [token.text.lower() for token in nlp(x) if token.text not in string.punctuation])

print(df_dev)   # Print the DataFrame
