import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from data_utils_baseline import load_data

# Load preprocessed data
train_data, test_data = load_data()

# Separate features and labels
X_train, y_train = train_data['text'], train_data['label']
X_test, y_test = test_data['text'], test_data['label']

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')