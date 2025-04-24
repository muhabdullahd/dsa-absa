import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import sys
import time
sys.path.append('../..')  # Add parent directory to path
from utils.data_utils_baseline import load_data
from utils.metrics_utils import ModelMetrics

class RandomClassifier:
    def __init__(self, class_weights=None):
        self.class_weights = class_weights
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.class_weights is None:
            # Use uniform distribution if no weights provided
            self.class_weights = np.ones(len(self.classes_)) / len(self.classes_)
        return self
        
    def predict(self, X):
        return np.random.choice(self.classes_, size=len(X), p=self.class_weights)
        
    def predict_proba(self, X):
        # Return probability matrix based on class weights
        return np.tile(self.class_weights, (len(X), 1))

def train_random_baseline():
    # Initialize metrics
    metrics = ModelMetrics("random")
    
    # Load preprocessed data
    train_data, test_data = load_data()

    # Separate features and labels
    X_train, y_train = train_data['text'], train_data['label']
    X_test, y_test = test_data['text'], test_data['label']

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Compute class weights based on training data distribution
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = class_weights / np.sum(class_weights)  # Normalize to probabilities

    # Initialize model with class distribution weights
    model = RandomClassifier(class_weights=class_weights)
    
    # Perform multiple random predictions to get a distribution
    n_iterations = 5
    for i in range(n_iterations):
        # Train is instantaneous for random model
        model.fit(X_train, y_train)
        
        # Make random predictions on validation set
        val_pred = model.predict(X_train)
        val_prob = model.predict_proba(X_train)
        
        # Compute metrics for this iteration
        metrics.compute_metrics(
            y_true=y_train,
            y_pred=val_pred,
            y_prob=val_prob,
            is_test=False
        )
    
    # Final evaluation on test set
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions on test set
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    prediction_time = time.time() - start_time

    # Compute final metrics on test set
    metrics.compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        training_time=training_time,
        prediction_time=prediction_time,
        is_test=True
    )
    
    # Generate plots
    metrics.plot_confusion_matrix()
    metrics.plot_learning_curves()
    metrics.plot_roc_curves(y_test, y_prob)
    
    # Save metrics to CSV
    metrics.save_metrics_to_csv()
    
    print("\nRandom Baseline Results:")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Prediction Time: {prediction_time:.2f} seconds")
    print("\nMetrics Summary:")
    print(metrics.get_metrics_summary())
    
    return metrics

if __name__ == "__main__":
    train_random_baseline() 