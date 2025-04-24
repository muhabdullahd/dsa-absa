import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import sys
import time
sys.path.append('../..')  # Add parent directory to path
from utils.data_utils_baseline import load_data
from utils.metrics_utils import ModelMetrics

def train_svm_baseline():
    # Initialize metrics
    metrics = ModelMetrics("svm")
    
    # Load preprocessed data
    train_data, test_data = load_data()

    # Separate features and labels
    X_train, y_train = train_data['text'], train_data['label']
    X_test, y_test = test_data['text'], test_data['label']

    # Convert to numpy arrays if they aren't already
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Initialize model
    model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
    
    # Use K-fold cross validation to get multiple evaluation points
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train with cross-validation to get learning curves
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        # Train on this fold
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate on validation fold
        val_pred = model.predict(X_val_fold)
        val_prob = model.predict_proba(X_val_fold)
        
        # Compute metrics for this fold
        metrics.compute_metrics(
            y_true=y_val_fold,
            y_pred=val_pred,
            y_prob=val_prob,
            is_test=False
        )
    
    # Final training on full training set
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
    
    print("\nSVM Model Results:")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Prediction Time: {prediction_time:.2f} seconds")
    print("\nMetrics Summary:")
    print(metrics.get_metrics_summary())
    
    return metrics

if __name__ == "__main__":
    train_svm_baseline()