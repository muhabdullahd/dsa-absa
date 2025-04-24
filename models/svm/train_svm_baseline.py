import pandas as pd
from sklearn.svm import SVC
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

    # Train an SVM classifier with RBF kernel
    print("Training SVM classifier...")
    model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
    
    # Time the training
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions and time them
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    prediction_time = time.time() - start_time

    # Compute and save metrics
    metrics.compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        training_time=training_time,
        prediction_time=prediction_time
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