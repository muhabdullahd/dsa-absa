import sys
sys.path.append('..')  # Add parent directory to path

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics_utils import ModelComparison
from models.baseline.train_baseline import train_baseline
from models.svm.train_svm_baseline import train_svm_baseline
from models.absa.absa_1 import train_absa

def compare_models():
    # Train all models and collect their metrics
    print("Training Baseline Model...")
    baseline_metrics = train_baseline()
    
    print("\nTraining SVM Model...")
    svm_metrics = train_svm_baseline()
    
    print("Training ABSA Model...")
    absa_metrics = train_absa()
    if absa_metrics is None:
        raise RuntimeError("ABSA model training failed to return metrics.")

    
    # Create comparison
    comparison = ModelComparison({
        'baseline': baseline_metrics,
        'svm': svm_metrics,
        'absa': absa_metrics
    })
    
    # Generate comparison plots and save results
    print("\nGenerating comparison plots...")
    comparison.plot_all_metrics_comparison()
    comparison.save_comparison_results()
    
    # Print final comparison table
    print("\nFinal Comparison Table:")
    print(comparison.generate_comparison_table())

if __name__ == "__main__":
    compare_models() 