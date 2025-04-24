import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Tuple, Optional
import pandas as pd
from datetime import datetime
import os

class ModelMetrics:
    def __init__(self, model_name: str, save_dir: str = "results"):
        """
        Initialize metrics tracker for a model.
        
        Args:
            model_name (str): Name of the model (e.g., 'baseline', 'svm', 'absa-1', 'absa-all-3')
            save_dir (str): Directory to save metrics and plots
        """
        self.model_name = model_name
        self.save_dir = save_dir
        self.metrics_history = {
            'accuracy': [],
            'precision_macro': [],
            'precision_weighted': [],
            'recall_macro': [],
            'recall_weighted': [],
            'f1_macro': [],
            'f1_weighted': [],
            'confusion_matrix': None,
            'training_time': [],
            'prediction_time': []
        }
        
        # Create base results directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create method-specific directories
        self.method_dirs = {
            'compute_metrics': os.path.join('images', 'compute_metrics', model_name),
            'confusion_matrix': os.path.join('images', 'confusion_matrix', model_name),
            'learning_curves': os.path.join('images', 'learning_curves', model_name),
            'roc_curves': os.path.join('images', 'roc_curves', model_name),
            'metrics_comparison': os.path.join('images', 'metrics_comparison', model_name),
            'all_metrics': os.path.join('images', 'all_metrics', model_name)
        }
        
        # Create all directories
        for dir_path in self.method_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
    def compute_metrics(self, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray, 
                       y_prob: Optional[np.ndarray] = None,
                       training_time: Optional[float] = None,
                       prediction_time: Optional[float] = None) -> Dict[str, float]:
        """
        Compute all metrics for the model.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray, optional): Predicted probabilities
            training_time (float, optional): Training time in seconds
            prediction_time (float, optional): Prediction time in seconds
            
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Update history
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                self.metrics_history[metric].append(value)
            else:
                self.metrics_history[metric] = value
                
        # Add timing metrics if provided
        if training_time is not None:
            self.metrics_history['training_time'].append(training_time)
        if prediction_time is not None:
            self.metrics_history['prediction_time'].append(prediction_time)
            
        # Save metrics to CSV in the compute_metrics directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_df = pd.DataFrame({
            k: v for k, v in self.metrics_history.items() 
            if isinstance(v, list) and v
        })
        metrics_df.to_csv(os.path.join(self.method_dirs['compute_metrics'], f'metrics_{timestamp}.csv'), index=False)
            
        return metrics
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get the latest metrics values."""
        return {k: v[-1] if isinstance(v, list) and v else v 
                for k, v in self.metrics_history.items()}
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        if self.metrics_history['confusion_matrix'] is None:
            return
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.metrics_history['confusion_matrix'], 
                    annot=True, 
                    fmt='d',
                    cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.method_dirs['confusion_matrix'], f'confusion_matrix_{timestamp}.png')
            
        plt.savefig(save_path)
        plt.close()
        
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot learning curves for all metrics."""
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            if self.metrics_history[metric]:
                axes[idx].plot(self.metrics_history[metric], 
                             label=metric.replace('_', ' ').title(),
                             marker='o')
                axes[idx].set_title(f'{metric.replace("_", " ").title()} over time')
                axes[idx].set_xlabel('Evaluation Step')
                axes[idx].set_ylabel(metric.replace('_', ' ').title())
                axes[idx].grid(True)
                axes[idx].legend()
        
        plt.tight_layout()
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.method_dirs['learning_curves'], f'learning_curves_{timestamp}.png')
            
        plt.savefig(save_path)
        plt.close()
        
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray, 
                        save_path: Optional[str] = None):
        """Plot ROC curves for each class."""
        n_classes = y_prob.shape[1]
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {self.model_name}')
        plt.legend(loc="lower right")
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.method_dirs['roc_curves'], f'roc_curves_{timestamp}.png')
            
        plt.savefig(save_path)
        plt.close()
        
    def save_metrics_to_csv(self, save_path: Optional[str] = None):
        """Save metrics history to CSV file."""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.method_dirs['compute_metrics'], f'metrics_{timestamp}.csv')
            
        metrics_df = pd.DataFrame({
            k: v for k, v in self.metrics_history.items() 
            if isinstance(v, list) and v
        })
        metrics_df.to_csv(save_path, index=False)
        return save_path

class ModelComparison:
    def __init__(self, models: Dict[str, ModelMetrics], save_dir: str = "results"):
        """
        Initialize model comparison.
        
        Args:
            models (Dict[str, ModelMetrics]): Dictionary of model metrics
            save_dir (str): Directory to save comparison plots
        """
        self.models = models
        self.save_dir = save_dir
        
        # Create base results directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create method-specific directories
        self.method_dirs = {
            'metrics_comparison': os.path.join('images', 'metrics_comparison', 'all_models'),
            'all_metrics': os.path.join('images', 'all_metrics', 'all_models')
        }
        
        # Create all directories
        for dir_path in self.method_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
    def plot_metrics_comparison(self, metric: str, save_path: Optional[str] = None):
        """Plot comparison of a specific metric across models."""
        plt.figure(figsize=(10, 6))
        
        for model_name, metrics in self.models.items():
            if isinstance(metrics.metrics_history[metric], list):
                plt.plot(metrics.metrics_history[metric], 
                        label=model_name, 
                        marker='o')
        
        plt.title(f'{metric.replace("_", " ").title()} Comparison Across Models')
        plt.xlabel('Evaluation Step')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.method_dirs['metrics_comparison'], f'{metric}_comparison_{timestamp}.png')
            
        plt.savefig(save_path)
        plt.close()
    
    def plot_all_metrics_comparison(self, save_path: Optional[str] = None):
        """Plot comparison of all metrics across models."""
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            for model_name, model_metrics in self.models.items():
                if model_metrics.metrics_history is not None and isinstance(model_metrics.metrics_history[metric], list):
                    axes[idx].plot(model_metrics.metrics_history[metric], 
                                 label=model_name,
                                 marker='o')
            
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[idx].set_xlabel('Evaluation Step')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].grid(True)
            axes[idx].legend()
        
        plt.tight_layout()
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.method_dirs['all_metrics'], f'all_metrics_comparison_{timestamp}.png')
            
        plt.savefig(save_path)
        plt.close()
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table of all metrics."""
        comparison_data = []
        for model_name, metrics in self.models.items():
            model_metrics = metrics.get_metrics_summary()
            model_metrics['model'] = model_name
            comparison_data.append(model_metrics)
        
        return pd.DataFrame(comparison_data).set_index('model')
    
    def save_comparison_results(self, save_dir: Optional[str] = None):
        """Save all comparison results."""
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(self.method_dirs['all_metrics'], f'comparison_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save comparison table
        comparison_table = self.generate_comparison_table()
        comparison_table.to_csv(os.path.join(save_dir, 'metrics_comparison.csv'))
        
        # Save individual plots
        self.plot_all_metrics_comparison(
            os.path.join(save_dir, 'all_metrics_comparison.png')
        )
        
        # Save confusion matrices
        for model_name, metrics in self.models.items():
            metrics.plot_confusion_matrix(
                os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
            )
            
        return save_dir 