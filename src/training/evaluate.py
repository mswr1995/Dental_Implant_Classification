import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.data_loader import DentalImplantDataLoader
from tensorflow.keras.models import load_model

class ModelEvaluator:
    """
    Evaluation utilities for dental implant classification models.
    """
    
    def __init__(
        self,
        model_path,
        test_data_dir,
        results_dir,
        img_size=(512, 512),
        batch_size=16
    ):
        """
        Initialize the evaluator with specified parameters.
        
        Args:
            model_path (str): Path to saved model
            test_data_dir (str): Path to test data directory
            results_dir (str): Path to save evaluation results
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size for evaluation
        """
        # Update to handle both path formats (with and without timestamp)
        self.model_path = model_path
        
        # Extract model info from path if possible
        self.model_info = self._extract_model_info_from_path(model_path)
        
        self.test_data_dir = test_data_dir
        self.results_dir = results_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
        
        # Load model
        self.model = load_model(model_path)
        
        # Initialize data loader
        self.data_loader = DentalImplantDataLoader(img_size=img_size, batch_size=batch_size)
        self.test_generator = self.data_loader.create_test_generator(test_data_dir)
        
        # Get class names
        self.class_names = list(self.test_generator.class_indices.keys())
    
    def _extract_model_info_from_path(self, path):
        """Extract model information from the filepath for reporting"""
        info = {}
        try:
            # Extract model name, stage, and whether it's best or final
            filename = os.path.basename(path)
            parts = filename.replace('.keras', '').replace('.h5', '').split('_')
            
            if len(parts) >= 3:
                info['model_name'] = parts[0]
                if parts[1] == 'stage' and parts[2].isdigit():
                    info['stage'] = int(parts[2])
                if 'best' in parts:
                    info['model_type'] = 'best'
                elif 'final' in parts:
                    info['model_type'] = 'final'
        except:
            pass
        return info
    
    def evaluate(self):
        """
        Evaluate model performance on test data.
        
        Returns:
            dict: Evaluation metrics
        """
        # Get test data
        x_test = []
        y_test = []
        
        for i in range(len(self.test_generator)):
            batch_x, batch_y = self.test_generator[i]
            x_test.append(batch_x)
            y_test.append(batch_y)
            
            if i + 1 >= len(self.test_generator):
                break
                
        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test)
        
        # Get predictions
        y_pred_proba = self.model.predict(x_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        metrics = self.model.evaluate(x_test, y_test)
        metrics_dict = dict(zip(self.model.metrics_names, metrics))
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # Save metrics
        metrics_file = os.path.join(self.results_dir, 'metrics', 'evaluation_metrics.csv')
        
        # Convert report to DataFrame for easier saving
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(metrics_file)
        
        # Calculate top-3 accuracy
        top3_acc = self.calculate_top_k_accuracy(y_test, y_pred_proba, k=3)
        print(f"Top-3 Accuracy: {top3_acc:.4f}")
        
        # Return metrics
        return {
            **metrics_dict,
            'top3_accuracy': top3_acc,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self):
        """
        Generate and save confusion matrix visualization.
        """
        # Get predictions
        y_true = []
        y_pred = []
        
        for i in range(len(self.test_generator)):
            batch_x, batch_y = self.test_generator[i]
            preds = self.model.predict(batch_x)
            y_true.extend(np.argmax(batch_y, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
            
            if i + 1 >= len(self.test_generator):
                break
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save plot
        plt.tight_layout()
        cm_file = os.path.join(self.results_dir, 'plots', 'confusion_matrix.png')
        plt.savefig(cm_file, dpi=300)
        plt.close()
        
        return cm
    
    def plot_roc_curves(self):
        """
        Generate and save ROC curves for each class.
        """
        # Get test data
        x_test = []
        y_test = []
        
        for i in range(len(self.test_generator)):
            batch_x, batch_y = self.test_generator[i]
            x_test.append(batch_x)
            y_test.append(batch_y)
            
            if i + 1 >= len(self.test_generator):
                break
                
        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test)
        
        # Get predictions
        y_pred_proba = self.model.predict(x_test)
        
        # Plot ROC curves
        plt.figure(figsize=(12, 10))
        
        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(self.class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(
                fpr[i], tpr[i],
                lw=2,
                label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})'
            )
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        
        # Save plot
        plt.tight_layout()
        roc_file = os.path.join(self.results_dir, 'plots', 'roc_curves.png')
        plt.savefig(roc_file, dpi=300)
        plt.close()
        
        return roc_auc
    
    def calculate_top_k_accuracy(self, y_true, y_pred_proba, k=3):
        """
        Calculate top-k accuracy.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred_proba: Predicted probabilities
            k (int): Number of top predictions to consider
            
        Returns:
            float: Top-k accuracy
        """
        # Convert one-hot encoded labels to class indices
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
            
        # Get top-k predictions
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        
        # Check if true label is in top-k predictions
        matches = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
        
        # Calculate accuracy
        return np.mean(matches)
    
    def visualize_predictions(self, num_samples=10):
        """
        Visualize model predictions on random test samples.
        
        Args:
            num_samples (int): Number of samples to visualize
        """
        # Get random samples from test generator
        x_batch, y_batch = next(self.test_generator)
        indices = np.random.choice(len(x_batch), min(num_samples, len(x_batch)), replace=False)
        
        # Make predictions
        predictions = self.model.predict(x_batch[indices])
        
        # Create figure
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
                
            # Get image and labels
            img = x_batch[idx]
            true_label = np.argmax(y_batch[idx])
            pred_label = np.argmax(predictions[i])
            
            # Display image
            axes[i].imshow(img)
            color = 'green' if true_label == pred_label else 'red'
            title = f"True: {self.class_names[true_label]}\nPred: {self.class_names[pred_label]}"
            axes[i].set_title(title, color=color)
            axes[i].axis('off')
        
        # Save figure
        plt.tight_layout()
        vis_file = os.path.join(self.results_dir, 'plots', 'sample_predictions.png')
        plt.savefig(vis_file, dpi=300)
        plt.close()
    
    def run_full_evaluation(self):
        """
        Run complete evaluation pipeline and generate all metrics and visualizations.
        
        Returns:
            dict: Evaluation results
        """
        print("Starting model evaluation...")
        
        # Calculate metrics
        metrics = self.evaluate()
        print(f"Test accuracy: {metrics.get('accuracy', 0):.4f}")
        
        # Generate confusion matrix
        print("Generating confusion matrix...")
        cm = self.plot_confusion_matrix()
        
        # Generate ROC curves
        print("Generating ROC curves...")
        roc_auc = self.plot_roc_curves()
        
        # Visualize predictions
        print("Visualizing sample predictions...")
        self.visualize_predictions(num_samples=10)
        
        print("Evaluation complete. Results saved to:", self.results_dir)
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'roc_auc': roc_auc
        }


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator(
        model_path='../../results/models/efficientnetb3_stage_3_final.keras',  # Changed from .h5 to .keras
        test_data_dir='../../data/data_processed/test',
        results_dir='../../results/evaluation'
    )
    
    results = evaluator.run_full_evaluation()
    print(results)