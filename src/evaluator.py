# src/evaluator.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_performance(true_labels, pred_labels):
    """
    Generates an academic performance report for A, C, G, T classification.
    """
    classes = ['A', 'C', 'G', 'T']
    
    # 1. Classification Report (Precision, Recall, F1)
    print("\n[Academic Report] Detailed Metrics:")
    print(classification_report(true_labels, pred_labels, target_names=classes))
    
    # 2. Confusion Matrix Visualization
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Basecalling Confusion Matrix (Bi-LSTM)')
    plt.xlabel('Predicted Base')
    plt.ylabel('True Base')
    plt.show()