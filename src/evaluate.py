import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(predictions, true_labels, embeddings, class_names):
    """
    Evaluates the fine-tuned model's predictions and visualizes the cell embeddings using UMAP.
    """
    print("Evaluating Model Performance...")
    
    # 1. Classification Metrics
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))
    
    # 2. Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix: Cell Type Prediction")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("../results/confusion_matrix.png")
    plt.close()
    
    # 3. UMAP Visualization
    print("Generating UMAP visualization of cell embeddings...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=true_labels, cmap='Spectral', s=5)
    plt.title("UMAP projection of Fine-Tuned Cell Embeddings")
    plt.colorbar(scatter, ticks=range(len(class_names)), label='Cell Type')
    plt.savefig("../results/umap_embeddings.png")
    plt.close()
    
    print("Evaluation complete. Plots saved to ../results/")

if __name__ == "__main__":
    print("Run this script after generating predictions and embeddings from fine_tune.py")
