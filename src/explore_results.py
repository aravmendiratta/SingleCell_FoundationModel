import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Path to the downloaded results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../results/evaluation_data.npz")
    plot_path = os.path.join(script_dir, "../results/custom_latent_plot.png")
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        return
        
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    embeddings = data['embeddings']
    true_labels = data['true_labels']
    preds = data['preds']
    class_names = data['class_names']
    
    print(f"\n--- Data Summary ---")
    print(f"Embeddings shape: {embeddings.shape} (Cells x Hidden Dimensions)")
    print(f"Predictions shape: {preds.shape}")
    print(f"Classes: {', '.join(class_names)}")
    
    # Example 1: Calculate accuracy
    accuracy = np.mean(preds == true_labels) * 100
    print(f"\n--- Metrics ---")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Example 2: Inspect a specific cell type
    target_class_id = 0
    class_name = class_names[target_class_id]
    class_mask = (preds == target_class_id)
    print(f"\n--- Custom Analysis ---")
    print(f"Number of cells predicted as {class_name}: {np.sum(class_mask)}")
    
    # Example 3: Doing your own plotting!
    # Here we plot the first two dimensions of the high-dimensional embeddings
    # (Since these are not UMAP reduced yet, this is just a slice of the raw latent space)
    print("\nGenerating a custom plot of the first 2 latent dimensions...")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=true_labels, cmap='tab20', s=15, alpha=0.8)
    plt.title("Latent Space (Dimensions 0 and 1)")
    
    # Add a legend with the class names
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Cell Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(plot_path)
    print(f"Saved custom plot to {plot_path}")

if __name__ == "__main__":
    main()
