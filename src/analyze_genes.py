import torch
import numpy as np
import scanpy as sc
import os
from transformers import BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "../results/fine_tuned_sc_model")
    data_path = os.path.join(script_dir, "../data/pbmc3k_processed.h5ad")
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} not found.")
        return
        
    print("Loading fine-tuned model...")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    
    # Extract the gene embeddings layer from BERT
    # shape: (vocab_size, hidden_dim)
    gene_embeddings = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
    print(f"Extracted gene embeddings of shape: {gene_embeddings.shape}")
    
    print("Loading processed AnnData to map vocabulary to gene symbols...")
    adata = sc.read_h5ad(data_path)
    gene_symbols = adata.var_names.tolist()
    
    # Note: Token ID 0 is <PAD>. 
    # Token IDs 1 to n_vars correspond to gene_symbols[0] to gene_symbols[-1].
    # Let's map Gene Symbol -> Token ID
    gene2id = {gene: idx + 1 for idx, gene in enumerate(gene_symbols)}
    id2gene = {idx + 1: gene for idx, gene in enumerate(gene_symbols)}
    
    # Let's define some known marker genes to query
    marker_genes = ['MS4A1', 'LYZ', 'CD3E', 'NKG7', 'IL7R', 'CD8A']
    
    print("\n--- Gene Embedding Nearest Neighbors ---")
    # Compute cosine similarity for all genes
    # Exclude <PAD> token (index 0) from the comparison matrix
    valid_embeddings = gene_embeddings[1:len(gene_symbols)+1]
    
    # Normalize valid embeddings for fast cosine similarity calculation
    norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
    valid_embeddings_norm = valid_embeddings / (norms + 1e-8)
    
    similarity_matrix = np.dot(valid_embeddings_norm, valid_embeddings_norm.T)
    
    for marker in marker_genes:
        if marker not in gene2id:
            print(f"{marker} not found in vocabulary.")
            continue
            
        marker_id = gene2id[marker]
        marker_idx = marker_id - 1 # index in the valid_embeddings matrix
        
        # Get similarities for this marker
        sims = similarity_matrix[marker_idx]
        
        # Get top 10 most similar genes (including itself at index 0)
        top_indices = np.argsort(sims)[::-1][:10]
        
        print(f"\nTop 10 nearest neighbors to {marker}:")
        for i, idx in enumerate(top_indices):
            gene_name = id2gene[idx + 1]
            score = sims[idx]
            print(f"  {i+1}. {gene_name} (Similarity: {score:.3f})")
            
    print("\n--- Generating Gene UMAP ---")
    print("Running UMAP on all gene embeddings (this may take a few seconds)...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine')
    gene_umap = reducer.fit_transform(valid_embeddings)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(gene_umap[:, 0], gene_umap[:, 1], s=1, alpha=0.3, c='gray')
    
    # Highlight the marker genes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    for idx, marker in enumerate(marker_genes):
        if marker in gene2id:
            m_idx = gene2id[marker] - 1
            plt.scatter(gene_umap[m_idx, 0], gene_umap[m_idx, 1], s=100, c=colors[idx], label=marker, edgecolors='black')
            plt.annotate(marker, (gene_umap[m_idx, 0]+0.1, gene_umap[m_idx, 1]+0.1), fontsize=12, weight='bold')
            
    plt.title("UMAP of Foundation Model Gene Embeddings")
    plt.legend()
    plot_path = os.path.join(script_dir, "../results/gene_embeddings_umap.png")
    plt.savefig(plot_path)
    print(f"Saved gene embedding UMAP to {plot_path}")

if __name__ == "__main__":
    main()
