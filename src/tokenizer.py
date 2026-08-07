import numpy as np
import anndata as ad
from datasets import Dataset

from typing import Tuple, Dict

def tokenize_anndata_for_geneformer(adata: ad.AnnData) -> Tuple[Dataset, Dict[str, int], Dict[str, int]]:
    """
    Converts an AnnData object into a Hugging Face Dataset suitable for Geneformer.
    Returns: HF Dataset, vocab mapping, and label mapping.
    """
    # Create vocab mapping for genes (1-indexed, 0 for padding)
    vocab = {f"ENSG000{i:08d}": i+1 for i in range(adata.n_vars)} 
    vocab["<PAD>"] = 0
    mock_ensembl_ids = np.array([f"ENSG000{i:08d}" for i in range(adata.n_vars)])
    
    # Create label mapping for cell types
    if 'cell_type' not in adata.obs:
        adata.obs['cell_type'] = "Unknown" # fallback
    unique_labels = adata.obs['cell_type'].unique()
    label2id = {str(label): idx for idx, label in enumerate(unique_labels)}
    
    tokenized_cells = []
    counts = adata.layers["counts"]
    
    print("Tokenizing cells...")
    for i in range(adata.n_obs):
        cell_counts = counts[i].toarray().flatten() if hasattr(counts, "toarray") else counts[i]
        nonzero_idx = np.nonzero(cell_counts)[0]
        sorted_nonzero_idx = nonzero_idx[np.argsort(-cell_counts[nonzero_idx])]
        
        # Map to integer IDs using vocab
        ranked_genes_str = mock_ensembl_ids[sorted_nonzero_idx].tolist()
        input_ids = [vocab[gene] for gene in ranked_genes_str]
        
        # Get label
        label_str = str(adata.obs['cell_type'].iloc[i])
        label_id = label2id[label_str]
        
        tokenized_cells.append({
            "input_ids": input_ids,
            "length": len(input_ids),
            "label": label_id
        })
        
    hf_dataset = Dataset.from_list(tokenized_cells)
    print(f"Created Hugging Face Dataset with {len(hf_dataset)} examples.")
    return hf_dataset, vocab, label2id

if __name__ == "__main__":
    from data_loader import download_and_preprocess_pbmc
    adata = download_and_preprocess_pbmc()
    hf_ds = tokenize_anndata_for_geneformer(adata)
    print("Sample tokenized cell:", hf_ds[0])
