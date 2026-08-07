import numpy as np
import anndata as ad
from datasets import Dataset

def tokenize_anndata_for_geneformer(adata: ad.AnnData) -> Dataset:
    """
    Converts an AnnData object into a Hugging Face Dataset suitable for Geneformer.
    Geneformer expects cells to be represented as a sequence of Ensembl gene IDs,
    rank-ordered by their non-zero expression values in that cell.
    """
    # For demonstration, we assume adata.var_names are gene symbols and we mock Ensembl IDs.
    # In a real scenario, you'd map symbols to real Ensembl IDs matching the model's vocab.
    mock_ensembl_ids = np.array([f"ENSG000{i:08d}" for i in range(adata.n_vars)])
    
    tokenized_cells = []
    
    # Extract the sparse count matrix
    counts = adata.layers["counts"]
    
    print("Tokenizing cells...")
    for i in range(adata.n_obs):
        # Get counts for the i-th cell
        cell_counts = counts[i].toarray().flatten() if hasattr(counts, "toarray") else counts[i]
        
        # Get indices of non-zero expressed genes
        nonzero_idx = np.nonzero(cell_counts)[0]
        
        # Sort these indices by expression value in descending order
        sorted_nonzero_idx = nonzero_idx[np.argsort(-cell_counts[nonzero_idx])]
        
        # Map to gene IDs
        ranked_genes = mock_ensembl_ids[sorted_nonzero_idx].tolist()
        
        tokenized_cells.append({
            "input_ids": ranked_genes,
            "length": len(ranked_genes)
        })
        
    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_list(tokenized_cells)
    print(f"Created Hugging Face Dataset with {len(hf_dataset)} examples.")
    return hf_dataset

if __name__ == "__main__":
    from data_loader import download_and_preprocess_pbmc
    adata = download_and_preprocess_pbmc()
    hf_ds = tokenize_anndata_for_geneformer(adata)
    print("Sample tokenized cell:", hf_ds[0])
