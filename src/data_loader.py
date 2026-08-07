import scanpy as sc
import anndata as ad
import os

def download_and_preprocess_pbmc(data_dir: str = "../data") -> ad.AnnData:
    """
    Downloads the PBMC 3k dataset, performs basic filtering and normalization,
    and saves it to the specified directory.
    """
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, "pbmc3k_processed.h5ad")
    
    if os.path.exists(save_path):
        print(f"Loading cached dataset from {save_path}")
        return sc.read_h5ad(save_path)
        
    print("Downloading PBMC 3k dataset...")
    adata = sc.datasets.pbmc3k()
    
    # Basic Filtering
    print("Filtering cells and genes...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Keep raw counts before normalization for some models
    adata.layers["counts"] = adata.X.copy()
    
    # Normalize to 10k counts per cell and log1p transform
    print("Normalizing and log-transforming...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Generate pseudo-labels using Leiden clustering
    print("Running PCA and Leiden clustering for pseudo-labels...")
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata, resolution=0.5, key_added='leiden')
    
    # Ensure leiden labels are stored as a categorical for downstream use
    adata.obs['cell_type'] = adata.obs['leiden']
    
    # Save the processed dataset
    adata.write(save_path)
    print(f"Saved processed dataset to {save_path}")
    
    return adata

if __name__ == "__main__":
    adata = download_and_preprocess_pbmc()
    print(adata)
