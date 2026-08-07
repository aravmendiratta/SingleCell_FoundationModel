import pytest
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from tokenizer import tokenize_anndata_for_geneformer

def test_tokenize_anndata():
    # Mock AnnData
    counts = csr_matrix(np.array([
        [0, 10, 0, 5],
        [2, 0, 8, 0]
    ]))
    adata = ad.AnnData(X=counts)
    adata.layers["counts"] = counts
    adata.obs['cell_type'] = ['T-cell', 'B-cell']
    
    hf_dataset, vocab, label2id = tokenize_anndata_for_geneformer(adata)
    
    assert len(hf_dataset) == 2
    assert "input_ids" in hf_dataset[0]
    assert "label" in hf_dataset[0]
    
    # Check that highest expression gene is first in input_ids
    # For cell 0, gene 1 (count 10) is highest, gene 3 (count 5) is second.
    expected_gene1 = f"ENSG000{1:08d}"
    expected_gene3 = f"ENSG000{3:08d}"
    
    assert hf_dataset[0]["input_ids"][0] == vocab[expected_gene1]
    assert hf_dataset[0]["input_ids"][1] == vocab[expected_gene3]
    assert len(hf_dataset[0]["input_ids"]) == 2
