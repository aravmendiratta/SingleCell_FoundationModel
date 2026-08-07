import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_loader import download_and_preprocess_pbmc

@pytest.mark.skip(reason="Downloads large dataset. Run manually when needed.")
def test_data_loader():
    adata = download_and_preprocess_pbmc(data_dir="../data")
    
    assert adata is not None
    assert "leiden" in adata.obs.columns
    assert "cell_type" in adata.obs.columns
    assert "counts" in adata.layers
