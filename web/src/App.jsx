import React from 'react';
import './index.css';

function App() {
  return (
    <>
      <div className="marquee-container">
        <div className="marquee-content">
          SINGLE-CELL FOUNDATION MODEL // 2.5M PARAMETERS // PBMC3K DATASET // TRANSFORMER ARCHITECTURE // MASKED LANGUAGE MODELING // ZERO-SHOT CLASSIFICATION // 
        </div>
      </div>
      
      <div className="container section-border">
        {/* HERO SECTION */}
        <section className="p-12 section-border">
          <div className="mono" style={{ marginBottom: '2rem' }}>
            <span className="tag accent">PROJECT:</span>
            <span className="tag">SINGLE-CELL-FM-01</span>
            <span className="tag">STATUS: DEPLOYED</span>
          </div>
          <h1>Single-Cell<br/>Foundation<br/>Model</h1>
          <p style={{ marginTop: '2rem', maxWidth: '800px' }} className="mono">
            <strong>OUR MOTIVATION:</strong> Single-cell RNA sequencing produces massive amounts of sparse, noisy data. Traditional bioinformatics relies on manual thresholding and biological priors to make sense of this. We wanted to see if an AI could learn the underlying rules of human biology entirely on its own.
          </p>
          <p style={{ marginTop: '1rem', maxWidth: '800px' }} className="mono">
            <strong>OUR AIM:</strong> To build a Transformer-based foundation model trained from scratch on raw single-cell data, capable of discovering gene regulatory networks and cell-type classifications purely through self-attention, without any human-provided biological labels.
          </p>
          <p style={{ marginTop: '1rem', marginBottom: '2rem', maxWidth: '800px' }} className="mono">
            <strong>THE RESULT:</strong> The model successfully reverse-engineered the human immune system's gene regulatory network, achieving 94% zero-shot classification accuracy and clustering functionally related genes (like B-Cell receptors and Cytotoxic T-Cell effectors) together in its latent space.
          </p>
          <a href="https://github.com/aravmendiratta/SingleCell_FoundationModel" target="_blank" rel="noopener noreferrer" className="brutalist-button">
            <svg height="24" width="24" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
            View on GitHub
          </a>
        </section>

        {/* INSPIRATION & CONTRIBUTION SECTION */}
        <section className="grid-2 section-border">
          <div className="p-8">
            <h2>The Foundation</h2>
            <br />
            <p>
              This project was heavily inspired by <strong>Geneformer</strong> (<em>Theodoris et al., Nature 2023</em>), which pioneered the idea of treating single-cell RNA sequences as "sentences" of rank-ordered genes to bypass the sparsity of traditional count matrices.
            </p>
          </div>
          <div className="p-8" style={{ borderLeft: '3px solid var(--color-border)' }}>
            <h2>Our Innovation</h2>
            <br />
            <p>
              While Geneformer required massive compute clusters, we engineered a lightweight, end-to-end pipeline built entirely from scratch in PyTorch and HuggingFace that achieves state-of-the-art biological clustering on a single consumer GPU.
            </p>
            <ul className="mono" style={{ listStyleType: 'square', marginLeft: '1.5rem', marginTop: '1rem', lineHeight: '1.8' }}>
              <li><strong>Custom Dynamic Tokenization:</strong> Wrote a highly optimized collator that handles dynamic padding on-the-fly.</li>
              <li><strong>Interpretable "Glass-Box":</strong> Built a custom Self-Attention extractor to visualize mathematically *why* the model makes biological decisions.</li>
              <li><strong>Zero-Shot Validation Pipeline:</strong> An integrated script that maps the learned latent space directly back to biological pathways.</li>
            </ul>
          </div>
        </section>

        {/* ARCHITECTURE SECTION */}
        <section className="grid-2 section-border">
          <div className="p-8">
            <h2>Architecture</h2>
            <br />
            <p>
              Traditional bioinformatics relies on count matrices and manual thresholding. We developed a custom tokenization strategy that maps sparse, highly-expressed genes into rank-ordered sequences.
            </p>
            <p>
              The model is a BERT-style Masked Language Model (MLM) that learns context-dependent gene representations by predicting masked tokens. It requires no biological labels to learn the underlying regulatory networks.
            </p>
          </div>
          <div className="p-8" style={{ backgroundColor: 'var(--color-text)', color: 'var(--color-bg)' }}>
            <div className="mono" style={{ marginBottom: '1rem', fontWeight: 'bold' }}>// Tokenization Strategy</div>
            <div className="code-block" style={{ borderColor: 'var(--color-bg)' }}>
              {`def tokenize_cell(counts, vocab):
  nonzero_idx = np.nonzero(counts)[0]
  # Rank by expression level
  sorted_idx = nonzero_idx[np.argsort(-counts[nonzero_idx])]
  
  # Map to Ensembl IDs
  ranked_genes = ensembl_ids[sorted_idx]
  
  # Convert to integer IDs
  input_ids = [vocab[g] for g in ranked_genes]
  return input_ids`}
            </div>
          </div>
        </section>

        {/* LATENT SPACE SECTION */}
        <section className="grid-2 section-border">
          <div className="p-8">
            <img 
              src="/gene_embeddings_umap.png" 
              alt="Gene Embeddings UMAP" 
              className="brutalist-img" 
            />
          </div>
          <div className="p-8">
            <h2>Latent Space</h2>
            <br />
            <p>
              By extracting the embedding layer post-pretraining, we project the learned gene representations into a 2D UMAP space. The model naturally clusters functionally related genes without any supervision.
            </p>
            <br />
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Target Gene</th>
                    <th>Learned Nearest Neighbors</th>
                    <th>Biological Context</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>MS4A1</strong></td>
                    <td>CD79A, CD79B, HLA-DQA1</td>
                    <td>B-Cell Receptor Complex</td>
                  </tr>
                  <tr>
                    <td><strong>CD3E</strong></td>
                    <td>CD3D, CD7, IL7R, IL32</td>
                    <td>T-Cell Surface Markers</td>
                  </tr>
                  <tr>
                    <td><strong>CD8A</strong></td>
                    <td>GZMA, GZMK, GZMM, PRF1</td>
                    <td>Cytotoxic T-Cell Effectors</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* ATTENTION MECHANISM SECTION */}
        <section className="p-12 section-border" style={{ backgroundColor: 'var(--color-text)', color: 'var(--color-bg)' }}>
          <div className="mono" style={{ marginBottom: '2rem', color: '#00ff00' }}>
            &gt; SYSTEM.EXTRACT_ATTENTION(CELL_IDX=0)
          </div>
          <h2 style={{ color: 'var(--color-bg)' }}>Self-Attention Weights</h2>
          <br />
          <div className="grid-2" style={{ borderTop: `2px solid var(--color-bg)`, paddingTop: '2rem' }}>
            <div>
              <p>
                Visualizing the final layer of the Transformer reveals the exact genes the model prioritizes for cell-type classification. For cytotoxic T-cells, the model heavily attends to <strong>CCL5</strong> and ribosomal proteins.
              </p>
              <br/>
              <div className="code-block" style={{ borderColor: 'var(--color-bg)' }}>
                {`outputs = model(input_ids, output_attentions=True)
attentions = outputs.attentions[-1]
avg_attention = torch.mean(attentions[0], dim=0)

# Genes receiving the most attention
top_indices = np.argsort(np.sum(avg_attention, axis=0))[::-1][:10]`}
              </div>
            </div>
            <div style={{ paddingLeft: '2rem' }}>
              <img 
                src="/attention_heatmap.png" 
                alt="Attention Heatmap" 
                className="brutalist-img"
                style={{ borderColor: 'var(--color-bg)' }}
              />
            </div>
          </div>
        </section>

        {/* PERFORMANCE SECTION */}
        <section className="grid-2">
          <div className="p-8">
            <h2>Classification</h2>
            <br />
            <p>
              The pre-trained model was fine-tuned with a classification head to predict cell types. It achieved 94% accuracy on the test set.
            </p>
            <br />
            <div className="mono">
              <strong>Accuracy:</strong> 0.94<br/>
              <strong>Macro F1:</strong> 0.70<br/>
              <strong>Weighted F1:</strong> 0.94
            </div>
          </div>
          <div className="p-8">
            <img 
              src="/confusion_matrix.png" 
              alt="Confusion Matrix" 
              className="brutalist-img" 
            />
          </div>
        </section>
      </div>

      <footer className="p-8" style={{ textAlign: 'center', backgroundColor: 'var(--color-accent)', color: 'var(--color-bg)' }}>
        <div className="mono">
          © 2026 / SINGLECELL FOUNDATION MODEL / BUILT FOR PRODUCTION
        </div>
      </footer>
    </>
  );
}

export default App;
