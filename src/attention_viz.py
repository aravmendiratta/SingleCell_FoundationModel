import os
import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertForSequenceClassification
from data_loader import download_and_preprocess_pbmc
from tokenizer import tokenize_anndata_for_geneformer

def plot_attention_heatmap(attention_matrix, tokens, title, save_path):
    """
    Plots a heatmap for a given attention matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title(title)
    plt.xlabel("Key Tokens (Attended To)")
    plt.ylabel("Query Tokens (Attending)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved attention heatmap to {save_path}")
    plt.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir_1 = os.path.join(script_dir, "../results/fine_tuned_sc_model")
    model_dir_2 = os.path.join(script_dir, "results/fine_tuned_sc_model")
    
    if os.path.exists(model_dir_1):
        model_dir = model_dir_1
    elif os.path.exists(model_dir_2):
        model_dir = model_dir_2
    else:
        print("Error: Model directory not found. Please re-run main.py first.")
        return

    print("Loading fine-tuned model with output_attentions=True...")
    # output_attentions=True forces the model to return attention weights!
    model = BertForSequenceClassification.from_pretrained(model_dir, output_attentions=True)
    model.eval()

    # Need data and vocab to map token IDs to gene names
    data_dir = os.path.join(script_dir, "../data")
    print("Loading data to get vocabulary...")
    adata = download_and_preprocess_pbmc(data_dir=data_dir)
    hf_dataset, vocab, label2id = tokenize_anndata_for_geneformer(adata)
    
    # Reverse vocabulary mapping (Token ID -> Ensembl ID)
    id2vocab = {v: k for k, v in vocab.items()}
    
    # We also need a mapping from Ensembl ID -> Gene Symbol to make the plot readable
    # adata.var_names contains the symbols, and they map 1:1 to the mock Ensembl IDs
    ensembl2symbol = {f"ENSG000{i:08d}": symbol for i, symbol in enumerate(adata.var_names)}
    ensembl2symbol["<PAD>"] = "<PAD>"
    
    # Pick a single interesting cell (e.g., cell index 0)
    cell_idx = 0
    cell_data = hf_dataset[cell_idx]
    
    print(f"Extracting attention for Cell #{cell_idx} (Label: {list(label2id.keys())[cell_data['label']]})")
    
    # Prepare input for the model
    input_ids = torch.tensor([cell_data["input_ids"]], dtype=torch.long)
    attention_mask = torch.tensor([cell_data["attention_mask"]], dtype=torch.long)
    
    # Run the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    # Extract attentions: Tuple of length num_layers, each (batch, num_heads, seq_len, seq_len)
    attentions = outputs.attentions
    
    # Let's look at the last layer, first attention head
    last_layer_attention = attentions[-1] # shape: (1, num_heads, seq_len, seq_len)
    head_0_attention = last_layer_attention[0, 0, :, :].numpy()
    
    # To make the plot readable, let's just plot the attention for the first 30 tokens
    # (otherwise a 2048x2048 heatmap is unreadable)
    num_tokens_to_plot = min(30, len(cell_data["input_ids"]))
    subset_attention = head_0_attention[:num_tokens_to_plot, :num_tokens_to_plot]
    
    # Get the human-readable gene symbols for these tokens
    subset_token_ids = cell_data["input_ids"][:num_tokens_to_plot]
    subset_symbols = []
    for token_id in subset_token_ids:
        ensembl_id = id2vocab.get(token_id, "UNKNOWN")
        symbol = ensembl2symbol.get(ensembl_id, ensembl_id)
        subset_symbols.append(symbol)
        
    # Plot it!
    results_dir = os.path.join(script_dir, "../results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "attention_heatmap.png")
    
    title = f"Self-Attention Weights (Last Layer, Head 0)\nCell type: {list(label2id.keys())[cell_data['label']]}"
    plot_attention_heatmap(subset_attention, subset_symbols, title, plot_path)
    
    # We can also compute the "Most attended to" genes globally for this cell
    # Average attention across all heads in the last layer
    avg_attention = torch.mean(last_layer_attention[0], dim=0).numpy() # (seq_len, seq_len)
    
    # Sum the attention each token *receives* from all other tokens
    received_attention = np.sum(avg_attention, axis=0)
    
    # Find the top 10 most attended tokens
    top_indices = np.argsort(received_attention)[::-1][:10]
    
    print("\nTop 10 Most 'Attended To' Genes in this cell:")
    for i, idx in enumerate(top_indices):
        token_id = cell_data["input_ids"][idx]
        ensembl_id = id2vocab.get(token_id, "UNKNOWN")
        symbol = ensembl2symbol.get(ensembl_id, ensembl_id)
        score = received_attention[idx]
        print(f"  {i+1}. {symbol} (Attention Score: {score:.3f})")

if __name__ == "__main__":
    main()
