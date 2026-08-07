import os
import matplotlib
matplotlib.use('Agg') # ensure headless plotting works without issues

from data_loader import download_and_preprocess_pbmc
from tokenizer import tokenize_anndata_for_geneformer
from fine_tune import setup_and_train, extract_predictions_and_embeddings
from evaluate import evaluate_model

def main():
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../data", exist_ok=True)
    
    # 1. Load Data
    print("--- Step 1: Loading and Preprocessing Data ---")
    adata = download_and_preprocess_pbmc(data_dir="../data")
    
    # 2. Tokenize Data
    print("\n--- Step 2: Tokenizing Data ---")
    hf_dataset, vocab, label2id = tokenize_anndata_for_geneformer(adata)
    
    # 3. Train-Test Split
    print("\n--- Step 3: Splitting Dataset ---")
    hf_dataset = hf_dataset.shuffle(seed=42)
    split = hf_dataset.train_test_split(test_size=0.2)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    
    # 4. Fine-Tune Model
    print("\n--- Step 4: Fine-Tuning Foundation Model ---")
    num_classes = len(label2id)
    vocab_size = max(vocab.values()) + 1
    model, trainer = setup_and_train(
        train_dataset=train_ds, 
        val_dataset=val_ds, 
        num_classes=num_classes,
        vocab_size=vocab_size,
        pad_token_id=vocab["<PAD>"]
    )
    
    # 5. Extract Embeddings & Predict
    print("\n--- Step 5: Extracting Embeddings & Predicting ---")
    preds, embeddings = extract_predictions_and_embeddings(trainer, val_ds)
    
    # 6. Evaluate
    print("\n--- Step 6: Evaluating Results ---")
    true_labels = val_ds["label"]
    
    # Reverse mapping for class names to ensure correct order
    id2label = {v: k for k, v in label2id.items()}
    class_names = [id2label[i] for i in range(num_classes)]
    
    evaluate_model(
        predictions=preds, 
        true_labels=true_labels, 
        embeddings=embeddings, 
        class_names=class_names
    )
    
    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
