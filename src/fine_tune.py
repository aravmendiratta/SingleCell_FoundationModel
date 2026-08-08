import torch
import numpy as np
from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from typing import List, Dict, Any

class SingleCellCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        max_len = max(f["length"] for f in features)
        
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)
        
        for i, feature in enumerate(features):
            length = feature["length"]
            input_ids[i, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[i, :length] = 1
            labels[i] = feature["label"]
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def setup_and_train(train_dataset: Dataset, val_dataset: Dataset, num_classes: int, vocab_size: int, pad_token_id: int, config_dict: dict):
    """
    Sets up a lightweight Transformer model and fine-tunes it on the cell-type classification task.
    """
    model_cfg = config_dict.get('model', {})
    train_cfg = config_dict.get('training', {})
    
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=model_cfg.get('hidden_size', 256),
        num_hidden_layers=model_cfg.get('num_hidden_layers', 4),
        num_attention_heads=model_cfg.get('num_attention_heads', 4),
        intermediate_size=model_cfg.get('intermediate_size', 512),
        max_position_embeddings=model_cfg.get('max_position_embeddings', 8192),
        num_labels=num_classes,
        output_hidden_states=True
    )
    
    print("Initializing Foundation Model for sequence classification...")
    model = BertForSequenceClassification(config)
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=train_cfg.get('learning_rate', 2e-4),
        per_device_train_batch_size=train_cfg.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=train_cfg.get('per_device_eval_batch_size', 8),
        num_train_epochs=train_cfg.get('num_train_epochs', 3),
        weight_decay=train_cfg.get('weight_decay', 0.01),
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False # Important for custom dataset format
    )
    
    collator = SingleCellCollator(pad_token_id=pad_token_id)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator
    )
    
    import os
    model_dir = "./results/fine_tuned_sc_model"
    if os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"Found existing fine-tuned model at {model_dir}, skipping training...")
        model = BertForSequenceClassification.from_pretrained(model_dir)
        trainer.model = model
    else:
        print("Starting fine-tuning...")
        trainer.train()
        trainer.save_model(model_dir)
        
    print("Model fine-tuning setup complete.")
    return model, trainer

def extract_predictions_and_embeddings(trainer: Trainer, dataset: Dataset):
    """
    Extracts predictions and mean-pooled hidden states for the given dataset.
    """
    print("Extracting predictions and embeddings...")
    
    dataloader = trainer.get_test_dataloader(dataset)
    device = trainer.args.device
    
    all_preds = []
    all_embeddings = []
    trainer.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = trainer.model(**inputs)
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            
            # Mean pooling over the sequence dimension, ignoring padding
            hidden_states = outputs.hidden_states[-1] # (batch, seq_len, hidden_size)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            all_embeddings.append(mean_embeddings.cpu().numpy())
            
    embeddings = np.vstack(all_embeddings)
    return np.array(all_preds), embeddings
