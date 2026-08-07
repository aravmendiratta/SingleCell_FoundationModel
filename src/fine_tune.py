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

def setup_and_train(train_dataset: Dataset, val_dataset: Dataset, num_classes: int, vocab_size: int, pad_token_id: int = 0):
    """
    Sets up a lightweight Transformer model and fine-tunes it on the cell-type classification task.
    """
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=8192, # some cells express many genes
        num_labels=num_classes,
        output_hidden_states=True
    )
    
    print("Initializing Foundation Model for sequence classification...")
    model = BertForSequenceClassification(config)
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
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
    
    print("Starting fine-tuning...")
    trainer.train()
    trainer.save_model("./results/fine_tuned_sc_model")
    print("Model fine-tuning setup complete.")
    return model, trainer

def extract_predictions_and_embeddings(trainer: Trainer, dataset: Dataset):
    """
    Extracts predictions and mean-pooled hidden states for the given dataset.
    """
    print("Extracting predictions and embeddings...")
    predictions_output = trainer.predict(dataset)
    logits = predictions_output.predictions[0] if isinstance(predictions_output.predictions, tuple) else predictions_output.predictions
    preds = np.argmax(logits, axis=1)
    
    dataloader = trainer.get_test_dataloader(dataset)
    device = trainer.args.device
    
    all_embeddings = []
    trainer.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = trainer.model(**inputs)
            
            # Mean pooling over the sequence dimension, ignoring padding
            hidden_states = outputs.hidden_states[-1] # (batch, seq_len, hidden_size)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            all_embeddings.append(mean_embeddings.cpu().numpy())
            
    embeddings = np.vstack(all_embeddings)
    return preds, embeddings
