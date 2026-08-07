import torch
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments
from datasets import Dataset
from typing import List, Dict, Any

class SingleCellMLMCollator:
    def __init__(self, pad_token_id: int, mask_token_id: int, vocab_size: int, mlm_probability=0.15):
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mlm_prob = mlm_probability

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        max_len = max(f["length"] for f in features)
        
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        for i, feature in enumerate(features):
            length = feature["length"]
            ids = torch.tensor(feature["input_ids"], dtype=torch.long)
            
            # 15% of tokens are masked
            prob_matrix = torch.full((length,), self.mlm_prob)
            masked_indices = torch.bernoulli(prob_matrix).bool()
            
            # The labels are the original IDs (only for masked tokens, rest is -100)
            labels[i, :length][masked_indices] = ids[masked_indices]
            
            # Set input to [MASK] token
            ids[masked_indices] = self.mask_token_id
            
            input_ids[i, :length] = ids
            attention_mask[i, :length] = 1
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def setup_and_pretrain(dataset: Dataset, vocab_size: int, pad_token_id: int, mask_token_id: int, config_dict: dict):
    """
    Sets up a Transformer model for Masked Language Modeling (MLM).
    """
    model_cfg = config_dict.get('model', {})
    train_cfg = config_dict.get('training', {})
    
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=model_cfg.get('hidden_size', 256),
        num_hidden_layers=model_cfg.get('num_hidden_layers', 4),
        num_attention_heads=model_cfg.get('num_attention_heads', 4),
        intermediate_size=model_cfg.get('intermediate_size', 512),
        max_position_embeddings=model_cfg.get('max_position_embeddings', 8192)
    )
    
    print("Initializing Foundation Model for Masked Language Modeling...")
    model = BertForMaskedLM(config)
    
    training_args = TrainingArguments(
        output_dir="./results_pretrain",
        learning_rate=train_cfg.get('learning_rate', 2e-4),
        per_device_train_batch_size=train_cfg.get('per_device_train_batch_size', 8),
        num_train_epochs=train_cfg.get('num_train_epochs', 3),
        weight_decay=train_cfg.get('weight_decay', 0.01),
        logging_dir='./logs_pretrain',
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False
    )
    
    collator = SingleCellMLMCollator(
        pad_token_id=pad_token_id, 
        mask_token_id=mask_token_id,
        vocab_size=vocab_size
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )
    
    print("Starting pre-training...")
    trainer.train()
    trainer.save_model("./results_pretrain/pretrained_sc_model")
    return model, trainer
