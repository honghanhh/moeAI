# -*- coding: utf-8 -*-

import sys
import json
import torch
import numpy as np
from typing import List, Any, Dict
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from peft import PeftModel, LoraConfig, TaskType
from modeling_llama import LlamaForSequenceClassification
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm


class CustomDataCollatorWithPadding:
    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        # Extract 'id' from the features
        ids = [feature['id'] for feature in features]
        
        # Remove 'id' before passing the rest to the actual data collator
        features_without_id = [{k: v for k, v in f.items() if k != 'id'} for f in features]
        
        # Apply the default collator to process the tokenized fields
        batch = self.data_collator(features_without_id)
        
        # Add 'id' back to the batch
        batch['id'] = ids
        return batch
def main():
    if len(sys.argv) < 4:
        print('Usage: python inference.py output_dir model_size input_data [input_format] [text_field]')
        print(' - output_dir: Directory where the trained model is saved.')
        print(' - model_size: Size of the model (e.g., 7b, 13b).')
        print(' - input_data: Name of dataset (JSONL format recommended).')
        print(' - input_format: (Optional) "jsonl" or "csv". Default is "jsonl".')
        print(' - text_field: (Optional) Name of the text field in the input data. Default is "text".')
        sys.exit(1)
    
    output_dir = sys.argv[1]
    model_size = sys.argv[2]
    input_data = sys.argv[3]
    # Configuration parameters
    max_length = 128
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define label mappings (ensure this matches your training setup)
    id2label = {0: "human", 1: "machine"}
    label2id = {v: k for k, v in id2label.items()}

    # Determine model_id based on model_size
    if model_size.lower() == '7b':
        base_model_id = 'NousResearch/Llama-2-7b-hf'
    elif model_size.lower() == '13b':
        base_model_id = 'NousResearch/Llama-2-13b-hf'
    else:
        print(f"Unsupported model size: {model_size}. Choose '7b' or '13b'.")
        sys.exit(1)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Load the base model
    model = LlamaForSequenceClassification.from_pretrained(
        base_model_id, 
        num_labels=len(label2id), 
        id2label=id2label, 
        label2id=label2id
    )

    # Load PEFT (LoRA) adapter
    model = PeftModel.from_pretrained(model, output_dir, device_map="auto")

    # Move model to appropriate device
    model.to(device)
    model.eval()

    dataset = load_dataset(input_data)

    text_name='text'
    columns_to_remove = [col for col in dataset['dev'].column_names if col not in ['label','id']]
    #tokenized_ds = dataset['dev'].map(preprocess_function, batched=True, remove_columns=columns_to_remove)
    tokenized_ds = dataset['dev'].map(lambda x: preprocess_function(x, tokenizer, max_length), batched=True, remove_columns=columns_to_remove)
    # Create DataLoader
    custom_data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        tokenized_ds, 
        batch_size=batch_size,
        collate_fn=custom_data_collator,
        shuffle=False
    )


    # Initialize metrics
    accuracy = evaluate.load("accuracy")

    # Perform inference
    all_predictions = []
    # If you have labels; else ignore
    all_labels = dataset['dev']['label']
    list_ids =[]
    output_file = 'output_results_latest.jsonl'
    with open(output_file, 'w') as f_out:
        with torch.no_grad():
            for batch in dataloader:
                ids = batch['id']  # Access the 'id' field
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'id'}  # Send other data to the device
                
                # Perform inference
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Convert logits to predicted labels (0 or 1)
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                
                # Write each prediction to the jsonl file
                for id_, pred in zip(ids, predictions):
                    result = {
                        'id': id_,         # The id of the test sample
                        'labels': int(pred) # The predicted label (0 or 1)
                    }
                    # Write to jsonl file
                    f_out.write(json.dumps(result) + '\n')

def preprocess_function(examples, tokenizer, max_length):
    text_name='text'
    if isinstance(text_name, str):
        d = examples[text_name]
    else:
        d = examples[text_name[0]]
        for n in text_name[1:]:
            nd = examples[n]
            assert len(d) == len(nd)
            for i, t in enumerate(nd):
                d[i] += '\n' + t
    return tokenizer(d, padding='longest', max_length=max_length, truncation=True)
if __name__ == "__main__":
    main()
