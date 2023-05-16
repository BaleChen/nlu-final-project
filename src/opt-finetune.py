from transformers import AutoModelForCausalLM, AutoTokenizer,\
TrainingArguments, Trainer, DataCollatorForLanguageModeling

from datasets import load_dataset, Dataset

from functools import partial
import torch
import json
import evaluate

import argparse

import os
import random
from random import shuffle

block_size = 128

def merge(json_files):
    re = {}
    for file in json_files:
        for key, value in file.items():
            re[key] = re.get(key, []) + value
    return re

def preprocess(examples, tokenizer):
    return tokenizer(examples["sampled"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, default=10)
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
    
    data_dir = "results/datasets/"
    
    args.base_model = args.base_model.replace("/", "_")
    
    raw_list = []
    for folder in os.listdir(data_dir):
        if args.base_model in folder:
            for subfolder in os.listdir(data_dir+folder):
                json_path = data_dir+folder+"/"+subfolder+"/raw_data.json"
                with open(json_path, "rb") as f:
                    r = json.load(f)
                    print("Loaded json file from", json_path)
                    print("Sampled data length:", len(r["sampled"]))
                    raw_list.append(r)
                    
    raw = merge(raw_list)
    
    print(len(raw["sampled"]))
    print(raw.keys())
    
    raw_data = Dataset.from_dict(raw)
    
    preprocess_function = partial(preprocess, tokenizer = tokenizer)
    raw_data = raw_data.map(preprocess_function,
                batched=True,
                num_proc=4,
                remove_columns=["sampled", "original"])
    processed_data = raw_data.map(group_texts, batched=True, num_proc=4)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f"ft-opt/{args.base_model}",
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,
        # weight_decay=0.01,
        # push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_data,
        data_collator=data_collator,
    )
    
    trainer.train()