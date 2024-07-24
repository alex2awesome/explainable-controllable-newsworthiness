import os
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to data
file_path = '/project/jonmay_231/spangher/Projects/newsworthiness-extended-dataset/data/full_newsworthiness_training_data.jsonl'

data = [json.loads(line) for line in lines]

# Convert to data frame
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Split train and test
train_df = df[(df['date'] >= datetime(2014, 1, 1)) & (df['date'] < datetime(2021, 1, 1))]
test_df = df[df['date'] >= datetime(2021, 1, 1)]

# Balance daataset
def balance_dataset(df):
    df_positive = df[df['label'] == True]
    df_negative = df[df['label'] == False]
    
    min_size = min(len(df_positive), len(df_negative))
    
    df_positive_balanced = df_positive.sample(min_size, random_state=42)
    df_negative_balanced = df_negative.sample(min_size, random_state=42)
    
    return pd.concat([df_positive_balanced, df_negative_balanced]).sample(frac=1, random_state=42)

# Balance the train dataset
balanced_train_df = balance_dataset(train_df)

# Converting to Huggingface Dataset format
train_dataset = Dataset.from_pandas(balanced_train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenizer
def create_prompt_and_tokenize(entry):
    transcribed_texts = [f"<{segment['speaker']}> spoke for {segment['end'] - segment['start']:.1f} minutes and said: \"{segment['text']}\"" for segment in entry['transcribed_text']]
    transcribed_text = " ".join(transcribed_texts)
    prompt = f"(1) Policy description: \"{entry['text']}\"\n" \
             f"Presented in {entry['index']} prior meetings\n\n" \
             f"(2) Introduced by speakers in the meeting for {entry['time'] - entry['end_time']:.1f} minutes:\n" \
             f"\"{transcribed_text}\"\n\n" \
             f"(3) {len(entry['transcribed_text'])} members of the public spoke for {sum([segment['end'] - segment['start'] for segment in entry['transcribed_text']]):.1f} minutes.\n" \
             f"Is this newsworthy? Answer \"yes\" or \"no\"."
    tokenized = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
    return tokenized

# Convert to number
def convert_labels(example):
    example['labels'] = 1 if example['label'] else 0
    return example

# MRR
def mean_reciprocal_rank(predictions, labels):
    ranks = []
    for i in range(len(predictions)):
        if labels[i] == 1:
            rank = 1 / (i + 1)
            ranks.append(rank)
    return sum(ranks) / len(ranks) if ranks else 0

# Evaluate metrics
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    roc_auc = roc_auc_score(p.label_ids, preds)

    confidences = p.predictions.max(axis=1)
    sorted_indices = confidences.argsort()[::-1]
    
    # Recal at 10
    top_10_indices = sorted_indices[:10]
    true_positives_in_top_10 = sum(p.label_ids[top_10_indices])
    recall_at_10 = true_positives_in_top_10 / sum(p.label_ids)
    
    # MRR
    sorted_labels = p.label_ids[sorted_indices]
    mrr = mean_reciprocal_rank(sorted_labels, sorted_labels)
    
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'recall_at_10': recall_at_10,
        'mrr': mrr
    }

# Models
model_names_and_targets = {
    "bert-base-uncased": ["classifier"],
    "roberta-base": ["classifier.dense", "classifier.out_proj"],
    "distilbert-base-uncased": ["classifier"],
    "albert-base-v2": ["classifier"],
    "t5-large": ["lm_head"], 
    "roberta-large": ["classifier.dense", "classifier.out_proj"],
}

# Train + evaluate models
for model_name, target_modules in model_names_and_targets.items():
    try:
        print(f"Processing model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create train and test datasets
        train_dataset = train_dataset.map(lambda x: create_prompt_and_tokenize(x), batched=False)
        test_dataset = test_dataset.map(lambda x: create_prompt_and_tokenize(x), batched=False)

        train_dataset = train_dataset.map(convert_labels)
        test_dataset = test_dataset.map(convert_labels)
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(device)

        # PEFT
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules
        )

        # Configure PEFT with LoRA
        model = get_peft_model(model, peft_config)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=5,
            weight_decay=0.01,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        print(f"Training model: {model_name}")
        # Train!!
        trainer.train()

        print(f"Evaluating model: {model_name}")
        results = trainer.evaluate()

        preds = trainer.predict(test_dataset)
        test_metrics = compute_metrics(preds)

        # Print results
        print(f"Results for {model_name}:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value}")

    except Exception as e:
        print(f"Error processing {model_name}: {e}")

