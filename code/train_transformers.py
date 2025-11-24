import os
import pandas as pd
import torch
import logging
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import csv
import time
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
DATA_DIR = '../data/processed/'
OUTPUT_DIR = '../outputs/models/'
REPORTS_DIR = '../outputs/reports/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MODELS = {
    'distilbert': 'distilbert-base-uncased',
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'albert': 'albert-base-v2',
    'electra': 'google/electra-base-discriminator',
    'xlnet': 'xlnet-base-cased'
}

DATASETS = {
    'sentiment140': {'path': 'sentiment140_train.csv', 'num_labels': 2},
    'reddit': {'path': 'reddit_train.csv', 'num_labels': 3},
    'amazon_combined': {'path': 'amazon_combined_train.csv', 'num_labels': 3},
    'imdb': {'path': 'imdb_train.csv', 'num_labels': 2}
}

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': round(acc, 5),
        'f1': round(f1_w, 5),
        'f1_macro': round(f1_m, 5),
        'precision': round(p_w, 5),
        'recall': round(r_w, 5)
    }

# Dataset loading
def load_dataset(dataset_name, split='train'):
    path = os.path.join(DATA_DIR, DATASETS[dataset_name]['path'].replace('train', split))
    df = pd.read_csv(path)
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.strip() != '']
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
    df = df.dropna(subset=['sentiment'])

    # Sentiment140 stays 0/1 if labels come as {0,2} from preprocessing
    if dataset_name == 'sentiment140':
        df['sentiment'] = df['sentiment'].replace({2: 1})

    # Enforce valid label ids [0..num_labels-1]
    num_labels = DATASETS[dataset_name]['num_labels']
    df = df[df['sentiment'].isin(range(num_labels))]

    dataset = Dataset.from_pandas(df[['text', 'sentiment']])
    dataset = dataset.rename_column('sentiment', 'labels')
    return dataset

# Training
def train_model(model_name, dataset_name, output_dir):
    logging.info(f"Training {model_name} on {dataset_name}")
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[model_name],
        num_labels=DATASETS[dataset_name]['num_labels']
    )

    train_dataset = load_dataset(dataset_name, 'train')
    val_dataset = load_dataset(dataset_name, 'val')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"{model_name}_{dataset_name}"),
        eval_strategy='steps',
        eval_steps=2000,
        save_strategy='no',
        logging_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=torch.cuda.is_available(),
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir='logs/',
        report_to='none',
        seed=42
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

    # Save model + tokenizer
    model_path = os.path.join(output_dir, f"{model_name}_{dataset_name}")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    # In-domain validation report (concise console log + JSON + CSV)
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    report = classification_report(predictions.label_ids, preds, output_dict=True)
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    acc = report['accuracy']

    logging.info(
        f"MODEL={model_name} DATASET={dataset_name} "
        f"labels={DATASETS[dataset_name]['num_labels']} "
        f"macroF1={macro_f1:.3f} acc={acc:.3f}"
    )

    # Save JSON report
    rep_path = os.path.join(REPORTS_DIR, f"{model_name}_{dataset_name}_val.json")
    with open(rep_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Append to CSV summary
    csv_path = '../outputs/metrics_summary.csv'
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                'timestamp', 'dataset', 'model', 'num_labels',
                'n_train', 'n_val', 'max_len', 'epochs', 'lr',
                'macro_f1', 'weighted_f1', 'accuracy'
            ])
        w.writerow([
            int(time.time()),
            dataset_name,
            model_name,
            DATASETS[dataset_name]['num_labels'],
            len(train_dataset),
            len(val_dataset),
            256, 2, 2e-5,
            macro_f1, weighted_f1, acc
        ])

    del trainer, model, tokenizer, train_dataset, val_dataset, predictions
    torch.cuda.empty_cache()
    gc.collect()

# in-domain evaluation with classification report
def evaluate_in_domain(model_name, train_dataset, test_dataset, output_dir):
    model_path = os.path.join(output_dir, f"{model_name}_{train_dataset}")
    if not os.path.exists(model_path):
        logging.warning(f"Model {model_path} not found. Skipping evaluation.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    test_data = load_dataset(test_dataset, 'test')
    test_data = test_data.map(
        lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=256),
        batched=True
    )

    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    predictions = trainer.predict(test_data)
    preds = predictions.predictions.argmax(-1)
    report = classification_report(predictions.label_ids, preds, output_dict=True)

    result_path = os.path.join(output_dir, f'{model_name}_{train_dataset}_to_{test_dataset}_report.json')
    with open(result_path, 'w') as f:
        json.dump(report, f, indent=4)

    logging.info(f"Saved classification report to {result_path}")

    del trainer, model, tokenizer, test_data, predictions
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    # Train all (unified script)
    for model_name in MODELS:
        for dataset_name in DATASETS:
            try:
                train_model(model_name, dataset_name, OUTPUT_DIR)
            except Exception as e:
                logging.error(f"Error training {model_name} on {dataset_name}: {str(e)}")

    # (Optional) Cross-dataset evaluation
    for model_name in MODELS:
        for train_dataset in DATASETS:
            for test_dataset in DATASETS:
                if train_dataset != test_dataset:
                    try:
                        evaluate_in_domain(model_name, train_dataset, test_dataset, OUTPUT_DIR)
                    except Exception as e:
                        logging.error(f"Error evaluating {model_name} from {train_dataset} to {test_dataset}: {str(e)}")
