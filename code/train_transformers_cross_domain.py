import os
import pandas as pd
import torch
import logging
from datetime import datetime
import time
import gc
import csv
import json

# IMPORT TRANSFORMER LIBRARIES
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer
)

from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# ============================================================================
# SECTION 1: SETUP AND CONFIGURATION (Matching Your Pattern)
# ============================================================================

# Output directories
DATA_DIR = '../data/processed/'
OUTPUT_DIR = '../outputs/models/'
REPORTS_DIR = '../outputs/reports/'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs('../outputs/results/', exist_ok=True)
os.makedirs('../logs/', exist_ok=True)

# LOGGING CONFIGURATION - Dual logging: console + file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # Ensure the log file is written with UTF-8 to avoid encoding errors on Windows
        logging.FileHandler('../logs/cross_domain_eval.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# MODEL AND DATASET DEFINITIONS (Exactly matching your setup)
MODELS = {
    'distilbert': 'distilbert-base-uncased',
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'albert': 'albert-base-v2',
    'electra': 'google/electra-base-discriminator',
    'xlnet': 'xlnet-base-cased'
}

DATASETS = {
    'sentiment140': {'path': 'sentiment140_train.csv', 'num_labels': 2},  # Twitter (binary)
    'reddit': {'path': 'reddit_train.csv', 'num_labels': 3},  # Reddit (ternary)
    'amazon_combined': {'path': 'amazon_combined_train.csv', 'num_labels': 3},  # Amazon (ternary)
    'imdb': {'path': 'imdb_train.csv', 'num_labels': 2}  # IMDb (binary)
}

# ============================================================================
# SECTION 2: METRICS FUNCTION (Your Pattern)
# ============================================================================

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate all averages for comprehensive analysis
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average='macro')
    p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': round(acc, 5),
        'f1_weighted': round(f1_w, 5),
        'f1_macro': round(f1_m, 5),
        'f1_micro': round(f1_mi, 5),
        'precision_weighted': round(p_w, 5),
        'precision_macro': round(p_m, 5),
        'recall_weighted': round(r_w, 5),
        'recall_macro': round(r_m, 5)
    }

# ============================================================================
# SECTION 3: DATA LOADING (Exactly matching your pattern)
# ============================================================================

def load_dataset(dataset_name, split='test'):
    path = os.path.join(DATA_DIR, DATASETS[dataset_name]['path'].replace('train', split))
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    # Load dataset
    df = pd.read_csv(path)
    
    # Clean text: handle NaN and empty strings
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.strip() != '']
    
    # Clean sentiment labels: convert to numeric
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
    df = df.dropna(subset=['sentiment'])
    
    # Handle Sentiment140 label mapping (0/4 -> 0/1 binary)
    if dataset_name == 'sentiment140':
        df['sentiment'] = df['sentiment'].replace({4: 1})
    
    # Validate labels are within expected range
    num_labels = DATASETS[dataset_name]['num_labels']
    df = df[df['sentiment'].isin(range(num_labels))]
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_pandas(df[['text', 'sentiment']])
    dataset = dataset.rename_column('sentiment', 'labels')
    
    return dataset


# ============================================================================
# SECTION 4: CROSS-DOMAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_cross_domain(model_name, train_dataset, test_dataset, output_dir):
    eval_start = time.time()
    
    # Find model checkpoint
    model_path = os.path.join(output_dir, f"{model_name}_{train_dataset}")
    
    if not os.path.exists(model_path):
        logging.warning(f"Model checkpoint not found: {model_path}")
        logging.warning(f"  Skipping: {model_name} ({train_dataset}->{test_dataset})")
        return False
    
    try:
        # Log the current cross-domain evaluation
        logging.info(f" Cross-domain: {model_name:12s} ({train_dataset:15s}->{test_dataset:15s})")
        
        # Load trained model & tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Load test dataset (different domain)
        test_data = load_dataset(test_dataset, 'test')
        
        # Tokenization (matching your pattern)
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=256
            )
        
        test_data = test_data.map(tokenize_function, batched=True)
        
        # Ensure the dataset returns torch tensors for Trainer.predict
        try:
            test_data = test_data.with_format("torch")
        except Exception:
            # Fallback: some older dataset versions use set_format
            test_data.set_format(type="torch")
        
        # Check compatibility between model num_labels and test dataset labels
        num_model_labels = getattr(model.config, 'num_labels', None)
        try:
            labels_unique = set(test_data['labels'])
        except Exception:
            # In some Dataset versions, accessing a column returns a torch tensor
            labels_unique = set([int(x) for x in test_data['labels']])
        
        if num_model_labels is not None:
            if len(labels_unique) == 0:
                logging.warning(f"No labels found in test dataset: {test_dataset}; skipping.")
                return False
            max_label = max(labels_unique)
            if max_label >= num_model_labels:
                logging.error(
                    f"Label mismatch: model expects {num_model_labels} labels but test labels contain {max_label} (dataset={test_dataset}). Skipping evaluation."
                )
                return False
        
        # Create Trainer and run prediction
        trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)
        predictions = trainer.predict(test_data)
        
        preds = predictions.predictions.argmax(-1)
        report = classification_report(predictions.label_ids, preds, output_dict=True)
        
        eval_time = time.time() - eval_start
        
        # Extract metrics (matching your output)
        macro_f1 = report['macro avg']['f1-score']
        accuracy = report['accuracy']
        
        logging.info(f" | Macro-F1: {macro_f1:.4f} | Time: {eval_time:.2f}s")
        
        # Save JSON report (your pattern)
        report_path = os.path.join(
            REPORTS_DIR,
            f'{model_name}_{train_dataset}_to_{test_dataset}_report.json'
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Append to CSV results (YOUR EXACT PATTERN)
        csv_path = '../outputs/results/cross_domain_results.csv'
        write_header = not os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            
            if write_header:
                w.writerow([
                    'timestamp', 'model', 'source_dataset', 'target_dataset',
                    'macro_f1', 'accuracy', 'eval_time_s'
                ])
            
            w.writerow([
                int(time.time()),
                model_name,
                train_dataset,
                test_dataset,
                macro_f1,
                accuracy,
                eval_time
            ])
        
        # Cleanup
        del trainer, model, tokenizer, test_data, predictions
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        logging.error(f"Error: {model_name} ({train_dataset}->{test_dataset}): {str(e)}", exc_info=True)
        return False


# ============================================================================
# SECTION 5: MAIN EXECUTION (Your startup pattern)
# ============================================================================

if __name__ == "__main__":
    
    # STARTUP: Log experiment information (EXACTLY your pattern)
    logging.info("\n" + "="*80)
    logging.info("PHASE 2: CROSS-DOMAIN EVALUATION (72 runs)")
    logging.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"GPU Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    logging.info("="*80 + "\n")
    
    overall_start = time.time()
    success_count = 0
    total_count = 0
    
    # Iterate through all combinations: model × source domain → target domain
    # Skip when source == target (those are in-domain, already done)
    
    model_list = sorted(MODELS.keys())
    dataset_list = sorted(DATASETS.keys())
    
    for model_name in model_list:
        logging.info(f"\nMODEL: {model_name.upper()}")
        logging.info("-" * 80)
        
        for train_dataset in dataset_list:
            for test_dataset in dataset_list:
                if train_dataset != test_dataset:  # Skip in-domain
                    total_count += 1
                    
                    success = evaluate_cross_domain(
                        model_name,
                        train_dataset,
                        test_dataset,
                        OUTPUT_DIR
                    )
                    
                    if success:
                        success_count += 1
    
    # Final summary (Your pattern)
    total_time = time.time() - overall_start
    
    logging.info("\n" + "="*80)
    logging.info("PHASE 2: CROSS-DOMAIN EVALUATION - COMPLETED")
    logging.info("="*80)
    logging.info(f"Total Execution Time: {total_time/60:.2f} minutes ({total_time:.2f}s)")
    logging.info(f"Success: {success_count}/{total_count} evaluations completed")
    logging.info(f"\nOutput files:")
    logging.info(f" CSV Results:   ../outputs/results/cross_domain_results.csv")
    logging.info(f" JSON Reports:  {REPORTS_DIR}")
    logging.info(f" Log File:      ../logs/cross_domain_eval.txt")
    logging.info(f" Model checkpoints: {OUTPUT_DIR}")
    logging.info("="*80 + "\n")