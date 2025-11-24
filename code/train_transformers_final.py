"""
IN-DOMAIN SENTIMENT ANALYSIS TRAINING SCRIPT

OUTPUT FILES:
- metrics_summary.csv (in-domain training results with detailed timings)
- training_log.txt (complete execution log)
- outputs/models/ (model checkpoints for 6 models × 4 datasets = 24 combinations)
- outputs/reports/ (detailed JSON evaluation reports)

PHASE: Training only (6 models × 4 datasets = 24 training runs)
Cross-domain evaluation is in a separate script: train_transformers_cross_domain.py
"""

import os
import pandas as pd
import torch
import logging
from datetime import datetime
import time
import gc
import json
import csv
# from collections import defaultdict

# IMPORT TRANSFORMER LIBRARIES
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report
)

# SECTION 1: SETUP AND CONFIGURATION
# ============================================================================

# output directories
DATA_DIR = '../data/processed/'
OUTPUT_DIR = '../outputs/models/'
REPORTS_DIR = '../outputs/reports/'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs('../outputs/results/', exist_ok=True)
os.makedirs('../logs/', exist_ok=True)

# LOGGING CONFIGURATION

# Dual logging: console + file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/training_log.txt', encoding='utf-8'),  # Log to file with UTF-8
        logging.StreamHandler()  # Log to console
    ]
)

# MODEL AND DATASET DEFINITIONS

# Six transformer models for comparison
MODELS = {
    'distilbert': 'distilbert-base-uncased',
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'albert': 'albert-base-v2',
    'electra': 'google/electra-base-discriminator',
    'xlnet': 'xlnet-base-cased'
}

# Four datasets across two domains: social media and product reviews
DATASETS = {
    'sentiment140': {'path': 'sentiment140_train.csv', 'num_labels': 2},    # Twitter (binary)
    'reddit': {'path': 'reddit_train.csv', 'num_labels': 3},               # Reddit (ternary)
    'amazon_combined': {'path': 'amazon_combined_train.csv', 'num_labels': 3},  # Amazon (ternary)
    'imdb': {'path': 'imdb_train.csv', 'num_labels': 2}                    # IMDb (binary)
}

# SECTION 2: METRICS AND MODEL INFORMATION FUNCTIONS
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

def get_model_memory_footprint(model_name):
    model_sizes = {
        'distilbert': 66_955_009,
        'albert': 11_683_585,
        'bert': 109_482_240,
        'electra': 110_290_564,
        'roberta': 124_645_632,
        'xlnet': 340_102_144
    }
    return model_sizes.get(model_name, 0)


# SECTION 3: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_dataset(dataset_name, split='train'):
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


# SECTION 4: MAIN TRAINING FUNCTION WITH TIMING
# ============================================================================

def train_model(model_name, dataset_name, output_dir):
    
    # START: Outer timing (complete pipeline)
    pipeline_start = time.time()
    
    logging.info(f"\n{'='*80}")
    logging.info(f"START TRAINING: {model_name.upper()} on {dataset_name.upper()}")
    logging.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"GPU Available: {torch.cuda.is_available()}")
    logging.info(f"{'='*80}")
    
    try:
        # ====================================================================
        # PHASE 1: LOAD MODEL AND TOKENIZER
        # ====================================================================
        load_start = time.time()
        logging.info(f"Loading model and tokenizer: {MODELS[model_name]}")
        
        tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
        model = AutoModelForSequenceClassification.from_pretrained(
            MODELS[model_name],
            num_labels=DATASETS[dataset_name]['num_labels']
        )
        
        load_time = time.time() - load_start
        logging.info(f" Model & Tokenizer loaded in {load_time:.2f}s")
        
        # ====================================================================
        # PHASE 2: LOAD DATASETS
        # ====================================================================
        data_load_start = time.time()
        logging.info(f"Loading datasets: {dataset_name}")
        
        train_dataset = load_dataset(dataset_name, 'train')
        val_dataset = load_dataset(dataset_name, 'val')
        
        data_load_time = time.time() - data_load_start
        logging.info(f" Datasets loaded in {data_load_time:.2f}s")
        logging.info(f"  Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
        
        # ====================================================================
        # PHASE 3: TOKENIZATION
        # ====================================================================
        tokenize_start = time.time()
        logging.info(f"Tokenizing datasets (max_length=256)")
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'], 
                padding='max_length', 
                truncation=True, 
                max_length=256
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        tokenize_time = time.time() - tokenize_start
        logging.info(f" Tokenization completed in {tokenize_time:.2f}s")
        
        # ====================================================================
        # PHASE 4: TRAINING CONFIGURATION
        # ====================================================================
        logging.info(f"Setting up training configuration:")
        logging.info(f"  Learning rate: 2e-5")
        logging.info(f"  Batch size: 8")
        logging.info(f"  Epochs: 4")
        logging.info(f"  Mixed precision (FP16): {torch.cuda.is_available()}")
        logging.info(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, f"{model_name}_{dataset_name}"),
            eval_strategy='steps',
            eval_steps=2000,
            save_strategy='no',
            logging_steps=500,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            fp16=torch.cuda.is_available(),  # Mixed precision training
            num_train_epochs=4,
            weight_decay=0.01,
            logging_dir='../logs/',
            report_to='none',
            seed=42  # Reproducibility
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
        
        # ====================================================================
        # PHASE 5: MAIN TRAINING LOOP (KEY TIMING)
        # ====================================================================
        logging.info(f"\nStarting training loop on {'GPU' if torch.cuda.is_available() else 'CPU'}...")
        training_start = time.time()
        
        trainer.train()  # This is where the actual training happens
        
        training_time = time.time() - training_start
        logging.info(f" Training completed in {training_time/60:.2f} minutes ({training_time:.2f}s)")
        
        # ====================================================================
        # PHASE 6: SAVE MODEL
        # ====================================================================
        save_start = time.time()
        model_path = os.path.join(output_dir, f"{model_name}_{dataset_name}")
        
        logging.info(f"Saving model to: {model_path}")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        save_time = time.time() - save_start
        logging.info(f"Model saved in {save_time:.2f}s")
        
        # ====================================================================
        # PHASE 7: VALIDATION EVALUATION
        # ====================================================================
        eval_start = time.time()
        logging.info(f"Evaluating on validation set...")
        
        predictions = trainer.predict(val_dataset)
        preds = predictions.predictions.argmax(-1)
        report = classification_report(predictions.label_ids, preds, output_dict=True)
        
        eval_time = time.time() - eval_start
        
        # Extract key metrics
        macro_f1 = report['macro avg']['f1-score']
        weighted_f1 = report['weighted avg']['f1-score']
        acc = report['accuracy']
        
        logging.info(f"Validation evaluation completed in {eval_time:.2f}s")
        logging.info(f"Results:")
        logging.info(f"  -- Macro-F1:    {macro_f1:.4f}")
        logging.info(f"  -- Weighted-F1: {weighted_f1:.4f}")
        logging.info(f"  -- Accuracy:    {acc:.4f}")
        
        # Save JSON report
        rep_path = os.path.join(REPORTS_DIR, f"{model_name}_{dataset_name}_val.json")
        with open(rep_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # ====================================================================
        # PHASE 8: LOG RESULTS TO CSV (ENHANCEMENT)
        # ====================================================================
        
        # Calculate total pipeline time
        pipeline_time = time.time() - pipeline_start
        
        # Append to detailed CSV summary
        csv_path = '../outputs/results/metrics_summary.csv'
        write_header = not os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                # Header row with all columns for analysis
                w.writerow([
                    'timestamp', 'datetime', 'dataset', 'model', 'num_labels',
                    'n_train', 'n_val', 'max_len', 'epochs', 'lr',
                    'macro_f1', 'weighted_f1', 'accuracy',
                    'model_params', 'gpu_available',
                    'load_time_s', 'data_load_time_s', 'tokenize_time_s', 
                    'training_time_s', 'eval_time_s', 'save_time_s', 'total_time_s'
                ])
            
            # Data row
            w.writerow([
                int(time.time()),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                dataset_name,
                model_name,
                DATASETS[dataset_name]['num_labels'],
                len(train_dataset),
                len(val_dataset),
                256, 2, 2e-5,
                macro_f1, weighted_f1, acc,
                get_model_memory_footprint(model_name),
                torch.cuda.is_available(),
                load_time, data_load_time, tokenize_time,
                training_time, eval_time, save_time, pipeline_time
            ])
        
        # ====================================================================
        # SUMMARY LOGGING
        # ====================================================================
        logging.info(f"\n{'='*80}")
        logging.info(f"TRAINING COMPLETED: {model_name.upper()} on {dataset_name.upper()}")
        logging.info(f"Total Pipeline Time: {pipeline_time/60:.2f} minutes ({pipeline_time:.2f}s)")
        logging.info(f"\nTiming Breakdown:")
        logging.info(f"  Model Load:      {load_time:>8.2f}s")
        logging.info(f"  Data Load:       {data_load_time:>8.2f}s")
        logging.info(f"  Tokenization:    {tokenize_time:>8.2f}s")
        logging.info(f"  Training:        {training_time:>8.2f}s ({training_time/60:.1f} min)  ← MAIN TIMING")
        logging.info(f"  Evaluation:      {eval_time:>8.2f}s")
        logging.info(f"  Model Save:      {save_time:>8.2f}s")
        logging.info(f"{'='*80}\n")
        
        # ====================================================================
        # CLEANUP
        # ====================================================================
        del trainer, model, tokenizer, train_dataset, val_dataset, predictions
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        logging.error(f"ERROR training {model_name} on {dataset_name}: {str(e)}", exc_info=True)
        return False


# SECTION 5: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # STARTUP: Log experiment information
    # ========================================================================
    logging.info("\n" + "="*80)
    logging.info("EXPERIMENT STARTED: Cross-Domain Sentiment Analysis")
    logging.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logging.info(f"Random Seed: 42 (reproducibility)")
    logging.info("="*80 + "\n")
    
    overall_start = time.time()
    
    # ========================================================================
    # IN-DOMAIN TRAINING (24 runs)
    # ========================================================================
    logging.info("PHASE 1: IN-DOMAIN TRAINING (24 runs: 6 models × 4 datasets)")
    logging.info("-" * 80)
    
    for model_name in sorted(MODELS.keys()):
        for dataset_name in sorted(DATASETS.keys()):
            train_model(model_name, dataset_name, OUTPUT_DIR)
    
    training_phase_time = time.time() - overall_start
    logging.info(f" Phase 1 completed in {training_phase_time/60:.2f} minutes\n")
    
    # Calculate total execution time
    total_time = time.time() - overall_start
    
    # ========================================================================
    # COMPLETION: Final summary
    # ========================================================================
    logging.info("\n" + "="*80)
    logging.info("EXPERIMENT COMPLETED: IN-DOMAIN TRAINING")
    logging.info(f"Total Execution Time: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)")
    logging.info(f"Training Phase (24 runs: 6 models × 4 datasets): {training_phase_time/60:.2f} minutes")
    logging.info(f"\nOutput files saved to:")
    logging.info(f"  Metrics Summary: ../outputs/results/metrics_summary.csv")
    logging.info(f"  Training Log: ../logs/training_log.txt")
    logging.info(f"  Model Checkpoints: {OUTPUT_DIR}")
    logging.info(f"  JSON Reports: {REPORTS_DIR}")
    logging.info("="*80 + "\n")
    