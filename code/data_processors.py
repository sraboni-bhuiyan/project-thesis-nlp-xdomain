# Functions for loading, preprocessing, and splitting datasets

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from math import gcd
from config import DATASET_CONFIG
from preprocessors import preprocess_social_media, preprocess_reviews
from sentiment_mappers import (
    map_sentiment140, map_reddit_sentiment, map_amazon_sentiment, map_imdb_sentiment
)

def load_datasets():
    datasets = {}
    try:
        for name, config in DATASET_CONFIG.items():
            df = pd.read_csv(config['path'], **config.get('load_args', {}))
            if config['columns']:
                df.columns = config['columns']
            if config['rename_columns']:
                df = df.rename(columns=config['rename_columns'])
            datasets[name] = df
        return datasets
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        exit(1)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        exit(1)

def preprocess_datasets(datasets):
    try:
        text_preprocessors = {
            'social_media': preprocess_social_media,
            'reviews': preprocess_reviews
        }
        sentiment_mappers = {
            'sentiment140': map_sentiment140,
            'reddit': map_reddit_sentiment,
            'amazon': map_amazon_sentiment,
            'imdb': map_imdb_sentiment
        }

        # -----------------------------
        # First pass: clean + map + drop NAs, stash prepared frames
        # -----------------------------
        prepared = {}
        for name, df in datasets.items():
            # Capping to 50,000 rows for all datasets
            df = df.sample(n=50000, random_state=42)

            # Step 1: Clean text
            df['text'] = df['text'].apply(
                text_preprocessors[DATASET_CONFIG[name]['text_preprocessor']]
            )

            # Step 2: Map sentiment labels (only if configured)
            if DATASET_CONFIG[name]['sentiment_mapper']:
                df['sentiment'] = df['sentiment'].apply(
                    sentiment_mappers[DATASET_CONFIG[name]['sentiment_mapper']]
                )

            # Step 3: Handle missing values
            print(f"\n{name} - Missing values:")
            print("Text:", df['text'].isnull().sum())
            print("Sentiment:", df['sentiment'].isnull().sum())
            df.dropna(subset=['text', 'sentiment'], inplace=True)

            prepared[name] = df

        # -----------------------------
        # Compute an equal target size per dataset (balanced by class)
        # -----------------------------
        def _lcm(a, b):
            return a * b // gcd(a, b) if a and b else a or b

        # For each dataset, the maximum balanced total is (num_classes * min_per_class)
        caps = {}
        class_sizes = set()
        for name, df in prepared.items():
            cc = Counter(df['sentiment'])
            if not cc:
                continue
            num_classes = len(cc)
            class_sizes.add(num_classes)
            min_per_class = min(cc.values())
            caps[name] = {
                'num_classes': num_classes,
                'min_per_class': min_per_class,
                'cap_total': num_classes * min_per_class
            }

        if not caps:
            print("Error: No data available after preprocessing.")
            exit(1)

        # Choose the largest target_total <= min(cap_total) that is divisible by all class counts
        lcm_all = 1
        for n in class_sizes:
            lcm_all = _lcm(lcm_all, n)

        raw_target = min(v['cap_total'] for v in caps.values())
        target_total = raw_target - (raw_target % lcm_all) if lcm_all else raw_target
        if target_total == 0:
            target_total = raw_target  # fallback (should rarely happen)

        # -----------------------------
        # Second pass: per-class random downsample to target_total, then splits
        # -----------------------------
        splits = {}
        for name, df in prepared.items():
            num_classes = caps[name]['num_classes']
            per_class_k = target_total // num_classes

            # Balanced random sample per class
            parts = []
            # sorted() keeps labels in a stable order; sampling is randomized by seed
            for label in sorted(df['sentiment'].dropna().unique()):
                df_c = df[df['sentiment'] == label]
                n_take = min(per_class_k, len(df_c))
                parts.append(df_c.sample(n=n_take, random_state=42))

            df_bal = pd.concat(parts).sample(frac=1.0, random_state=42)  # shuffle

            # Split dataset: 80/10/10 
            train, temp = train_test_split(
                df_bal, test_size=0.2, stratify=df_bal['sentiment'], random_state=42
            )
            val, test = train_test_split(
                temp, test_size=0.5, stratify=temp['sentiment'], random_state=42
            )
            splits[name] = (train, val, test)

            # Verify preprocessing
            print(f"\n{name} - Sample preprocessed data:")
            print(df_bal[['text', 'sentiment']].head())
            print(f"{name} - Class distribution:")
            print("Train:", Counter(train['sentiment']))
            print("Val:", Counter(val['sentiment']))
            print("Test:", Counter(test['sentiment']))

        return splits

    except Exception as e:
        print(f"Error preprocessing datasets: {e}")
        exit(1)
