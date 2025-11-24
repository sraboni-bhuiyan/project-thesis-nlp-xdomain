import pandas as pd
import json
import gzip
import os
from collections import Counter
from sklearn.utils import resample
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
DATA_DIR = 'data/raw/'
AMAZON_FASHION_FILE = os.path.join(DATA_DIR, 'Amazon_Fashion.jsonl')
CLOTHING_SHOES_JEWELRY_FILE = os.path.join(DATA_DIR, 'Clothing_Shoes_and_Jewelry.jsonl')
OUTPUT_CSV = os.path.join(DATA_DIR, 'amazon_combined_reviews.csv')
ANALYSIS_FILE = os.path.join(DATA_DIR, 'amazon_combined_analysis.txt')

# Sentiment mapping function
def map_amazon_sentiment(rating):
    if pd.isna(rating):
        return None
    try:
        rating = float(rating)
        if rating <= 2:
            return 0  # Negative
        elif rating == 3:
            return 1  # Neutral
        else:
            return 2  # Positive
    except (ValueError, TypeError):
        return None

# Function to read .jsonl file in chunks
def read_jsonl_chunked(file_path, chunk_size=100000, max_reviews=None):
    data = []
    total_loaded = 0
    logging.info(f"Reading {file_path}...")
    
    try:
        # Check if file is compressed
        if file_path.endswith('.gz'):
            opener = gzip.open(file_path, 'rt', encoding='utf-8')
        else:
            opener = open(file_path, 'r', encoding='utf-8')
        
        with opener as f:
            chunk = []
            for line in f:
                try:
                    chunk.append(json.loads(line.strip()))
                    if len(chunk) >= chunk_size:
                        chunk_df = pd.DataFrame(chunk)[['text', 'rating']].dropna(subset=['text', 'rating'])
                        chunk_df = chunk_df[chunk_df['text'].str.strip() != '']
                        data.append(chunk_df)
                        total_loaded += len(chunk_df)
                        logging.info(f"Processed {total_loaded} valid reviews from {file_path}")
                        chunk = []
                        if max_reviews and total_loaded >= max_reviews:
                            break
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line in {file_path}")
            
            # Process remaining chunk
            if chunk:
                chunk_df = pd.DataFrame(chunk)[['text', 'rating']].dropna(subset=['text', 'rating'])
                chunk_df = chunk_df[chunk_df['text'].str.strip() != '']
                data.append(chunk_df)
                total_loaded += len(chunk_df)
                logging.info(f"Processed {total_loaded} valid reviews from {file_path}")
        
        if not data:
            raise ValueError(f"No valid data found in {file_path}")
        
        df = pd.concat(data, ignore_index=True)
        if max_reviews:
            df = df.sample(n=min(max_reviews, len(df)), random_state=42)
        logging.info(f"Loaded {len(df)} reviews from {file_path}")
        return df
    
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading {file_path}: {str(e)}")
        raise

# Main processing function
def process_amazon_reviews():
    try:
        # Step 1: Read .jsonl files in chunks
        # Load ~2.5M reviews from Amazon_Fashion
        fashion_df = read_jsonl_chunked(AMAZON_FASHION_FILE, chunk_size=100000, max_reviews=2500000)
        
        # Load ~2.5M reviews from Clothing_Shoes_and_Jewelry
        clothing_df = read_jsonl_chunked(CLOTHING_SHOES_JEWELRY_FILE, chunk_size=100000, max_reviews=2500000)
        
        # Step 2: Combine data
        logging.info("Combining datasets...")
        df = pd.concat([fashion_df, clothing_df], ignore_index=True)
        logging.info(f"Total combined reviews: {len(df)}")

        # Step 3: Map ratings to sentiments
        logging.info("Mapping sentiments...")
        df['sentiment'] = df['rating'].apply(map_amazon_sentiment)
        df = df.dropna(subset=['sentiment'])
        df['sentiment'] = df['sentiment'].astype(int)
        logging.info(f"Reviews after sentiment mapping: {len(df)}")

        # Step 4: Analyze data
        logging.info("\n--- Data Analysis ---")
        # Sentiment distribution
        sentiment_counts = Counter(df['sentiment'])
        logging.info("Sentiment Distribution:")
        for sentiment, count in sorted(sentiment_counts.items()):
            label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}[sentiment]
            logging.info(f"{label} ({sentiment}): {count} ({count/len(df)*100:.2f}%)")

        # Sample reviews by sentiment
        logging.info("\nSample Reviews by Sentiment:")
        for sentiment in sorted(df['sentiment'].unique()):
            label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}[sentiment]
            logging.info(f"\nSentiment {label} ({sentiment}):")
            samples = df[df['sentiment'] == sentiment][['text', 'rating']].head(2)
            for _, row in samples.iterrows():
                logging.info(f"Text: {row['text'][:100]}...")
                logging.info(f"Rating: {row['rating']}")

        # Step 5: Balance dataset
        logging.info("\nBalancing dataset...")
        target_size = 20000  # 20,000 per class for ~60,000 total
        balanced_dfs = []
        for sentiment in sorted(df['sentiment'].unique()):
            df_sent = df[df['sentiment'] == sentiment]
            if len(df_sent) > target_size:
                df_sent = df_sent.sample(target_size, random_state=42)
            elif len(df_sent) < target_size:
                df_sent = resample(df_sent, n_samples=target_size, random_state=42, replace=True)
            balanced_dfs.append(df_sent)
        
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        logging.info(f"Balanced dataset size: {len(df_balanced)}")
        logging.info("Balanced Sentiment Distribution:")
        logging.info(Counter(df_balanced['sentiment']))

        # Step 6: Save to CSV
        output_df = df_balanced[['text', 'sentiment']].copy()
        output_df.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"\nSaved balanced dataset to {OUTPUT_CSV}")
        logging.info(f"CSV columns: {list(output_df.columns)}")
        logging.info(f"Sample CSV data:")
        logging.info(output_df.head().to_string())

        # Step 7: Save analysis for thesis
        logging.info(f"Saving analysis to {ANALYSIS_FILE}")
        with open(ANALYSIS_FILE, 'w', encoding='utf-8') as f:
            f.write("Amazon Combined Reviews Analysis\n")
            f.write("==============================\n\n")
            f.write(f"Total Reviews (Combined): {len(df)}\n")
            f.write(f"Balanced Dataset Size: {len(df_balanced)}\n\n")
            f.write("Sentiment Distribution (Raw):\n")
            for sentiment, count in sorted(sentiment_counts.items()):
                label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}[sentiment]
                f.write(f"{label} ({sentiment}): {count} ({count/len(df)*100:.2f}%)\n")
            f.write("\nBalanced Sentiment Distribution:\n")
            f.write(str(Counter(df_balanced['sentiment'])))
        logging.info(f"Saved analysis to {ANALYSIS_FILE}")

    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    process_amazon_reviews()