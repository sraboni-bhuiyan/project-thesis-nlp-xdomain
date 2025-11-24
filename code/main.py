# Main script to run the preprocessing pipeline

from data_processors import load_datasets, preprocess_datasets

def main():
    # Load datasets
    datasets = load_datasets()
    
    # Preprocess and split datasets
    splits = preprocess_datasets(datasets)
    
    # Save splits to CSV
    for name, (train, val, test) in splits.items():
        train.to_csv(f'../data/processed/{name}_train.csv', index=False)
        val.to_csv(f'../data/processed/{name}_val.csv', index=False)
        test.to_csv(f'../data/processed/{name}_test.csv', index=False)
        print(f"Saved splits for {name}: train={len(train)}, val={len(val)}, test={len(test)}")

if __name__ == "__main__":
    main()