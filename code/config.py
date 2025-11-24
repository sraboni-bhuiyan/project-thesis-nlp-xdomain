# Dataset configuration for preprocessing pipeline

DATASET_CONFIG = {
    'sentiment140': {
        'path': '../data/raw/sentiment140.csv',
        'columns': ['sentiment', 'id', 'date', 'query', 'user', 'text'],
        'load_args': {'encoding': 'latin-1', 'header': None},
        'rename_columns': {},
        'text_preprocessor': 'social_media',
        'sentiment_mapper': 'sentiment140'
    },
    'reddit': {
        'path': '../data/raw/reddit_comments.csv',
        'columns': None,
        'load_args': {},
        'rename_columns': {'body': 'text', 'score': 'sentiment'},
        'text_preprocessor': 'social_media',
        'sentiment_mapper': 'reddit'
    },
    'amazon_combined': {
        'path': '../data/raw/amazon_combined_reviews.csv',
        'columns': ['text', 'sentiment'],
        'load_args': {},
        'rename_columns': {},
        'text_preprocessor': 'reviews',
        'sentiment_mapper': None
    },
    'imdb': {
        'path': '../data/raw/imdb.csv',
        'columns': None,
        'load_args': {},
        'rename_columns': {'review': 'text'},
        'text_preprocessor': 'reviews',
        'sentiment_mapper': 'imdb'
    }
}