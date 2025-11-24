# Sentiment mapping functions for different datasets

import pandas as pd

def map_sentiment140(sentiment):
    if sentiment == 0: return 0
    if sentiment == 4: return 1
    return None

def map_reddit_sentiment(score):
    if score > 0:
        return 2
    elif score == 0:
        return 1
    else:
        return 0

def map_amazon_sentiment(rating):
    if pd.isna(rating):
        return 0
    rating = float(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

def map_imdb_sentiment(sentiment):
    if isinstance(sentiment, str):
        return 1 if sentiment.lower() == 'positive' else 0
    return 1 if sentiment == 1 else 0