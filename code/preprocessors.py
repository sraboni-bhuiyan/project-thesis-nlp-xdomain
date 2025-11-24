# Text preprocessing functions for social media and reviews

import pandas as pd
import re
import emoji

def preprocess_social_media(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = emoji.replace_emoji(text, replace='')
    slang_dict = {
        'u': 'you', 'lol': 'laughing out loud', 'brb': 'be right back',
        'idk': 'i do not know', 'smh': 'shaking my head', 'thx': 'thanks',
        'wat': 'what', 'pls': 'please', 'gr8': 'great', 'ppl': 'people',
        'bc': 'because', 'tbh': 'to be honest', 'lmao': 'laughing my ass off'
    }
    words = text.split()
    text = ' '.join(slang_dict.get(word.lower(), word) for word in words)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def preprocess_reviews(text, max_length=512):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split()[:max_length])
    return text.strip()