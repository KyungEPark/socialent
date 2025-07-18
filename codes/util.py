import pandas as pd
import os
import re
import numpy as np
import string
from sentence_transformers import SentenceTransformer, util

# Preprocess keywords
def preprocess_keywords(df):
    category_keywords = {}
    seen = set()

    for col in df.columns:
        # Drop NA, lower, strip, remove punctuation if needed
        cleaned = (
            df[col].dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r'[^\w\s]', '', regex=True)
        )
        # Remove duplicates across categories
        unique_cleaned = [kw for kw in cleaned if kw not in seen]
        seen.update(unique_cleaned)
        category_keywords[col] = unique_cleaned

    return category_keywords

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


# Call csv
def call_csv(file_path):
    full_path = os.path.join(os.getcwd(), file_path)
    data = pd.read_csv(full_path)
    return data

# OpenAI calling
def call_openai(prompt, model="gpt-3.5-turbo",  max_tokens=1000):
    import openai
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message['content']