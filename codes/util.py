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
def call_openai(prompt, text, model="gpt-3.5-turbo",  max_tokens=1000):
    import openai
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": f"{prompt}\n\n{text}"}]
        max_tokens=max_tokens
        temperature = 0.7
    )
    return response.choices[0].message['content']

# Using huggingface models
def load_huggingface_model(prompt, text, model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    messages = [
        {"role": "user", "content": f"{prompt}\n\n{text}"}
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        # Fallback for non-chat models
        full_prompt = prompt + "\n\n" + text
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


# Using huggingface models-locked
def load_huggingface_model_locked(prompt, text, model_name, hf_token):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token).to("cuda" if torch.cuda.is_available() else "cpu")

    messages = [
        {"role": "user", "content": f"{prompt}\n\n{text}"}
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        # Fallback for non-chat models
        full_prompt = prompt + "\n\n" + text
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)