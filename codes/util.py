import pandas as pd
import os
import re
import numpy as np
import string
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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
        messages=[{"role": "user", "content": f"{prompt}\n\n{text}"}],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message['content']

# Using huggingface models
def load_huggingface_model(prompt, text, model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    messages = [
        {"role": "user", "content": f"{prompt} Antworte nur mit Ja oder Nein. \n\n{text}"}
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
        full_prompt = prompt + "Antworte nur mit Ja oder Nein." + "\n\n" + text
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

def chunk_text(text, tokenizer, chunk_size=10000, stride=512):
    input_ids = tokenizer.encode(text, truncation=False)
    chunks = []

    for i in range(0, len(input_ids), stride):
        chunk_ids = input_ids[i:i + chunk_size]
        if not chunk_ids:
            break
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if i + chunk_size >= len(input_ids):
            break
    return chunks

def get_chunk_prediction(prompt, chunk, model, tokenizer):
    instruction = (
        "Bitte antworte genau in folgendem Format (auf Deutsch):\n"
        "Grund: <kurze Erklärung>\n"
        "Vorhersage: <Ja oder Nein>\n\n"
        "Gib nur diese zwei Zeilen aus. Füge nichts anderes hinzu."
    )
    messages = [
        {"role": "user", "content": f"{prompt}\n\n{instruction}"},
        {"role": "user", "content": chunk}
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False
        ).to(model.device)
    else:
        full_prompt = f"{prompt}\n\n{instruction}\n\n{chunk}"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)  # increase tokens
        try:
            start_idx = inputs["input_ids"].shape[-1]
            raw = tokenizer.decode(outputs[0][start_idx:], skip_special_tokens=True).strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
        except Exception:
            raw = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()

    # broaden regex to catch English/German variants and common synonyms
    reason_match = re.search(r"(?:reason|grund|erklär(?:ung)?)[\s:]*([^\n]+)", raw, flags=re.I)
    if not reason_match:
        # try multiline block before prediction
        reason_match = re.search(r"(.+?)(?=\n(?:prediction|vorhersage|prediction)[\s:])", raw, flags=re.I | re.S)

    reason = reason_match.group(1).strip() if reason_match else ""

    pred_match = re.search(r"(?:prediction|vorhersage)[\s:]*([^\n]+)", raw, flags=re.I)
    pred_text = pred_match.group(1).strip() if pred_match else raw.strip()  # fallback: whole output

    pred_text_lower = pred_text.lower()
    if re.search(r"\b(1|yes|ja|y|true)\b", pred_text_lower):
        pred = 1
    elif re.search(r"\b(0|no|nein|n|false)\b", pred_text_lower):
        pred = 0
    else:
        digit_match = re.search(r"\b([01])\b", pred_text_lower)
        pred = int(digit_match.group(1)) if digit_match else 0

    # debug: print raw when no reason found (optional)
    if not reason:
        print("No reason parsed for chunk. Raw output:", repr(raw))

    return reason, pred, raw

def sliding_window_labeling(prompt, text, model, tokenizer, chunk_size=10000, stride=512, stop_on_positive=True):
    """
    Wrapper that splits text into sliding-window chunks, runs get_chunk_prediction on each chunk,
    and returns a tuple (combined_reason, overall_prediction) where overall_prediction is 1 if any
    chunk predicted positive, else 0. This matches the expected return used by the caller.
    """
    if not text or not str(text).strip():
        return "", 0

    chunks = chunk_text(text, tokenizer, chunk_size=chunk_size, stride=stride)
    if not chunks:
        return "", 0

    results = []
    overall = 0

    for idx, chunk in enumerate(chunks):
        reason, pred, raw = get_chunk_prediction(prompt, chunk, model, tokenizer)
        results.append({"index": idx, "reason": reason, "pred": pred, "raw": raw})
        if pred == 1:
            overall = 1
            if stop_on_positive:
                break

    # Choose combined reason: first positive reason, otherwise first chunk reason if available
    positive_reasons = [r["reason"] for r in results if r["pred"] == 1 and r["reason"]]
    if positive_reasons:
        combined_reason = positive_reasons[0]
    else:
        combined_reason = results[0]["reason"] if results else ""

    return combined_reason, overall
