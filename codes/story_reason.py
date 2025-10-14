from util import *
import huggingface
import argparse
import os
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

def main(model_name, savefile):
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=hf_token
    )

    # Load your text data
    storytexts = pd.read_csv('data/raw/storytexts_sample.csv')

    # Define prompts and column names
    prompts = [
        (storytexts, "text", "storytelling_pred", "Analysiere, ob der Website-Text persönliche Geschichten von Gründer:innen, Erfahrungsberichte von Begünstigten oder emotionale Fallstudien, Bilder und Videos enthält, die genutzt werden, um die Mission und Werte des Unternehmens authentisch zu vermitteln. Suche gezielt nach Passagen, in denen individuelle Lebenswege, direkte Zitate oder Geschichten von Betroffenen im Mittelpunkt stehen und eine emotionale Bindung zu den Stakeholdern erzeugt werden soll. Achte darauf, dass diese Narrative nicht primär institutionelle Konformität oder gesellschaftliche Trends adressieren, sondern auf Authentizität, Missionstreue und Vertrauensaufbau durch persönliche Erfahrungen setzen. Prüfe, ob das Storytelling dazu dient, die emotionale Identifikation mit dem Unternehmen zu stärken und dessen Werte auf einer persönlichen Ebene zu vermitteln. ")
        ]

    # Process each dataframe and store in a dictionary
    processed_dfs = {}
    for df, text_col, label_col, prompt in prompts:
        print(f"Processing {label_col} for {len(df)} rows")
        reason_col = label_col.replace("_pred", "_reason") if label_col.endswith("_pred") else f"{label_col}_reason"

        # sliding_window_labeling now returns (final_reason, final_prediction)
        results = [sliding_window_labeling(prompt, text, model, tokenizer) for text in df[text_col]]

        if results:
            reasons, preds = zip(*results)
            df[label_col] = list(preds)
            df[reason_col] = list(reasons)
        else:
            df[label_col] = []
            df[reason_col] = []

        processed_dfs[label_col] = df

    # Save each processed dataframe separately
    for label_col, df in processed_dfs.items():
        save_path = os.path.join('data/output', f"{label_col}_{os.path.basename(savefile)}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model to be used")
    parser.add_argument("--savefile", type=str, required=True,
                        help="Save results")
    args = parser.parse_args()

    main(args.model, args.savefile)