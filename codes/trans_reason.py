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
    transtexts = pd.read_csv('data/raw/transtexts_sample.csv')


    # Define prompts and column names
    prompts = [
        (transtexts, "text", "transparency_pred", "Analysiere den Website-Text daraufhin, ob das Unternehmen gezielt auf institutionelle Mechanismen wie Zertifizierungen (z. B. B Corp), Impact-Reports oder die Einhaltung von Compliance-Standards verweist, um Transparenz und Verantwortlichkeit zu demonstrieren. Suche nach Textstellen, in denen die Veröffentlichung von Berichten, die Teilnahme an externen Audits oder die Übernahme regulatorischer Vorgaben als Beleg für vertrauenswürdiges und regelkonformes Handeln hervorgehoben werden. Achte darauf, dass der Fokus auf der institutionellen Legitimation durch externe Prüfungen und Standards liegt, nicht auf persönlichen Geschichten oder gesellschaftlichen Bewegungen. Prüfe, ob das Unternehmen explizit betont, dass diese Maßnahmen dazu dienen, Glaubwürdigkeit und Rechenschaftspflicht gegenüber Stakeholdern zu stärken.")
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