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
    hybtexts = pd.read_csv('data/raw/hybtexts_sample.csv')
    hybtexts = hybtexts[0:10]

    # Define prompts and column names
    prompts = [
        (hybtexts, "text", "hybrid_identity_pred", "Analysiere den Website-Text daraufhin, ob das Unternehmen explizit eine doppelte Identität kommuniziert, indem es sowohl seine soziale Mission als auch seine geschäftliche Kompetenz als gleichwertige und miteinander verbundene Kernbestandteile darstellt. Suche nach Passagen, in denen die bewusste Integration von sozialem Impact und unternehmerischem Erfolg betont wird, etwa durch Formulierungen, die moralische Verantwortung und betriebswirtschaftliche Professionalität als untrennbar beschreiben. Achte darauf, dass die Organisation nicht nur beide Aspekte nebeneinander aufführt, sondern aktiv als hybrides Selbstverständnis präsentiert – beispielsweise durch die Verknüpfung von sozialen Zielen mit Wachstums- oder Effizienzkriterien. Prüfe, ob das Unternehmen diese hybride Identität als Alleinstellungsmerkmal nutzt, um sowohl moralische als auch pragmatische Legitimität zu signalisieren und sich damit von rein auf Transparenz, Storytelling oder aktuelle Bewegungen fokussierten Ansätzen abgrenzt. Stelle sicher, dass die Texte deutlich machen, dass soziale Wirkung und wirtschaftlicher Erfolg nicht als Gegensätze, sondern als integraler Bestandteil der Unternehmensidentität verstanden werden.")
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