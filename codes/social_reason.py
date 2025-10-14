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
    socialtexts = pd.read_csv('data/raw/socialtexts_sample.csv')

    # Define prompts and column names
    prompts = [
        (socialtexts, "text", "social_reference_pred", "Untersuche, ob das Unternehmen in seinen Texten explizit auf aktuelle gesellschaftliche Bewegungen oder Debatten wie Klimaschutz, Diversität, Inklusion oder soziale Gerechtigkeit Bezug nimmt. Suche nach Hinweisen, dass das Unternehmen sich öffentlich mit diesen Themen positioniert, Partnerschaften mit entsprechenden Initiativen eingeht oder sich als Teil einer größeren gesellschaftlichen Bewegung versteht. Achte darauf, dass der Fokus auf der aktiven Bezugnahme zu gesellschaftlichen Trends liegt, um normative Legitimität zu gewinnen, und nicht auf institutioneller Transparenz, persönlichem Storytelling oder hybrider Identität. Prüfe, ob das Engagement für diese Themen als zentrales Element der Kommunikation genutzt wird, um das Unternehmen als gesellschaftlich verantwortlichen Akteur zu präsentieren.")
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