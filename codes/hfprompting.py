from util import *
import huggingface
import argparse
import os

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
    texts = pd.read_parquet('data/subsettext100_socialco_20250718.parquet')
    texts = texts[0:4]
    
    # Define prompts and column names
    prompts_df = pd.read_csv('data/raw/prompts.csv', header=None)
    colnames = prompts_df.iloc[0].tolist()
    prompts = prompts_df.iloc[1].tolist()

    # Initialize output columns
    for col in colnames:
        if col not in texts.columns:
            texts[col] = None

    # Run sliding window classification
    for row in texts.itertuples():
        text = row.text  # Assuming the column name is 'text'
        for i, prompt in enumerate(prompts):
            print(f"Processing row {row.Index}, prompt {i+1}/{len(prompts)}")
            label, _ = sliding_window_labeling(prompt, text, model, tokenizer)

            # Convert label to binary (1 for 'yes', 0 for 'no')
            binary = 1 if str(label).strip().lower() == "ja" else 0

            # Save binary result in the strategy column
            texts.at[row.Index, colnames[i]] = binary

    # Save results
    savefile = os.path.join('data/output', os.path.basename(savefile))
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    texts.to_parquet(savefile, index=False)
    return texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model to be used")
    parser.add_argument("--savefile", type=str, required=True,
                        help="Save results")
    args = parser.parse_args()

    main(args.model, args.savefile)