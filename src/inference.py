# import argparse
# import os
# import logging
# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from datasets import load_dataset
# import nltk
# from nltk.tokenize import sent_tokenize
# import random
# from tqdm import tqdm
# from src.utils import clean_text

# # Set up logging
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )
# logger = logging.getLogger(__name__)

# # Download NLTK resources
# nltk.download('punkt', quiet=True)


# def load_model(model_path):
#     """Loads model and tokenizer from path"""
#     logger.info(f"Loading model and tokenizer from {model_path}")

#     # Load tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

#     # Move model to device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()

#     return model, tokenizer, device


# def generate_summary(model, tokenizer, document, max_input_length=1024, max_output_length=256, device="cpu"):
#     """Generates a summary for the given document"""
#     # Clean and prepare document
#     document = clean_text(document)

#     # Tokenize input
#     inputs = tokenizer(
#         document,
#         max_length=max_input_length,
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt"
#     )

#     # Move inputs to device
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     # Generate summary
#     with torch.no_grad():
#         output_ids = model.generate(
#             inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             max_length=max_output_length,
#             num_beams=4,
#             early_stopping=True
#         )

#     # Decode summary
#     summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     return summary


# def main():
#     # Parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
#     parser.add_argument("--dataset", type=str, default="alexfabbri/multi_news", help="Dataset name")
#     parser.add_argument("--split", type=str, default="test", help="Dataset split")
#     parser.add_argument("--num_samples", type=int, default=5, help="Number of examples to generate")
#     parser.add_argument("--output_file", type=str, default="summaries.txt", help="Output file path")
#     parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum input length")
#     parser.add_argument("--max_output_length", type=int, default=128, help="Maximum output length")
#     parser.add_argument("--seed", type=int, default=43, help="Random seed")
#     args = parser.parse_args()

#     # Set random seed
#     random.seed(args.seed)

#     # Load model and tokenizer
#     model, tokenizer, device = load_model(args.model_path)

#     # Load dataset
#     logger.info(f"Loading {args.split} split from {args.dataset}")
#     dataset = load_dataset(args.dataset, split=args.split)

#     # Select samples
#     if args.num_samples >= len(dataset):
#         indices = list(range(len(dataset)))
#     else:
#         indices = random.sample(range(len(dataset)), args.num_samples)

#     # Generate summaries
#     results = []
#     for idx in tqdm(indices, desc="Generating summaries"):
#         example = dataset[idx]

#         # Generate summary
#         generated_summary = generate_summary(
#             model=model,
#             tokenizer=tokenizer,
#             document=example["document"],
#             max_input_length=args.max_input_length,
#             max_output_length=args.max_output_length,
#             device=device
#         )

#         # logger.info(f"ref sum len: {len(example["summary"])}, gen sum len: {len(generated_summary)}")

#         # Store results
#         results.append({
#             "index": idx,
#             "document": example["document"],
#             "reference_summary": example["summary"],
#             "generated_summary": generated_summary
#         })

#     # Write results to file
#     with open(args.output_file, 'w', encoding='utf-8') as f:
#         for i, result in enumerate(results):
#             f.write(f"Example {i + 1} (Dataset index: {result['index']})\n")
#             f.write("=" * 80 + "\n")
#             f.write("DOCUMENT:\n")
#             f.write(result['document'][:1000] + "...\n\n")
#             f.write("REFERENCE SUMMARY:\n")
#             f.write(result['reference_summary'] + "\n\n")
#             f.write("GENERATED SUMMARY:\n")
#             f.write(result['generated_summary'] + "\n\n")
#             f.write("=" * 80 + "\n\n")
#             f.write("ref sum len: " + str(len(result['reference_summary'])) + " gen sum len: " + str(len(result['generated_summary'])) + "\n\n")

#     logger.info(f"Generated summaries saved to {args.output_file}")


# if __name__ == "__main__":
#     main()


# src/inference.py

# import argparse
# import logging
# import random
# import os
# import torch
# from tqdm import tqdm
# from datasets import load_dataset
# from src.model import load_model_and_tokenizer
# from src.utils import clean_text

# # Configure logging
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO
# )
# logger = logging.getLogger(__name__)

# def generate_summaries(
#     model,
#     tokenizer,
#     dataset_name: str,
#     split: str,
#     num_samples: int,
#     output_file: str,
#     max_input_length: int,
#     max_output_length: int,
#     seed: int
# ):
#     # Set seed
#     random.seed(seed)
#     torch.manual_seed(seed)

#     # Load dataset
#     logger.info(f"Loading '{split}' split from {dataset_name}")
#     ds = load_dataset(dataset_name, split=split)
#     total = len(ds)
#     if num_samples < total:
#         indices = random.sample(range(total), num_samples)
#     else:
#         indices = list(range(total))

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Generate and write summaries
#     with open(output_file, 'w', encoding='utf-8') as fout:
#         for idx in tqdm(indices, desc="Generating summaries"):
#             example = ds[idx]
#             raw_doc = example["document"]
#             cleaned_doc = clean_text(raw_doc)

#             # Tokenize and move to device
#             inputs = tokenizer(
#                 cleaned_doc,
#                 max_length=max_input_length,
#                 truncation=True,
#                 padding="max_length",
#                 return_tensors="pt"
#             ).to(device)

#             # Generate
#             with torch.no_grad():
#                 out_ids = model.generate(
#                     inputs["input_ids"],
#                     attention_mask=inputs["attention_mask"],
#                     max_length=max_output_length,
#                     num_beams=4,
#                     length_penalty=1.0,
#                     early_stopping=True,
#                     decoder_start_token_id=tokenizer.bos_token_id,   # ensure decoding starts correctly
#                     # no_repeat_ngram_size=2,                          # relax the constraint
#                 )

#             summary = tokenizer.decode(out_ids[0], skip_special_tokens=True)

#             # Write to file
#             fout.write(f"Example {idx}\n")
#             fout.write("-" * 80 + "\n")
#             fout.write("DOCUMENT:\n")
#             fout.write(raw_doc.replace("\n", " ")[:2000] + "...\n\n")
#             fout.write("REFERENCE SUMMARY\n")
#             fout.write(example["summary"] + "\n\n")
#             fout.write("GENERATED SUMMARY:\n")
#             fout.write(summary + "\n\n")
#             fout.write("=" * 80 + "\n\n")

#     logger.info(f"Summaries written to {output_file}")

# def main():
#     parser = argparse.ArgumentParser(description="Generate summaries with a trained model")
#     parser.add_argument("--model_path", type=str, required=True,
#                         help="Path to the trained model directory")
#     parser.add_argument("--dataset", type=str, default="alexfabbri/multi_news",
#                         help="Hugging Face dataset identifier")
#     parser.add_argument("--split", type=str, default="test",
#                         help="Dataset split to use")
#     parser.add_argument("--num_samples", type=int, default=5,
#                         help="Number of examples to summarize")
#     parser.add_argument("--output_file", type=str, default="summaries.txt",
#                         help="Path to the output file")
#     parser.add_argument("--max_input_length", type=int, default=1024,
#                         help="Maximum input tokens")
#     parser.add_argument("--max_output_length", type=int, default=256,
#                         help="Maximum summary tokens")
#     parser.add_argument("--seed", type=int, default=42,
#                         help="Random seed for sampling")
#     args = parser.parse_args()

#     # Load model & tokenizer
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model, tokenizer = load_model_and_tokenizer(
#         args.model_path,
#         device,
#         ddp=False,
#         local_rank=0
#     )

#     generate_summaries(
#         model=model,
#         tokenizer=tokenizer,
#         dataset_name=args.dataset,
#         split=args.split,
#         num_samples=args.num_samples,
#         output_file=args.output_file,
#         max_input_length=args.max_input_length,
#         max_output_length=args.max_output_length,
#         seed=args.seed
#     )

# if __name__ == "__main__":
#     main()


import argparse
import logging
import random
import torch
from datasets import load_dataset

from src.main import load_config
from src.model import load_model_and_tokenizer
from src.utils import generate_summary

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser("Generate summaries")
    p.add_argument("--config",      type=str, required=True,
                   help="Path to YAML config")
    p.add_argument("--model_path",  type=str, required=True,
                   help="Trained model directory")
    p.add_argument("--split",       type=str, default="test")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--output_file", type=str, default="summaries.txt")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model & tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, device, ddp=False, local_rank=0
    )

    # Load dataset
    ds = load_dataset(config.data.dataset_name, split=args.split)
    total = len(ds)
    indices = (random.sample(range(total), args.num_samples)
               if args.num_samples < total else list(range(total)))

    # Generate & write
    with open(args.output_file, 'w', encoding='utf-8') as fout:
        for idx in indices:
            doc = ds[idx]["document"]
            ref = ds[idx]["summary"]
            summ = generate_summary(model, tokenizer, doc, config, device)

            fout.write(f"Example {idx}\n")
            fout.write("DOCUMENT:\n"           + doc.replace("\n", " ")[:1000] + "...\n\n")
            fout.write("REFERENCE SUMMARY:\n"   + ref + "\n\n")
            fout.write("GENERATED SUMMARY:\n"   + summ + "\n\n")
            fout.write("="*80 + "\n\n")

    logger.info(f"Wrote {len(indices)} summaries to {args.output_file}")

if __name__ == "__main__":
    main()
