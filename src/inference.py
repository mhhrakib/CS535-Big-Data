import argparse
import os
import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
import random
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)


def clean_text(text):
    """Basic text cleaning"""
    # Remove multiple spaces
    text = ' '.join(text.split())
    # Remove special HTML characters
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return text


def load_model(model_path):
    """Loads model and tokenizer from path"""
    logger.info(f"Loading model and tokenizer from {model_path}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_summary(model, tokenizer, document, max_input_length=1024, max_output_length=256, device="cpu"):
    """Generates a summary for the given document"""
    # Clean and prepare document
    document = clean_text(document)

    # Tokenize input
    inputs = tokenizer(
        document,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate summary
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_output_length,
            num_beams=4,
            early_stopping=True
        )

    # Decode summary
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return summary


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--dataset", type=str, default="alexfabbri/multi_news", help="Dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of examples to generate")
    parser.add_argument("--output_file", type=str, default="summaries.txt", help="Output file path")
    parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum input length")
    parser.add_argument("--max_output_length", type=int, default=256, help="Maximum output length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load model and tokenizer
    model, tokenizer, device = load_model(args.model_path)

    # Load dataset
    logger.info(f"Loading {args.split} split from {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)

    # Select samples
    if args.num_samples >= len(dataset):
        indices = list(range(len(dataset)))
    else:
        indices = random.sample(range(len(dataset)), args.num_samples)

    # Generate summaries
    results = []
    for idx in tqdm(indices, desc="Generating summaries"):
        example = dataset[idx]

        # Generate summary
        generated_summary = generate_summary(
            model=model,
            tokenizer=tokenizer,
            document=example["document"],
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            device=device
        )

        # Store results
        results.append({
            "index": idx,
            "document": example["document"],
            "reference_summary": example["summary"],
            "generated_summary": generated_summary
        })

    # Write results to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i, result in enumerate(results):
            f.write(f"Example {i + 1} (Dataset index: {result['index']})\n")
            f.write("=" * 80 + "\n")
            f.write("DOCUMENT:\n")
            f.write(result['document'][:1000] + "...\n\n")
            f.write("REFERENCE SUMMARY:\n")
            f.write(result['reference_summary'] + "\n\n")
            f.write("GENERATED SUMMARY:\n")
            f.write(result['generated_summary'] + "\n\n")
            f.write("=" * 80 + "\n\n")

    logger.info(f"Generated summaries saved to {args.output_file}")


if __name__ == "__main__":
    main()