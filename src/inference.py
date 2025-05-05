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

            logger.info(f"doc: {doc[:50]}\n, sum type: {type(summ)}, sum: {summ}")

            doc_snip = doc.replace("\n", " ")[:800]

            fout.write(f"Example {idx}\n")
            fout.write(f"DOCUMENT:\n{doc_snip}...\n\n")
            fout.write(f"REFERENCE SUMMARY (chars: {len(ref)}):\n{ref}\n\n")
            fout.write(f"GENERATED SUMMARY (chars: {len(summ)}):\n{summ}\n\n")
            fout.write(f"{'=' * 80}\n\n")

    logger.info(f"Wrote {len(indices)} summaries to {args.output_file}")

if __name__ == "__main__":
    main()
