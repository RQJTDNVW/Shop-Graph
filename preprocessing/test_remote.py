from datasets import load_dataset
from huggingface_hub import hf_hub_url


REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

FILE = (
    "raw_meta_Electronics/"
    "full-00000-of-00010.parquet"
)

url = hf_hub_url(
    repo_id=REPO_ID,
    filename=FILE,
    repo_type="dataset",
)

print("Remote URL:")
print(url)

print("\nConnecting to remote dataset...")

dataset = load_dataset(
    "parquet",
    data_files=url,
    split="train",
    streaming=True,
)

print("Connection successful!")

print("\nFirst product:")

first_product = next(iter(dataset))

print(first_product)