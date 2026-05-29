"""Download the full FinReflectKG dataset using the HuggingFace datasets library.

Downloads all 743 companies / 17.5M triplets and saves locally as parquet.
We subset to top-20 companies later in the pipeline, not at download time.

Usage:
    uv run python tools/download_data.py
"""

import os
import ssl

# Disable SSL verification before importing anything that makes network calls.
# Required on corporate networks with SSL-intercepting proxies.
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Also patch ssl globally for urllib-based downloads
ssl._create_default_https_context = ssl._create_unverified_context

from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv


def download_full_dataset() -> None:
    """Download the full FinReflectKG dataset and save locally as parquet."""
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    data_dir = Path(__file__).parent.parent / "data"
    kg_dir = data_dir / "finreflectkg"
    kg_dir.mkdir(parents=True, exist_ok=True)

    output_file = kg_dir / "finreflectkg_full.parquet"

    if output_file.exists():
        print(f"Dataset already exists at {output_file}. Delete it to re-download.")
        return

    print("Downloading full FinReflectKG from HuggingFace...")
    print("This is ~1.67 GB / 17.5M triplets. This will take a while.")

    ds = load_dataset(
        "domyn/FinReflectKG",
        split="train",
        token=token,
    )

    print(f"Downloaded {len(ds):,} triplets.")

    # Save as parquet
    print(f"Saving to {output_file}...")
    ds.to_parquet(str(output_file))

    print(f"Saved. File size: {output_file.stat().st_size / (1024*1024):.1f} MB")

    # Print basic stats
    import pandas as pd

    df = pd.read_parquet(output_file)
    print("\n--- Dataset Stats ---")
    print(f"Total triplets: {len(df):,}")
    print(f"Companies (tickers): {df['ticker'].nunique()}")
    print(f"Unique entities: {df['entity'].nunique():,}")
    print(f"Unique targets: {df['target'].nunique():,}")
    print(f"Unique relationships: {df['relationship'].nunique()}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Entity types: {sorted(df['entity_type'].unique())}")


if __name__ == "__main__":
    download_full_dataset()
