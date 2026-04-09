"""
Dataset Download Script
=======================
Downloads the '60k Stack Overflow Questions with Quality Rating' dataset
from Kaggle for the Question Quality Evaluator project.

Dataset: https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate
Source:  Moradnejad/StackOverflow-Questions-Quality-Dataset (GitHub)
Paper:   "Improving Question Quality in Online Q&A Communities"

The dataset contains 60,000 Stack Overflow questions (2016-2020) classified into:
  - HQ       (High Quality):    20,000 questions - score > 30, never edited
  - LQ_EDIT  (Medium Quality):  20,000 questions - low score, edited by community
  - LQ_CLOSE (Low Quality):     20,000 questions - closed by community

Prerequisites:
  1. Install kaggle CLI: pip install kaggle
  2. Set up Kaggle API credentials:
     - Go to https://www.kaggle.com/settings -> API -> Create New Token
     - Save kaggle.json to ~/.kaggle/kaggle.json
     - chmod 600 ~/.kaggle/kaggle.json

Usage:
  python download_dataset.py
"""

import os
import subprocess
import sys


def download_kaggle_dataset():
    """Download the dataset using the Kaggle API."""
    dataset_name = "imoore/60k-stack-overflow-questions-with-quality-rate"
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", output_dir, "--unzip"],
            check=True
        )
        print("\nDataset downloaded successfully!")
        print(f"Files saved to: {output_dir}")

        # List downloaded files
        for f in os.listdir(output_dir):
            filepath = os.path.join(output_dir, f)
            size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  - {f} ({size:.1f} MB)")

    except FileNotFoundError:
        print("\nERROR: 'kaggle' command not found.")
        print("Install it with: pip install kaggle")
        print("Then set up your API key: https://www.kaggle.com/settings -> API -> Create New Token")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your Kaggle API credentials (~/.kaggle/kaggle.json)")
        print("  2. Accept the dataset's terms on Kaggle first")
        print(f"  3. Visit: https://www.kaggle.com/datasets/{dataset_name}")
        sys.exit(1)


def manual_download_instructions():
    """Print manual download instructions."""
    print("\n" + "=" * 60)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
If the automated download doesn't work, you can download manually:

1. Go to: https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate
2. Click 'Download' (you'll need a free Kaggle account)
3. Extract the ZIP file
4. Place the CSV file(s) in the 'data/' folder of this project

Expected file: data/train.csv

The CSV should have these columns:
  - Id: Question ID
  - Title: Question title
  - Body: Question body (HTML)
  - Tags: Question tags
  - CreationDate: When the question was posted
  - Y: Quality label (HQ, LQ_EDIT, LQ_CLOSE)
""")


if __name__ == "__main__":
    print("=" * 60)
    print("  Stack Overflow Question Quality Dataset Downloader")
    print("=" * 60)
    print()

    download_kaggle_dataset()
    manual_download_instructions()
