#!/usr/bin/env python3
"""
Download Theory of Space benchmark dataset from HuggingFace.

Downloads the MLL-Lab/tos-data dataset which contains spatial reasoning tasks.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is required. Install with: pip install huggingface_hub")
    exit(1)


TOS_REPO_ID = "MLL-Lab/tos-data"
TOS_SPLITS = ["train", "val", "test"]


def print_dataset_statistics(data_dir: Path, split: str) -> None:
    """
    Print statistics about the downloaded dataset.
    
    Args:
        data_dir: Directory containing the downloaded data
        split: Dataset split to analyze
    """
    split_dir = data_dir / split
    
    if not split_dir.exists():
        print(f"Warning: Split directory {split_dir} does not exist")
        return
    
    # Count JSON files
    json_files = list(split_dir.glob("**/*.json"))
    
    if not json_files:
        print(f"  No JSON files found in {split}")
        return
    
    # Analyze data
    total_samples = 0
    qa_types: Dict[str, int] = {}
    difficulties: Dict[str, int] = {}
    
    for json_file in tqdm(json_files, desc=f"Analyzing {split} split"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    samples = data
                else:
                    samples = [data]
                
                for sample in samples:
                    total_samples += 1
                    
                    if 'qa_type' in sample:
                        qa_type = sample['qa_type']
                        qa_types[qa_type] = qa_types.get(qa_type, 0) + 1
                    
                    if 'difficulty' in sample:
                        difficulty = sample['difficulty']
                        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
                        
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    # Print statistics
    print(f"\n  {split.upper()} Split Statistics:")
    print(f"    Total samples: {total_samples}")
    print(f"    JSON files: {len(json_files)}")
    
    if qa_types:
        print(f"    QA Types:")
        for qa_type, count in sorted(qa_types.items()):
            print(f"      {qa_type}: {count}")
    
    if difficulties:
        print(f"    Difficulties:")
        for difficulty, count in sorted(difficulties.items()):
            print(f"      {difficulty}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Theory of Space benchmark dataset from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/tos",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=TOS_SPLITS + ["all"],
        help="Dataset split to download (train/val/test/all)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=TOS_REPO_ID,
        help="HuggingFace repository ID"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Theory of Space dataset from {args.repo_id}")
    print(f"Output directory: {output_dir}")
    print(f"Split: {args.split}")
    print()
    
    # Determine which splits to download
    if args.split == "all":
        splits_to_download = TOS_SPLITS
    else:
        splits_to_download = [args.split]
    
    # Download from HuggingFace
    try:
        print("Downloading dataset...")
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✓ Download complete")
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return
    
    # Print statistics for each split
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    for split in splits_to_download:
        print_dataset_statistics(output_dir, split)
    
    print("\n" + "="*60)
    print(f"Dataset downloaded to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
