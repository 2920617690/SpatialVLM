#!/usr/bin/env python3
"""
Download Apple Hypersim dataset from official AWS S3 bucket.

Supports selective downloading of scenes and data types (RGB, Depth, Semantic).
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Set
from tqdm import tqdm


HYPERSIM_BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/"
HYPERSIM_SCENES = [
    "ai_001_001", "ai_001_002", "ai_001_003", "ai_001_004", "ai_001_005",
    "ai_002_001", "ai_002_002", "ai_002_003", "ai_002_004", "ai_002_005",
    "ai_003_001", "ai_003_002", "ai_003_003", "ai_003_004", "ai_003_005",
    "ai_004_001", "ai_004_002", "ai_004_003", "ai_004_004", "ai_004_005",
    "ai_005_001", "ai_005_002", "ai_005_003", "ai_005_004", "ai_005_005",
    "ai_006_001", "ai_006_002", "ai_006_003", "ai_006_004", "ai_006_005",
    "ai_007_001", "ai_007_002", "ai_007_003", "ai_007_004", "ai_007_005",
    "ai_008_001", "ai_008_002", "ai_008_003", "ai_008_004", "ai_008_005",
    "ai_009_001", "ai_009_002", "ai_009_003", "ai_009_004", "ai_009_005",
    "ai_010_001", "ai_010_002", "ai_010_003", "ai_010_004", "ai_010_005",
]


def download_file_with_resume(url: str, output_path: Path) -> bool:
    """
    Download a file with resume capability using wget.
    
    Args:
        url: The URL to download from
        output_path: The local path to save the file
        
    Returns:
        True if download succeeded, False otherwise
    """
    try:
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use wget with resume capability (-c)
        cmd = [
            "wget",
            "-c",  # Continue/resume
            "-O", str(output_path),
            "--show-progress",
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
        
    Returns:
        True if extraction succeeded, False otherwise
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(zip_path), str(extract_to))
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def download_scene_data(
    scene_id: str,
    output_dir: Path,
    data_types: Set[str],
    base_url: str = HYPERSIM_BASE_URL
) -> bool:
    """
    Download data for a single scene.
    
    Args:
        scene_id: The scene identifier (e.g., "ai_001_001")
        output_dir: Base output directory
        data_types: Set of data types to download (rgb, depth, semantic)
        base_url: Base URL for Hypersim data
        
    Returns:
        True if all downloads succeeded, False otherwise
    """
    scene_dir = output_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # Download camera parameters
    camera_url = f"{base_url}{scene_id}.zip"
    camera_zip = scene_dir / f"{scene_id}.zip"
    
    print(f"  Downloading camera parameters for {scene_id}...")
    if download_file_with_resume(camera_url, camera_zip):
        print(f"  Extracting camera parameters...")
        if extract_zip(camera_zip, scene_dir):
            camera_zip.unlink()  # Remove zip after extraction
        else:
            success = False
    else:
        success = False
    
    # Download images based on data types
    for frame_type in data_types:
        if frame_type == "rgb":
            zip_name = f"{scene_id}_images.hdf5.zip"
        elif frame_type == "depth":
            zip_name = f"{scene_id}_depth_meters.hdf5.zip"
        elif frame_type == "semantic":
            zip_name = f"{scene_id}_semantic_segmentation.hdf5.zip"
        else:
            print(f"  Unknown data type: {frame_type}")
            continue
        
        image_url = f"{base_url}{zip_name}"
        image_zip = scene_dir / zip_name
        
        print(f"  Downloading {frame_type} data for {scene_id}...")
        if download_file_with_resume(image_url, image_zip):
            print(f"  Extracting {frame_type} data...")
            if extract_zip(image_zip, scene_dir):
                image_zip.unlink()
            else:
                success = False
        else:
            success = False
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download Apple Hypersim dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/hypersim",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=5,
        help="Number of scenes to download (max 50)"
    )
    parser.add_argument(
        "--data_types",
        type=str,
        nargs="+",
        default=["rgb", "depth"],
        choices=["rgb", "depth", "semantic"],
        help="Data types to download"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    num_scenes = min(args.num_scenes, len(HYPERSIM_SCENES))
    data_types = set(args.data_types)
    
    if num_scenes <= 0:
        print("Error: num_scenes must be greater than 0")
        return
    
    if not data_types:
        print("Error: at least one data type must be specified")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {num_scenes} scenes to {output_dir}")
    print(f"Data types: {', '.join(sorted(data_types))}")
    print()
    
    # Download scenes
    scenes_to_download = HYPERSIM_SCENES[:num_scenes]
    successful_scenes = 0
    
    for scene_id in tqdm(scenes_to_download, desc="Downloading scenes"):
        print(f"\nProcessing scene: {scene_id}")
        if download_scene_data(scene_id, output_dir, data_types):
            successful_scenes += 1
            print(f"  ✓ Successfully downloaded {scene_id}")
        else:
            print(f"  ✗ Failed to download {scene_id}")
    
    # Print summary
    print("\n" + "="*60)
    print(f"Download complete!")
    print(f"  Scenes requested: {num_scenes}")
    print(f"  Scenes successful: {successful_scenes}")
    print(f"  Scenes failed: {num_scenes - successful_scenes}")
    print(f"  Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
