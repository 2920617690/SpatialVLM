#!/usr/bin/env python3
"""
Preprocess Hypersim dataset for SpatialVLM training.

Converts raw Hypersim HDF5 files to training format:
- RGB images: HDF5 → PNG
- Depth images: HDF5 → NPY
- Camera parameters: HDF5 → JSON metadata
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

try:
    import h5py
    import numpy as np
    from PIL import Image
except ImportError:
    print("Error: Required packages not found. Install with:")
    print("pip install h5py numpy pillow")
    exit(1)


def load_camera_parameters(camera_file: Path) -> Dict:
    """
    Load camera parameters from HDF5 file.
    
    Args:
        camera_file: Path to camera_keyframe_positions.hdf5
        
    Returns:
        Dictionary containing camera intrinsics and extrinsics
    """
    try:
        with h5py.File(camera_file, 'r') as f:
            # Extract camera intrinsics (focal length, principal point)
            if 'camera_intrinsics' in f:
                intrinsics = f['camera_intrinsics'][:]
            else:
                # Default intrinsics if not available
                intrinsics = np.array([[512, 0, 256], [0, 512, 256], [0, 0, 1]])
            
            # Extract camera extrinsics (position and rotation)
            if 'camera_positions' in f:
                positions = f['camera_positions'][:]
            else:
                positions = None
            
            if 'camera_rotations' in f:
                rotations = f['camera_rotations'][:]
            else:
                rotations = None
            
            return {
                'intrinsics': intrinsics.tolist(),
                'positions': positions.tolist() if positions is not None else None,
                'rotations': rotations.tolist() if rotations is not None else None
            }
    except Exception as e:
        print(f"Error loading camera parameters from {camera_file}: {e}")
        return {}


def process_frame(
    scene_id: str,
    frame_id: int,
    rgb_file: Path,
    depth_file: Path,
    camera_params: Dict,
    output_dir: Path,
    image_size: int
) -> Dict:
    """
    Process a single frame from Hypersim dataset.
    
    Args:
        scene_id: Scene identifier
        frame_id: Frame identifier
        rgb_file: Path to RGB HDF5 file
        depth_file: Path to depth HDF5 file
        camera_params: Camera parameters dictionary
        output_dir: Output directory for processed data
        image_size: Target image size (resize to square)
        
    Returns:
        Dictionary containing metadata for the processed frame
    """
    try:
        # Create output directory for this scene
        scene_output_dir = output_dir / scene_id
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load RGB image
        with h5py.File(rgb_file, 'r') as f:
            rgb_data = f['dataset'][:]
            rgb_image = Image.fromarray(rgb_data)
            rgb_image = rgb_image.resize((image_size, image_size), Image.LANCZOS)
            
            rgb_path = scene_output_dir / f"frame_{frame_id:06d}_rgb.png"
            rgb_image.save(rgb_path)
        
        # Load depth image
        with h5py.File(depth_file, 'r') as f:
            depth_data = f['dataset'][:]
            
            # Resize depth
            depth_image = Image.fromarray(depth_data)
            depth_image = depth_image.resize((image_size, image_size), Image.NEAREST)
            depth_array = np.array(depth_image)
            
            depth_path = scene_output_dir / f"frame_{frame_id:06d}_depth.npy"
            np.save(depth_path, depth_array)
        
        # Extract camera parameters for this frame
        frame_metadata = {
            'image_path': str(rgb_path.relative_to(output_dir)),
            'depth_path': str(depth_path.relative_to(output_dir)),
            'camera_intrinsics': camera_params['intrinsics'],
            'camera_extrinsics': None,
            'scene_id': scene_id,
            'frame_id': frame_id
        }
        
        # Add extrinsics if available
        if camera_params['positions'] is not None and frame_id < len(camera_params['positions']):
            frame_metadata['camera_extrinsics'] = camera_params['positions'][frame_id]
        
        # Save metadata
        metadata_path = scene_output_dir / f"frame_{frame_id:06d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(frame_metadata, f, indent=2)
        
        return frame_metadata
        
    except Exception as e:
        print(f"Error processing frame {frame_id} in scene {scene_id}: {e}")
        return None


def process_scene(
    scene_dir: Path,
    output_dir: Path,
    image_size: int
) -> List[Dict]:
    """
    Process all frames in a single scene.
    
    Args:
        scene_dir: Directory containing scene data
        output_dir: Output directory for processed data
        image_size: Target image size
        
    Returns:
        List of metadata dictionaries for processed frames
    """
    scene_id = scene_dir.name
    metadata_list = []
    
    # Find camera parameters file
    camera_file = scene_dir / "camera_keyframe_positions.hdf5"
    if not camera_file.exists():
        camera_file = scene_dir / "camera_parameters.hdf5"
    
    if not camera_file.exists():
        print(f"Warning: No camera parameters found for scene {scene_id}")
        camera_params = {}
    else:
        camera_params = load_camera_parameters(camera_file)
    
    # Find RGB and depth files
    rgb_file = scene_dir / f"{scene_id}_images.hdf5"
    depth_file = scene_dir / f"{scene_id}_depth_meters.hdf5"
    
    if not rgb_file.exists():
        print(f"Warning: No RGB file found for scene {scene_id}")
        return metadata_list
    
    if not depth_file.exists():
        print(f"Warning: No depth file found for scene {scene_id}")
        return metadata_list
    
    # Get number of frames
    try:
        with h5py.File(rgb_file, 'r') as f:
            num_frames = f['dataset'].shape[0]
    except Exception as e:
        print(f"Error reading RGB file for scene {scene_id}: {e}")
        return metadata_list
    
    # Process each frame
    for frame_id in range(num_frames):
        metadata = process_frame(
            scene_id, frame_id, rgb_file, depth_file,
            camera_params, output_dir, image_size
        )
        if metadata is not None:
            metadata_list.append(metadata)
    
    return metadata_list


def process_scene_wrapper(args: Tuple[Path, Path, int]) -> List[Dict]:
    """Wrapper function for multiprocessing."""
    scene_dir, output_dir, image_size = args
    return process_scene(scene_dir, output_dir, image_size)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Hypersim dataset for SpatialVLM training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/hypersim",
        help="Input directory containing raw Hypersim data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/hypersim_processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=384,
        help="Target image size (square)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Find all scene directories
    scene_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not scene_dirs:
        print(f"No scene directories found in {input_dir}")
        return
    
    print(f"Found {len(scene_dirs)} scenes to process")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Number of workers: {args.num_workers}")
    print()
    
    # Prepare arguments for multiprocessing
    process_args = [
        (scene_dir, output_dir, args.image_size)
        for scene_dir in scene_dirs
    ]
    
    # Process scenes in parallel
    all_metadata = []
    
    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_scene_wrapper, process_args),
                total=len(scene_dirs),
                desc="Processing scenes"
            ))
            all_metadata = [m for scene_metadata in results for m in scene_metadata]
    else:
        for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
            metadata = process_scene(scene_dir, output_dir, args.image_size)
            all_metadata.extend(metadata)
    
    # Print summary
    print("\n" + "="*60)
    print("Preprocessing Complete")
    print("="*60)
    print(f"  Scenes processed: {len(scene_dirs)}")
    print(f"  Total frames: {len(all_metadata)}")
    print(f"  Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
