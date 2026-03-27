#!/usr/bin/env python3
"""
Preprocess Infinigen rendered output for SpatialVLM training.

Converts Infinigen directory structure to training format with:
- Scene metadata (objects, 3D positions, room layout)
- Frame metadata (RGB, Depth, camera pose, visible objects)
- Exploration sequences (multi-view exploration paths)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("Error: Required packages not found. Install with:")
    print("pip install numpy pillow")
    exit(1)


def load_objects_json(objects_file: Path) -> List[Dict]:
    """
    Load and filter objects from Infinigen Objects JSON file.
    
    Filters out LIGHT type objects, keeping only MESH type furniture objects.
    
    Args:
        objects_file: Path to objects.json
        
    Returns:
        List of filtered object dictionaries
    """
    try:
        with open(objects_file, 'r') as f:
            data = json.load(f)
        
        # Filter objects: keep only MESH type (furniture), exclude LIGHT
        filtered_objects = []
        for obj in data.get('objects', []):
            if obj.get('type') == 'MESH' and obj.get('category') != 'LIGHT':
                filtered_objects.append(obj)
        
        return filtered_objects
    except Exception as e:
        print(f"Error loading objects from {objects_file}: {e}")
        return []


def load_camera_pose(camview_file: Path) -> Optional[Dict]:
    """
    Load camera pose from Infinigen camview npz file.
    
    Args:
        camview_file: Path to camview.npz
        
    Returns:
        Dictionary containing camera position and rotation
    """
    try:
        data = np.load(camview_file)
        
        # Extract camera position and rotation
        position = data['camera_position'] if 'camera_position' in data else None
        rotation = data['camera_rotation'] if 'camera_rotation' in data else None
        
        return {
            'position': position.tolist() if position is not None else None,
            'rotation': rotation.tolist() if rotation is not None else None
        }
    except Exception as e:
        print(f"Error loading camera pose from {camview_file}: {e}")
        return None


def get_visible_objects(
    objects: List[Dict],
    camera_pose: Dict,
    frame_dir: Path
) -> List[str]:
    """
    Determine which objects are visible in a given frame.
    
    Uses instance segmentation masks to check visibility.
    
    Args:
        objects: List of all objects in the scene
        camera_pose: Camera pose dictionary
        frame_dir: Directory containing frame data
        
    Returns:
        List of visible object IDs
    """
    visible_objects = []
    
    # Check for instance segmentation file
    seg_file = frame_dir / "InstanceSegmentation" / "instance_segmentation.png"
    
    if not seg_file.exists():
        # If no segmentation, assume all objects are potentially visible
        return [obj.get('id', '') for obj in objects]
    
    try:
        # Load segmentation mask
        seg_image = Image.open(seg_file)
        seg_array = np.array(seg_image)
        
        # Get unique instance IDs (excluding background)
        unique_ids = np.unique(seg_array)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)
        
        # Map instance IDs to object IDs
        visible_objects = [str(int(obj_id)) for obj_id in unique_ids]
        
    except Exception as e:
        print(f"Error loading segmentation from {seg_file}: {e}")
    
    return visible_objects


def process_scene(input_dir: Path, output_dir: Path, image_size: int) -> bool:
    """
    Process a single Infinigen scene.
    
    Args:
        input_dir: Input directory containing Infinigen scene
        output_dir: Output directory for processed data
        image_size: Target image size for resizing
        
    Returns:
        True if processing succeeded, False otherwise
    """
    scene_id = input_dir.name
    
    # Create output directory
    scene_output_dir = output_dir / scene_id
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load objects
    objects_file = input_dir / "frames" / "Objects" / "objects.json"
    if not objects_file.exists():
        print(f"Warning: No objects file found for scene {scene_id}")
        objects = []
    else:
        objects = load_objects_json(objects_file)
    
    # Create scene metadata
    scene_metadata = {
        'scene_id': scene_id,
        'objects': objects,
        'num_objects': len(objects),
        'room_layout': {}  # Could be populated if room info is available
    }
    
    # Save scene metadata
    scene_metadata_file = scene_output_dir / "scene_metadata.json"
    with open(scene_metadata_file, 'w') as f:
        json.dump(scene_metadata, f, indent=2)
    
    # Find all frames
    frames_dir = input_dir / "frames"
    if not frames_dir.exists():
        print(f"Warning: No frames directory found for scene {scene_id}")
        return False
    
    # Get frame directories
    frame_dirs = sorted([d for d in (frames_dir / "Image").iterdir() if d.is_dir()])
    
    if not frame_dirs:
        print(f"Warning: No frames found for scene {scene_id}")
        return False
    
    # Process each frame
    frame_metadata_list = []
    
    for frame_dir in tqdm(frame_dirs, desc=f"Processing {scene_id} frames"):
        frame_id = frame_dir.name
        
        # Paths to frame data
        rgb_file = frame_dir / "rgb.png"
        depth_file = frame_dir.parent.parent / "Depth" / frame_id / "depth.exr"
        camview_file = frame_dir.parent.parent / "camview" / f"{frame_id}.npz"
        
        if not rgb_file.exists():
            continue
        
        # Resize RGB image
        try:
            rgb_image = Image.open(rgb_file)
            rgb_image = rgb_image.resize((image_size, image_size), Image.LANCZOS)
            rgb_output_path = scene_output_dir / f"frame_{frame_id}_rgb.png"
            rgb_image.save(rgb_output_path)
        except Exception as e:
            print(f"Error processing RGB for frame {frame_id}: {e}")
            continue
        
        # Process depth
        depth_output_path = None
        if depth_file.exists():
            try:
                depth_image = Image.open(depth_file)
                depth_array = np.array(depth_image)
                depth_image = Image.fromarray(depth_array)
                depth_image = depth_image.resize((image_size, image_size), Image.NEAREST)
                depth_output_path = scene_output_dir / f"frame_{frame_id}_depth.npy"
                np.save(depth_output_path, np.array(depth_image))
            except Exception as e:
                print(f"Error processing depth for frame {frame_id}: {e}")
        
        # Load camera pose
        camera_pose = load_camera_pose(camview_file) if camview_file.exists() else {}
        
        # Get visible objects
        visible_objects = get_visible_objects(objects, camera_pose, frame_dir)
        
        # Create frame metadata
        frame_metadata = {
            'frame_id': frame_id,
            'image_path': f"frame_{frame_id}_rgb.png",
            'depth_path': f"frame_{frame_id}_depth.npy" if depth_output_path else None,
            'camera_position': camera_pose.get('position'),
            'camera_rotation': camera_pose.get('rotation'),
            'visible_objects': visible_objects,
            'num_visible_objects': len(visible_objects)
        }
        
        # Save frame metadata
        frame_metadata_file = scene_output_dir / f"frame_{frame_id}.json"
        with open(frame_metadata_file, 'w') as f:
            json.dump(frame_metadata, f, indent=2)
        
        frame_metadata_list.append(frame_metadata)
    
    # Create exploration sequence
    exploration_sequence = create_exploration_sequence(frame_metadata_list)
    
    # Save exploration sequence
    exploration_file = scene_output_dir / "exploration_sequence.json"
    with open(exploration_file, 'w') as f:
        json.dump(exploration_sequence, f, indent=2)
    
    print(f"  Processed {len(frame_metadata_list)} frames for scene {scene_id}")
    return True


def create_exploration_sequence(frame_metadata_list: List[Dict]) -> Dict:
    """
    Create an exploration sequence from frame metadata.
    
    Organizes frames into a temporal sequence with actions between frames.
    
    Args:
        frame_metadata_list: List of frame metadata dictionaries
        
    Returns:
        Dictionary containing exploration sequence
    """
    if not frame_metadata_list:
        return {}
    
    # Sort frames by ID to get temporal order
    sorted_frames = sorted(frame_metadata_list, key=lambda x: int(x['frame_id']))
    
    # Determine actions between consecutive frames
    actions = []
    for i in range(len(sorted_frames) - 1):
        curr_pos = sorted_frames[i].get('camera_position')
        next_pos = sorted_frames[i + 1].get('camera_position')
        
        if curr_pos and next_pos:
            # Calculate movement direction
            movement = np.array(next_pos) - np.array(curr_pos)
            distance = np.linalg.norm(movement)
            
            if distance < 0.1:
                action = "stay"
            else:
                # Simple action classification based on movement
                dx, dy, dz = movement
                if abs(dx) > abs(dy) and abs(dx) > abs(dz):
                    action = "move_right" if dx > 0 else "move_left"
                elif abs(dy) > abs(dx) and abs(dy) > abs(dz):
                    action = "move_forward" if dy > 0 else "move_backward"
                else:
                    action = "move_up" if dz > 0 else "move_down"
        else:
            action = "unknown"
        
        actions.append(action)
    
    exploration_sequence = {
        'sequence_type': 'exploration',
        'num_frames': len(sorted_frames),
        'frame_ids': [f['frame_id'] for f in sorted_frames],
        'actions': actions,
        'total_actions': len(actions)
    }
    
    return exploration_sequence


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Infinigen rendered output for SpatialVLM training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/infinigen",
        help="Input directory containing Infinigen scenes"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/infinigen_processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=384,
        help="Target image size (square)"
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
    print()
    
    # Process each scene
    successful_scenes = 0
    total_frames = 0
    
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        if process_scene(scene_dir, output_dir, args.image_size):
            successful_scenes += 1
            # Count frames
            frames_dir = scene_dir / "frames" / "Image"
            if frames_dir.exists():
                total_frames += len(list(frames_dir.iterdir()))
    
    # Print summary
    print("\n" + "="*60)
    print("Preprocessing Complete")
    print("="*60)
    print(f"  Scenes processed: {successful_scenes}/{len(scene_dirs)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
