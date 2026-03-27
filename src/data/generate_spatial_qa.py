#!/usr/bin/env python3
"""
Generate spatial reasoning QA pairs from scene ground truth.

Automatically generates QA pairs for Theory of Space tasks:
- Pairwise Direction
- Perspective Taking
- Mental Rotation
- Allocentric Mapping
- Distance Estimation
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    exit(1)


# Direction discretization (8 directions)
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Distance discretization (6 levels)
DISTANCE_LEVELS = ["same", "near", "mid", "slightly_far", "far", "very_far"]

# Distance thresholds (can be adjusted based on scene scale)
DISTANCE_THRESHOLDS = [0.1, 0.5, 1.0, 2.0, 5.0]


def angle_to_direction(angle: float) -> str:
    """
    Convert angle (in radians) to discrete direction.
    
    Args:
        angle: Angle in radians, 0 = East, increasing counterclockwise
        
    Returns:
        Direction string (N, NE, E, SE, S, SW, W, NW)
    """
    # Normalize angle to [0, 2π)
    angle = angle % (2 * np.pi)
    
    # Convert to degrees and adjust so 0 = North
    degrees = np.degrees(angle)
    degrees = (degrees + 90) % 360
    
    # Discretize into 8 directions
    index = int((degrees + 22.5) / 45) % 8
    
    return DIRECTIONS[index]


def distance_to_level(distance: float) -> str:
    """
    Discretize continuous distance into discrete levels.
    
    Args:
        distance: Euclidean distance
        
    Returns:
        Distance level string
    """
    if distance < DISTANCE_THRESHOLDS[0]:
        return DISTANCE_LEVELS[0]
    elif distance < DISTANCE_THRESHOLDS[1]:
        return DISTANCE_LEVELS[1]
    elif distance < DISTANCE_THRESHOLDS[2]:
        return DISTANCE_LEVELS[2]
    elif distance < DISTANCE_THRESHOLDS[3]:
        return DISTANCE_LEVELS[3]
    elif distance < DISTANCE_THRESHOLDS[4]:
        return DISTANCE_LEVELS[4]
    else:
        return DISTANCE_LEVELS[5]


def compute_pairwise_direction(
    obj_a: Dict,
    obj_b: Dict
) -> str:
    """
    Compute allocentric direction from object A to object B.
    
    Args:
        obj_a: Source object dictionary with 'position' field
        obj_b: Target object dictionary with 'position' field
        
    Returns:
        Direction string (N, NE, E, SE, S, SW, W, NW)
    """
    pos_a = np.array(obj_a.get('position', [0, 0, 0]))
    pos_b = np.array(obj_b.get('position', [0, 0, 0]))
    
    # Compute direction vector
    direction = pos_b - pos_a
    
    # Compute angle (0 = East)
    angle = np.arctan2(direction[1], direction[0])
    
    return angle_to_direction(angle)


def compute_perspective_taking(
    obj_a: Dict,
    obj_b: Dict
) -> str:
    """
    Compute direction from object A's perspective to object B.
    
    Performs coordinate transformation to get egocentric direction.
    
    Args:
        obj_a: Observer object with 'position' and 'orientation'
        obj_b: Target object with 'position'
        
    Returns:
        Direction string from A's perspective
    """
    pos_a = np.array(obj_a.get('position', [0, 0, 0]))
    pos_b = np.array(obj_b.get('position', [0, 0, 0]))
    
    # Get A's orientation (rotation)
    orientation = obj_a.get('orientation', 0)
    
    # Compute direction vector
    direction = pos_b - pos_a
    
    # Rotate by negative orientation to get egocentric direction
    angle = np.arctan2(direction[1], direction[0])
    egocentric_angle = angle - orientation
    
    return angle_to_direction(egocentric_angle)


def compute_euclidean_distance(
    obj_a: Dict,
    obj_b: Dict
) -> float:
    """
    Compute Euclidean distance between two objects.
    
    Args:
        obj_a: First object with 'position'
        obj_b: Second object with 'position'
        
    Returns:
        Euclidean distance
    """
    pos_a = np.array(obj_a.get('position', [0, 0, 0]))
    pos_b = np.array(obj_b.get('position', [0, 0, 0]))
    
    return np.linalg.norm(pos_a - pos_b)


def generate_pairwise_direction_qa(
    objects: List[Dict],
    scene_id: str
) -> List[Dict]:
    """
    Generate pairwise direction QA pairs.
    
    Args:
        objects: List of objects in the scene
        scene_id: Scene identifier
        
    Returns:
        List of QA dictionaries
    """
    qa_pairs = []
    
    if len(objects) < 2:
        return qa_pairs
    
    # Sample object pairs
    for _ in range(5):
        obj_a, obj_b = random.sample(objects, 2)
        
        direction = compute_pairwise_direction(obj_a, obj_b)
        
        qa = {
            'question': f"What is the spatial direction from {obj_a.get('name', 'object A')} to {obj_b.get('name', 'object B')}?",
            'answer': direction,
            'qa_type': 'pairwise_direction',
            'scene_id': scene_id,
            'difficulty': 'easy',
            'objects': [obj_a.get('id'), obj_b.get('id')]
        }
        
        qa_pairs.append(qa)
    
    return qa_pairs


def generate_perspective_taking_qa(
    objects: List[Dict],
    scene_id: str
) -> List[Dict]:
    """
    Generate perspective taking QA pairs.
    
    Args:
        objects: List of objects in the scene
        scene_id: Scene identifier
        
    Returns:
        List of QA dictionaries
    """
    qa_pairs = []
    
    if len(objects) < 2:
        return qa_pairs
    
    # Sample object pairs
    for _ in range(5):
        obj_a, obj_b = random.sample(objects, 2)
        
        direction = compute_perspective_taking(obj_a, obj_b)
        
        qa = {
            'question': f"From {obj_a.get('name', 'object A')}'s perspective, where is {obj_b.get('name', 'object B')}?",
            'answer': direction,
            'qa_type': 'perspective_taking',
            'scene_id': scene_id,
            'difficulty': 'medium',
            'objects': [obj_a.get('id'), obj_b.get('id')]
        }
        
        qa_pairs.append(qa)
    
    return qa_pairs


def generate_mental_rotation_qa(
    objects: List[Dict],
    scene_id: str
) -> List[Dict]:
    """
    Generate mental rotation QA pairs.
    
    Simulates clockwise rotation and lists visible objects.
    
    Args:
        objects: List of objects in the scene
        scene_id: Scene identifier
        
    Returns:
        List of QA dictionaries
    """
    qa_pairs = []
    
    if len(objects) < 3:
        return qa_pairs
    
    # Sort objects by angle around center
    center = np.mean([np.array(obj.get('position', [0, 0, 0])) for obj in objects], axis=0)
    
    angles = []
    for obj in objects:
        pos = np.array(obj.get('position', [0, 0, 0]))
        direction = pos - center
        angle = np.arctan2(direction[1], direction[0])
        angles.append((angle, obj))
    
    # Sort by angle
    angles.sort(key=lambda x: x[0])
    
    # Generate rotation sequence
    for _ in range(3):
        start_idx = random.randint(0, len(angles) - 1)
        num_objects = min(random.randint(3, 5), len(angles))
        
        sequence = []
        for i in range(num_objects):
            idx = (start_idx + i) % len(angles)
            sequence.append(angles[idx][1].get('name', f'object {idx}'))
        
        qa = {
            'question': f"If you rotate clockwise, what objects do you see in order?",
            'answer': ', '.join(sequence),
            'qa_type': 'mental_rotation',
            'scene_id': scene_id,
            'difficulty': 'hard',
            'objects': [obj.get('id') for _, obj in angles[start_idx:start_idx + num_objects]]
        }
        
        qa_pairs.append(qa)
    
    return qa_pairs


def generate_allocentric_mapping_qa(
    objects: List[Dict],
    scene_id: str
) -> List[Dict]:
    """
    Generate allocentric mapping QA pairs.
    
    Returns 2D coordinates of all objects.
    
    Args:
        objects: List of objects in the scene
        scene_id: Scene identifier
        
    Returns:
        List of QA dictionaries
    """
    qa_pairs = []
    
    if not objects:
        return qa_pairs
    
    # Create coordinate mapping
    coordinates = {}
    for obj in objects:
        pos = obj.get('position', [0, 0, 0])
        coordinates[obj.get('name', f'object {obj.get("id")}')] = (pos[0], pos[1])
    
    # Format as readable string
    coord_str = ', '.join([f"{name}: ({x:.2f}, {y:.2f})" for name, (x, y) in coordinates.items()])
    
    qa = {
        'question': "What are the 2D coordinates of all objects?",
        'answer': coord_str,
        'qa_type': 'allocentric_mapping',
        'scene_id': scene_id,
        'difficulty': 'medium',
        'objects': [obj.get('id') for obj in objects]
    }
    
    qa_pairs.append(qa)
    
    return qa_pairs


def generate_distance_estimation_qa(
    objects: List[Dict],
    scene_id: str
) -> List[Dict]:
    """
    Generate distance estimation QA pairs.
    
    Args:
        objects: List of objects in the scene
        scene_id: Scene identifier
        
    Returns:
        List of QA dictionaries
    """
    qa_pairs = []
    
    if len(objects) < 2:
        return qa_pairs
    
    # Sample object pairs
    for _ in range(5):
        obj_a, obj_b = random.sample(objects, 2)
        
        distance = compute_euclidean_distance(obj_a, obj_b)
        distance_level = distance_to_level(distance)
        
        qa = {
            'question': f"How far is {obj_a.get('name', 'object A')} from {obj_b.get('name', 'object B')}?",
            'answer': distance_level,
            'qa_type': 'distance_estimation',
            'scene_id': scene_id,
            'difficulty': 'easy',
            'objects': [obj_a.get('id'), obj_b.get('id')]
        }
        
        qa_pairs.append(qa)
    
    return qa_pairs


def generate_qa_for_scene(
    scene_metadata: Dict,
    num_qa_per_scene: int,
    seed: int
) -> List[Dict]:
    """
    Generate QA pairs for a single scene.
    
    Args:
        scene_metadata: Scene metadata dictionary
        num_qa_per_scene: Number of QA pairs to generate per scene
        seed: Random seed for reproducibility
        
    Returns:
        List of QA dictionaries
    """
    random.seed(seed)
    np.random.seed(seed)
    
    objects = scene_metadata.get('objects', [])
    scene_id = scene_metadata.get('scene_id', 'unknown')
    
    if not objects:
        return []
    
    all_qa = []
    
    # Generate different types of QA
    all_qa.extend(generate_pairwise_direction_qa(objects, scene_id))
    all_qa.extend(generate_perspective_taking_qa(objects, scene_id))
    all_qa.extend(generate_mental_rotation_qa(objects, scene_id))
    all_qa.extend(generate_allocentric_mapping_qa(objects, scene_id))
    all_qa.extend(generate_distance_estimation_qa(objects, scene_id))
    
    # Sample to get desired number
    if len(all_qa) > num_qa_per_scene:
        all_qa = random.sample(all_qa, num_qa_per_scene)
    
    return all_qa


def main():
    parser = argparse.ArgumentParser(
        description="Generate spatial reasoning QA pairs from scene ground truth"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/processed",
        help="Input directory containing scene_metadata.json files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/spatial_qa",
        help="Output directory for generated QA pairs"
    )
    parser.add_argument(
        "--num_qa_per_scene",
        type=int,
        default=25,
        help="Number of QA pairs to generate per scene"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Find all scene metadata files
    scene_metadata_files = list(input_dir.glob("**/scene_metadata.json"))
    
    if not scene_metadata_files:
        print(f"No scene_metadata.json files found in {input_dir}")
        return
    
    print(f"Found {len(scene_metadata_files)} scenes")
    print(f"Generating {args.num_qa_per_scene} QA pairs per scene")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate QA for each scene
    all_qa = []
    qa_type_counts = {}
    
    for metadata_file in tqdm(scene_metadata_files, desc="Generating QA"):
        try:
            with open(metadata_file, 'r') as f:
                scene_metadata = json.load(f)
            
            scene_qa = generate_qa_for_scene(
                scene_metadata,
                args.num_qa_per_scene,
                args.seed
            )
            
            all_qa.extend(scene_qa)
            
            # Count QA types
            for qa in scene_qa:
                qa_type = qa.get('qa_type', 'unknown')
                qa_type_counts[qa_type] = qa_type_counts.get(qa_type, 0) + 1
                
        except Exception as e:
            print(f"Error processing {metadata_file}: {e}")
    
    # Save all QA pairs
    output_file = output_dir / "spatial_qa.json"
    with open(output_file, 'w') as f:
        json.dump(all_qa, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("QA Generation Complete")
    print("="*60)
    print(f"  Total scenes: {len(scene_metadata_files)}")
    print(f"  Total QA pairs: {len(all_qa)}")
    print(f"  Output file: {output_file}")
    print("\n  QA Type Distribution:")
    for qa_type, count in sorted(qa_type_counts.items()):
        print(f"    {qa_type}: {count}")
    print("="*60)


if __name__ == "__main__":
    main()
