"""Spatial exploration dataset for sequential spatial reasoning."""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class SpatialExplorationDataset(Dataset):
    """Sequential spatial exploration dataset.
    
    Each sample contains:
    - Scene configuration
    - Exploration trajectory (multi-step observation sequence)
    - Ground truth scene graph
    """
    
    def __init__(
        self,
        data_path: str,
        max_trajectory_length: int = 50,
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Any] = None
    ):
        """Initialize spatial exploration dataset.
        
        Args:
            data_path: Path to dataset directory
            max_trajectory_length: Maximum length of exploration trajectory
            image_size: Size of observation images
            transform: Optional image transformations
        """
        self.data_path = data_path
        self.max_trajectory_length = max_trajectory_length
        self.image_size = image_size
        self.transform = transform
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from data directory.
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        # In practice, load from actual data files
        # For now, create placeholder structure
        num_samples = 100
        
        for i in range(num_samples):
            sample = {
                "scene_id": f"scene_{i}",
                "scene_config": {
                    "room_type": "living_room",
                    "num_objects": np.random.randint(5, 15),
                    "area_size": (10.0, 10.0, 3.0)
                },
                "trajectory": self._generate_trajectory(i),
                "ground_truth_scene_graph": self._generate_scene_graph(i)
            }
            samples.append(sample)
        
        return samples
    
    def _generate_trajectory(self, sample_id: int) -> List[Dict[str, Any]]:
        """Generate exploration trajectory for a sample.
        
        Args:
            sample_id: Sample identifier
        
        Returns:
            List of trajectory steps
        """
        trajectory_length = np.random.randint(10, self.max_trajectory_length)
        trajectory = []
        
        for step in range(trajectory_length):
            step_data = {
                "step": step,
                "position": np.random.randn(3).tolist(),
                "facing": np.random.randn(3).tolist(),
                "action": np.random.choice(["move_forward", "turn_left", "turn_right", "stop"]),
                "observation": np.random.randn(3, *self.image_size).astype(np.float32),
                "camera_pose": np.random.randn(4, 4).astype(np.float32)
            }
            trajectory.append(step_data)
        
        return trajectory
    
    def _generate_scene_graph(self, sample_id: int) -> Dict[str, Any]:
        """Generate ground truth scene graph.
        
        Args:
            sample_id: Sample identifier
        
        Returns:
            Scene graph dictionary
        """
        num_objects = np.random.randint(5, 15)
        
        nodes = []
        for i in range(num_objects):
            node = {
                "object_id": f"obj_{sample_id}_{i}",
                "category": np.random.choice(["chair", "table", "sofa", "lamp", "plant", "tv"]),
                "position": np.random.randn(3).tolist(),
                "orientation": np.random.randn(3).tolist(),
                "size": np.random.rand(3).tolist()
            }
            nodes.append(node)
        
        edges = []
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                if np.random.random() < 0.3:
                    edge = {
                        "source": f"obj_{sample_id}_{i}",
                        "target": f"obj_{sample_id}_{j}",
                        "relation": np.random.choice(["near", "above", "below", "left_of", "right_of", "behind"]),
                        "distance": np.random.rand()
                    }
                    edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Sample dictionary containing:
            - scene_config: Scene configuration
            - trajectory: Exploration trajectory
            - ground_truth_scene_graph: Ground truth scene graph
        """
        sample = self.samples[idx]
        
        # Convert trajectory to tensors
        trajectory = sample["trajectory"]
        trajectory_length = len(trajectory)
        
        images = torch.stack([
            torch.from_numpy(step["observation"])
            for step in trajectory
        ])
        
        positions = torch.tensor([
            step["position"] for step in trajectory
        ], dtype=torch.float32)
        
        facings = torch.tensor([
            step["facing"] for step in trajectory
        ], dtype=torch.float32)
        
        actions = torch.tensor([
            self._action_to_idx(step["action"])
            for step in trajectory
        ], dtype=torch.long)
        
        camera_poses = torch.stack([
            torch.from_numpy(step["camera_pose"])
            for step in trajectory
        ])
        
        # Pad trajectory to max length
        padded_images = torch.zeros(
            self.max_trajectory_length, *images.shape[1:],
            dtype=images.dtype
        )
        padded_images[:trajectory_length] = images
        
        padded_positions = torch.zeros(
            self.max_trajectory_length, 3,
            dtype=positions.dtype
        )
        padded_positions[:trajectory_length] = positions
        
        padded_facings = torch.zeros(
            self.max_trajectory_length, 3,
            dtype=facings.dtype
        )
        padded_facings[:trajectory_length] = facings
        
        padded_actions = torch.zeros(
            self.max_trajectory_length,
            dtype=actions.dtype
        )
        padded_actions[:trajectory_length] = actions
        
        padded_camera_poses = torch.zeros(
            self.max_trajectory_length, *camera_poses.shape[1:],
            dtype=camera_poses.dtype
        )
        padded_camera_poses[:trajectory_length] = camera_poses
        
        # Convert scene graph to tensors
        scene_graph = sample["ground_truth_scene_graph"]
        gt_nodes = torch.tensor([
            node["position"] + node["orientation"] + node["size"]
            for node in scene_graph["nodes"]
        ], dtype=torch.float32)
        
        gt_edges = torch.tensor([
            [self._object_to_idx(edge["source"], scene_graph["nodes"]),
             self._object_to_idx(edge["target"], scene_graph["nodes"]),
             self._relation_to_idx(edge["relation"]),
             edge["distance"]]
            for edge in scene_graph["edges"]
        ], dtype=torch.float32)
        
        return {
            "scene_id": sample["scene_id"],
            "scene_config": sample["scene_config"],
            "images": padded_images,
            "positions": padded_positions,
            "facings": padded_facings,
            "actions": padded_actions,
            "camera_poses": padded_camera_poses,
            "trajectory_length": trajectory_length,
            "ground_truth_nodes": gt_nodes,
            "ground_truth_edges": gt_edges,
            "ground_truth_scene_graph": scene_graph
        }
    
    def _action_to_idx(self, action: str) -> int:
        """Convert action string to index."""
        action_map = {
            "move_forward": 0,
            "turn_left": 1,
            "turn_right": 2,
            "stop": 3
        }
        return action_map.get(action, 0)
    
    def _object_to_idx(self, object_id: str, nodes: List[Dict]) -> int:
        """Convert object ID to index."""
        for idx, node in enumerate(nodes):
            if node["object_id"] == object_id:
                return idx
        return 0
    
    def _relation_to_idx(self, relation: str) -> int:
        """Convert relation string to index."""
        relation_map = {
            "near": 0,
            "above": 1,
            "below": 2,
            "left_of": 3,
            "right_of": 4,
            "behind": 5
        }
        return relation_map.get(relation, 0)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for variable-length sequences.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched dictionary with padded sequences
    """
    scene_ids = [item["scene_id"] for item in batch]
    scene_configs = [item["scene_config"] for item in batch]
    
    images = torch.stack([item["images"] for item in batch])
    positions = torch.stack([item["positions"] for item in batch])
    facings = torch.stack([item["facings"] for item in batch])
    actions = torch.stack([item["actions"] for item in batch])
    camera_poses = torch.stack([item["camera_poses"] for item in batch])
    trajectory_lengths = torch.tensor([item["trajectory_length"] for item in batch])
    
    # Handle variable-length ground truth graphs
    max_nodes = max(item["ground_truth_nodes"].shape[0] for item in batch)
    max_edges = max(item["ground_truth_edges"].shape[0] for item in batch)
    
    gt_nodes = torch.zeros(len(batch), max_nodes, 9, dtype=torch.float32)
    gt_edges = torch.zeros(len(batch), max_edges, 4, dtype=torch.float32)
    
    for idx, item in enumerate(batch):
        num_nodes = item["ground_truth_nodes"].shape[0]
        num_edges = item["ground_truth_edges"].shape[0]
        gt_nodes[idx, :num_nodes] = item["ground_truth_nodes"]
        gt_edges[idx, :num_edges] = item["ground_truth_edges"]
    
    ground_truth_scene_graphs = [item["ground_truth_scene_graph"] for item in batch]
    
    return {
        "scene_ids": scene_ids,
        "scene_configs": scene_configs,
        "images": images,
        "positions": positions,
        "facings": facings,
        "actions": actions,
        "camera_poses": camera_poses,
        "trajectory_lengths": trajectory_lengths,
        "ground_truth_nodes": gt_nodes,
        "ground_truth_edges": gt_edges,
        "ground_truth_scene_graphs": ground_truth_scene_graphs
    }
