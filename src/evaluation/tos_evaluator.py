"""Theory of Space (ToS) benchmark evaluator."""

import torch
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from ..model.spatial_vlm import SpatialVLM
from .metrics import (
    positional_accuracy,
    directional_accuracy,
    facing_accuracy,
    belief_inertia_score,
    information_gain,
    coordinate_similarity
)


class ToSEvaluator:
    """Theory of Space benchmark evaluator.
    
    Evaluates spatial reasoning capabilities including:
    - Route planning tasks
    - Survey tasks
    - Belief quality
    - Belief revision (False Belief paradigm)
    """
    
    def __init__(
        self,
        model: SpatialVLM,
        device: str = "cuda",
        tolerance_distance: float = 1.0,
        tolerance_angle: float = 30.0
    ):
        """Initialize ToS evaluator.
        
        Args:
            model: Spatial VLM model to evaluate
            device: Device to run evaluation on
            tolerance_distance: Distance tolerance for position accuracy (meters)
            tolerance_angle: Angle tolerance for direction accuracy (degrees)
        """
        self.model = model
        self.device = device
        self.tolerance_distance = tolerance_distance
        self.tolerance_angle = tolerance_angle
        
        self.model.to(device)
        self.model.eval()
    
    def evaluate_route_tasks(
        self,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate route planning tasks.
        
        Args:
            test_data: List of route planning test cases
        
        Returns:
            Dictionary of route task metrics
        """
        total_tasks = len(test_data)
        successful_tasks = 0
        total_path_length_error = 0.0
        total_time_steps = 0
        
        for task in test_data:
            start_position = task["start_position"]
            goal_position = task["goal_position"]
            obstacles = task.get("obstacles", [])
            
            # Run route planning
            planned_path = self._plan_route(
                start_position,
                goal_position,
                obstacles
            )
            
            # Evaluate success
            success = self._evaluate_route_success(
                planned_path,
                goal_position
            )
            
            if success:
                successful_tasks += 1
            
            # Compute path length error
            optimal_length = self._compute_optimal_path_length(
                start_position,
                goal_position,
                obstacles
            )
            planned_length = self._compute_path_length(planned_path)
            path_length_error = abs(planned_length - optimal_length) / optimal_length
            total_path_length_error += path_length_error
            
            total_time_steps += len(planned_path)
        
        metrics = {
            "route_success_rate": successful_tasks / total_tasks,
            "average_path_length_error": total_path_length_error / total_tasks,
            "average_time_steps": total_time_steps / total_tasks
        }
        
        return metrics
    
    def evaluate_survey_tasks(
        self,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate survey tasks (spatial mapping).
        
        Args:
            test_data: List of survey test cases
        
        Returns:
            Dictionary of survey task metrics
        """
        total_tasks = len(test_data)
        total_coverage = 0.0
        total_map_accuracy = 0.0
        total_objects_found = 0
        total_objects_gt = 0
        
        for task in test_data:
            scene_config = task["scene_config"]
            exploration_trajectory = task["exploration_trajectory"]
            ground_truth_map = task["ground_truth_map"]
            ground_truth_objects = task["ground_truth_objects"]
            
            # Run exploration and mapping
            belief_map = self._run_exploration(
                scene_config,
                exploration_trajectory
            )
            
            # Evaluate coverage
            coverage = self._compute_coverage(belief_map, ground_truth_map)
            total_coverage += coverage
            
            # Evaluate map accuracy
            map_accuracy = self._compute_map_accuracy(belief_map, ground_truth_map)
            total_map_accuracy += map_accuracy
            
            # Evaluate object detection
            detected_objects = self._detect_objects(belief_map)
            objects_found = self._match_objects(detected_objects, ground_truth_objects)
            total_objects_found += objects_found
            total_objects_gt += len(ground_truth_objects)
        
        metrics = {
            "survey_coverage": total_coverage / total_tasks,
            "survey_map_accuracy": total_map_accuracy / total_tasks,
            "object_detection_recall": total_objects_found / total_objects_gt
        }
        
        return metrics
    
    def evaluate_belief_quality(
        self,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate cognitive map quality.
        
        Args:
            test_data: List of belief quality test cases
        
        Returns:
            Dictionary of belief quality metrics
        """
        total_cases = len(test_data)
        total_positional_acc = 0.0
        total_directional_acc = 0.0
        total_facing_acc = 0.0
        total_coordinate_sim = 0.0
        
        for case in test_data:
            # Get model belief
            belief_output = self.model.probe_belief()
            belief_map = belief_output["belief_map"]
            uncertainty_map = belief_output["uncertainty_map"]
            
            # Ground truth
            ground_truth_map = case["ground_truth_map"]
            ground_truth_positions = case["ground_truth_positions"]
            ground_truth_directions = case["ground_truth_directions"]
            ground_truth_facing = case["ground_truth_facing"]
            
            # Compute metrics
            pos_acc = positional_accuracy(
                belief_map,
                ground_truth_positions,
                tolerance=self.tolerance_distance
            )
            dir_acc = directional_accuracy(
                belief_map,
                ground_truth_directions,
                tolerance=self.tolerance_angle
            )
            face_acc = facing_accuracy(
                belief_map,
                ground_truth_facing,
                tolerance=self.tolerance_angle
            )
            coord_sim = coordinate_similarity(
                belief_map,
                ground_truth_map
            )
            
            total_positional_acc += pos_acc
            total_directional_acc += dir_acc
            total_facing_acc += face_acc
            total_coordinate_sim += coord_sim
        
        metrics = {
            "belief_positional_accuracy": total_positional_acc / total_cases,
            "belief_directional_accuracy": total_directional_acc / total_cases,
            "belief_facing_accuracy": total_facing_acc / total_cases,
            "belief_coordinate_similarity": total_coordinate_sim / total_cases
        }
        
        return metrics
    
    def evaluate_belief_revision(
        self,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate belief revision using False Belief paradigm.
        
        Args:
            test_data: List of false belief test cases
        
        Returns:
            Dictionary of belief revision metrics
        """
        total_cases = len(test_data)
        total_inertia_score = 0.0
        total_info_gain = 0.0
        successful_revisions = 0
        
        for case in test_data:
            initial_belief = case["initial_belief"]
            new_observation = case["new_observation"]
            ground_truth_revision = case["ground_truth_revision"]
            
            # Get initial belief
            self.model.spatial_belief_memory.reset()
            self.model.spatial_belief_memory.write(
                initial_belief["tokens"],
                camera_poses=initial_belief["camera_poses"]
            )
            initial_belief_map = self.model.spatial_belief_memory.get_belief_map()
            
            # Process new observation
            self.model.explore_step(
                image=new_observation["image"],
                position=new_observation["position"],
                facing=new_observation["facing"]
            )
            
            # Get revised belief
            revised_belief_map = self.model.spatial_belief_memory.get_belief_map()
            
            # Compute belief inertia (resistance to change)
            inertia_score = belief_inertia_score(
                initial_belief_map,
                revised_belief_map
            )
            total_inertia_score += inertia_score
            
            # Compute information gain
            info_gain = information_gain(
                initial_belief_map,
                revised_belief_map,
                new_observation["uncertainty"]
            )
            total_info_gain += info_gain
            
            # Evaluate if belief was correctly revised
            if self._check_revision_correctness(
                revised_belief_map,
                ground_truth_revision
            ):
                successful_revisions += 1
        
        metrics = {
            "belief_inertia_score": total_inertia_score / total_cases,
            "information_gain": total_info_gain / total_cases,
            "belief_revision_success_rate": successful_revisions / total_cases
        }
        
        return metrics
    
    def _plan_route(
        self,
        start_position: Tuple[float, float, float],
        goal_position: Tuple[float, float, float],
        obstacles: List[Dict[str, Any]]
    ) -> List[Tuple[float, float, float]]:
        """Plan a route from start to goal avoiding obstacles."""
        # Simplified route planning - in practice, use proper pathfinding
        path = [start_position, goal_position]
        return path
    
    def _evaluate_route_success(
        self,
        planned_path: List[Tuple[float, float, float]],
        goal_position: Tuple[float, float, float]
    ) -> bool:
        """Evaluate if route successfully reaches goal."""
        if not planned_path:
            return False
        
        final_position = planned_path[-1]
        distance = np.linalg.norm(
            np.array(final_position) - np.array(goal_position)
        )
        
        return distance < self.tolerance_distance
    
    def _compute_optimal_path_length(
        self,
        start_position: Tuple[float, float, float],
        goal_position: Tuple[float, float, float],
        obstacles: List[Dict[str, Any]]
    ) -> float:
        """Compute optimal path length."""
        return np.linalg.norm(
            np.array(start_position) - np.array(goal_position)
        )
    
    def _compute_path_length(self, path: List[Tuple[float, float, float]]) -> float:
        """Compute total path length."""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            total_length += np.linalg.norm(
                np.array(path[i + 1]) - np.array(path[i])
            )
        
        return total_length
    
    def _run_exploration(
        self,
        scene_config: Dict[str, Any],
        exploration_trajectory: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Run exploration trajectory and return belief map."""
        self.model.spatial_belief_memory.reset()
        
        for step in exploration_trajectory:
            self.model.explore_step(
                image=step["image"],
                position=step["position"],
                facing=step["facing"]
            )
        
        belief_output = self.model.probe_belief()
        return belief_output["belief_map"]
    
    def _compute_coverage(
        self,
        belief_map: torch.Tensor,
        ground_truth_map: torch.Tensor
    ) -> float:
        """Compute map coverage percentage."""
        # Simplified coverage computation
        explored_area = (belief_map > 0).float().sum()
        total_area = ground_truth_map.numel()
        
        return (explored_area / total_area).item()
    
    def _compute_map_accuracy(
        self,
        belief_map: torch.Tensor,
        ground_truth_map: torch.Tensor
    ) -> float:
        """Compute map accuracy."""
        # Simplified accuracy computation
        correct = ((belief_map > 0.5) == (ground_truth_map > 0.5)).float()
        return correct.mean().item()
    
    def _detect_objects(self, belief_map: torch.Tensor) -> List[Dict[str, Any]]:
        """Detect objects from belief map."""
        # Simplified object detection
        return []
    
    def _match_objects(
        self,
        detected_objects: List[Dict[str, Any]],
        ground_truth_objects: List[Dict[str, Any]]
    ) -> int:
        """Match detected objects with ground truth."""
        # Simplified matching
        return 0
    
    def _check_revision_correctness(
        self,
        revised_belief: torch.Tensor,
        ground_truth_revision: torch.Tensor
    ) -> bool:
        """Check if belief revision is correct."""
        similarity = coordinate_similarity(revised_belief, ground_truth_revision)
        return similarity > 0.8
