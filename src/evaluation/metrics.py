"""Evaluation metrics for spatial reasoning."""

import torch
import numpy as np
from typing import Tuple, Optional


def positional_accuracy(
    predicted_map: torch.Tensor,
    ground_truth_positions: torch.Tensor,
    tolerance: float = 1.0
) -> float:
    """Compute positional accuracy.
    
    Measures how accurately predicted positions match ground truth positions.
    
    Args:
        predicted_map: Predicted spatial map of shape (H, W) or (H, W, D)
        ground_truth_positions: Ground truth positions of shape (N, 2) or (N, 3)
        tolerance: Distance tolerance for correct prediction
    
    Returns:
        Positional accuracy score between 0 and 1
    """
    if len(ground_truth_positions) == 0:
        return 1.0
    
    # Extract predicted positions from map (peak detection)
    predicted_positions = _extract_peak_positions(predicted_map)
    
    if len(predicted_positions) == 0:
        return 0.0
    
    # Match predicted positions to ground truth
    num_correct = 0
    for gt_pos in ground_truth_positions:
        for pred_pos in predicted_positions:
            distance = np.linalg.norm(
                np.array(gt_pos) - np.array(pred_pos)
            )
            if distance <= tolerance:
                num_correct += 1
                break
    
    accuracy = num_correct / len(ground_truth_positions)
    return accuracy


def directional_accuracy(
    predicted_map: torch.Tensor,
    ground_truth_directions: torch.Tensor,
    tolerance: float = 30.0
) -> float:
    """Compute directional accuracy.
    
    Measures how accurately predicted directions match ground truth directions.
    
    Args:
        predicted_map: Predicted spatial map
        ground_truth_directions: Ground truth directions of shape (N, 3)
        tolerance: Angle tolerance in degrees
    
    Returns:
        Directional accuracy score between 0 and 1
    """
    if len(ground_truth_directions) == 0:
        return 1.0
    
    # Extract predicted directions from map
    predicted_directions = _extract_directions(predicted_map)
    
    if len(predicted_directions) == 0:
        return 0.0
    
    # Normalize directions
    gt_dirs = torch.nn.functional.normalize(ground_truth_directions, dim=-1)
    pred_dirs = torch.nn.functional.normalize(predicted_directions, dim=-1)
    
    # Compute angular differences
    num_correct = 0
    for gt_dir in gt_dirs:
        for pred_dir in pred_dirs:
            dot_product = torch.dot(gt_dir, pred_dir)
            angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
            angle_deg = torch.rad2deg(angle)
            
            if angle_deg <= tolerance:
                num_correct += 1
                break
    
    accuracy = num_correct / len(ground_truth_directions)
    return accuracy


def facing_accuracy(
    predicted_map: torch.Tensor,
    ground_truth_facing: torch.Tensor,
    tolerance: float = 30.0
) -> float:
    """Compute facing accuracy.
    
    Measures how accurately predicted facing directions match ground truth.
    
    Args:
        predicted_map: Predicted spatial map
        ground_truth_facing: Ground truth facing directions of shape (N, 3)
        tolerance: Angle tolerance in degrees
    
    Returns:
        Facing accuracy score between 0 and 1
    """
    # Facing accuracy is similar to directional accuracy
    return directional_accuracy(predicted_map, ground_truth_facing, tolerance)


def belief_inertia_score(
    initial_belief: torch.Tensor,
    revised_belief: torch.Tensor
) -> float:
    """Compute belief inertia score.
    
    Measures resistance to belief revision. Higher score indicates more inertia.
    
    Args:
        initial_belief: Initial belief map
        revised_belief: Revised belief map after new observation
    
    Returns:
        Belief inertia score between 0 and 1
    """
    # Normalize beliefs
    initial_norm = torch.nn.functional.normalize(initial_belief.flatten(), dim=0)
    revised_norm = torch.nn.functional.normalize(revised_belief.flatten(), dim=0)
    
    # Compute cosine similarity
    similarity = torch.dot(initial_norm, revised_norm)
    
    # Inertia is high when similarity is high
    inertia_score = similarity.item()
    
    return inertia_score


def information_gain(
    initial_belief: torch.Tensor,
    revised_belief: torch.Tensor,
    observation_uncertainty: Optional[torch.Tensor] = None
) -> float:
    """Compute information gain from belief revision.
    
    Measures how much new information was gained from observation.
    
    Args:
        initial_belief: Initial belief map
        revised_belief: Revised belief map after new observation
        observation_uncertainty: Uncertainty of the observation
    
    Returns:
        Information gain score
    """
    # Compute entropy reduction
    initial_entropy = _compute_entropy(initial_belief)
    revised_entropy = _compute_entropy(revised_belief)
    
    entropy_reduction = initial_entropy - revised_entropy
    
    # Adjust for observation uncertainty if provided
    if observation_uncertainty is not None:
        avg_uncertainty = observation_uncertainty.mean().item()
        info_gain = entropy_reduction * (1.0 - avg_uncertainty)
    else:
        info_gain = entropy_reduction
    
    return info_gain


def coordinate_similarity(
    predicted_map: torch.Tensor,
    ground_truth_map: torch.Tensor
) -> float:
    """Compute coordinate similarity between predicted and ground truth maps.
    
    Args:
        predicted_map: Predicted spatial map
        ground_truth_map: Ground truth spatial map
    
    Returns:
        Coordinate similarity score between 0 and 1
    """
    # Ensure maps have same shape
    if predicted_map.shape != ground_truth_map.shape:
        min_shape = [
            min(p, g) for p, g in zip(predicted_map.shape, ground_truth_map.shape)
        ]
        predicted_map = predicted_map[:min_shape[0], :min_shape[1]]
        ground_truth_map = ground_truth_map[:min_shape[0], :min_shape[1]]
    
    # Normalize maps
    pred_flat = predicted_map.flatten()
    gt_flat = ground_truth_map.flatten()
    
    pred_norm = torch.nn.functional.normalize(pred_flat, dim=0)
    gt_norm = torch.nn.functional.normalize(gt_flat, dim=0)
    
    # Compute cosine similarity
    similarity = torch.dot(pred_norm, gt_norm)
    
    return similarity.item()


def _extract_peak_positions(
    spatial_map: torch.Tensor,
    num_peaks: int = 10,
    threshold: float = 0.5
) -> list:
    """Extract peak positions from spatial map.
    
    Args:
        spatial_map: Spatial map
        num_peaks: Maximum number of peaks to extract
        threshold: Minimum value threshold for peaks
    
    Returns:
        List of peak positions
    """
    # Flatten map and find peaks
    flat_map = spatial_map.flatten()
    values, indices = torch.topk(flat_map, min(num_peaks, flat_map.numel()))
    
    # Filter by threshold
    valid_mask = values > threshold
    valid_indices = indices[valid_mask]
    
    # Convert indices to positions
    positions = []
    for idx in valid_indices:
        if spatial_map.dim() == 2:
            pos = np.unravel_index(idx.item(), spatial_map.shape)
        elif spatial_map.dim() == 3:
            pos = np.unravel_index(idx.item(), spatial_map.shape)
        positions.append(pos)
    
    return positions


def _extract_directions(
    spatial_map: torch.Tensor
) -> torch.Tensor:
    """Extract direction vectors from spatial map.
    
    Args:
        spatial_map: Spatial map
    
    Returns:
        Direction vectors
    """
    # Compute gradients to estimate directions
    if spatial_map.dim() == 2:
        grad_y, grad_x = torch.gradient(spatial_map)
        directions = torch.stack([grad_x, grad_y], dim=-1)
    elif spatial_map.dim() == 3:
        grad_z, grad_y, grad_x = torch.gradient(spatial_map)
        directions = torch.stack([grad_x, grad_y, grad_z], dim=-1)
    else:
        directions = torch.zeros_like(spatial_map)
    
    # Flatten and normalize
    directions_flat = directions.flatten(end_dim=-2)
    directions_norm = torch.nn.functional.normalize(directions_flat, dim=-1)
    
    return directions_norm


def _compute_entropy(belief_map: torch.Tensor) -> float:
    """Compute entropy of belief map.
    
    Args:
        belief_map: Belief map
    
    Returns:
        Entropy value
    """
    # Normalize to probabilities
    probs = torch.softmax(belief_map.flatten(), dim=0)
    
    # Compute entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
    
    return entropy.item()
