import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConfidenceGatedWriter(nn.Module):
    """
    Confidence-gated writer for spatial belief memory.
    
    This module manages memory updates based on confidence scores and conflict detection.
    The update strategy depends on the current memory confidence and the level of conflict
    with new observations.
    """
    
    def __init__(self, embedding_dim: int, conflict_threshold: float = 0.5):
        """
        Initialize the confidence-gated writer.
        
        Args:
            embedding_dim: Dimension of memory embeddings
            conflict_threshold: Threshold for detecting high conflict between old and new info
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conflict_threshold = conflict_threshold
        
        # Conflict detection network
        self.conflict_detector = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # Revision gate for controlling update weight
        self.revision_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # Confidence update network
        self.confidence_updater = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def write(
        self,
        memory: torch.Tensor,
        new_observation_embedding: torch.Tensor,
        object_id: str,
        current_confidence: float,
        observation_count: int,
        memory_confidence_map: Optional[dict] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Write new observation to memory with confidence-gated update.
        
        Update strategy:
        - High confidence + Low conflict: Fine-tune (small update)
        - High confidence + High conflict: Trigger belief revision (larger update)
        - Low confidence: Direct update (replace with new observation)
        
        Args:
            memory: Current memory embedding of shape (embedding_dim,)
            new_observation_embedding: New observation embedding of shape (embedding_dim,)
            object_id: Identifier of the object being updated
            current_confidence: Current confidence score of the memory (0.0 to 1.0)
            observation_count: Number of times this object has been observed
            memory_confidence_map: Optional dictionary mapping object IDs to confidence scores
            
        Returns:
            Tuple of (updated_memory, new_confidence)
        """
        # Detect conflict between current memory and new observation
        conflict = self._detect_conflict(memory, new_observation_embedding)
        
        # Determine update strategy based on confidence and conflict
        high_confidence = current_confidence > 0.7
        high_conflict = conflict > self.conflict_threshold
        
        if high_confidence and not high_conflict:
            # Fine-tune: small update based on revision gate
            update_weight = self.revision_gate(torch.cat([memory, new_observation_embedding], dim=-1))
            updated_memory = memory + update_weight * (new_observation_embedding - memory)
            
        elif high_confidence and high_conflict:
            # Belief revision: larger update to resolve conflict
            revision_weight = 0.5 * (1 + conflict)  # Higher conflict = larger update
            updated_memory = (1 - revision_weight) * memory + revision_weight * new_observation_embedding
            
        else:
            # Low confidence: direct update with new observation
            updated_memory = new_observation_embedding
            
        # Update confidence based on conflict and observation count
        new_confidence = self._update_confidence(
            current_confidence,
            conflict,
            observation_count,
            memory,
            new_observation_embedding
        )
        
        return updated_memory, new_confidence
        
    def _detect_conflict(self, memory: torch.Tensor, new_embedding: torch.Tensor) -> float:
        """
        Detect the degree of conflict between current memory and new observation.
        
        Args:
            memory: Current memory embedding
            new_embedding: New observation embedding
            
        Returns:
            Conflict score between 0.0 and 1.0
        """
        combined = torch.cat([memory, new_embedding], dim=-1)
        conflict_score = self.conflict_detector(combined)
        return conflict_score.item()
        
    def _update_confidence(
        self,
        current_confidence: float,
        conflict: float,
        observation_count: int,
        memory: torch.Tensor,
        new_embedding: torch.Tensor
    ) -> float:
        """
        Update confidence score based on conflict level and observation count.
        
        Logic:
        - Low conflict increases confidence
        - High conflict decreases confidence
        - More observations increase confidence (up to a point)
        - Consistency between memory and new observation affects confidence
        
        Args:
            current_confidence: Current confidence score
            conflict: Detected conflict level
            observation_count: Number of observations
            memory: Current memory embedding
            new_embedding: New observation embedding
            
        Returns:
            Updated confidence score between 0.0 and 1.0
        """
        # Compute similarity between memory and new observation
        similarity = F.cosine_similarity(
            memory.unsqueeze(0),
            new_embedding.unsqueeze(0),
            dim=-1
        ).item()
        
        # Use neural network to compute confidence update
        combined = torch.cat([
            memory,
            new_embedding,
            torch.tensor([conflict, float(observation_count)], dtype=memory.dtype)
        ], dim=-1)
        
        confidence_delta = self.confidence_updater(combined).item()
        
        # Adjust confidence based on conflict and similarity
        if conflict > self.conflict_threshold:
            # High conflict reduces confidence
            confidence_adjustment = -0.1 * conflict
        elif similarity > 0.8:
            # High similarity increases confidence
            confidence_adjustment = 0.05 * similarity
        else:
            # Moderate similarity
            confidence_adjustment = 0.01
            
        # Observation count factor: more consistent observations = higher confidence
        observation_factor = min(0.1, 0.02 * observation_count)
        
        # Compute new confidence
        new_confidence = current_confidence + confidence_adjustment + observation_factor + confidence_delta
        
        # Clamp to valid range
        new_confidence = max(0.0, min(1.0, new_confidence))
        
        return new_confidence
        
    def batch_write(
        self,
        memories: torch.Tensor,
        new_embeddings: torch.Tensor,
        current_confidences: torch.Tensor,
        observation_counts: torch.Tensor,
        object_ids: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch version of write operation for multiple objects.
        
        Args:
            memories: Batch of current memory embeddings (batch_size, embedding_dim)
            new_embeddings: Batch of new observation embeddings (batch_size, embedding_dim)
            current_confidences: Batch of current confidence scores (batch_size,)
            observation_counts: Batch of observation counts (batch_size,)
            object_ids: List of object identifiers
            
        Returns:
            Tuple of (updated_memories, new_confidences)
        """
        batch_size = memories.shape[0]
        
        # Detect conflicts for all pairs
        combined = torch.cat([memories, new_embeddings], dim=-1)
        conflicts = self.conflict_detector(combined).squeeze(-1)  # (batch_size,)
        
        # Compute revision weights
        revision_weights = self.revision_gate(combined).squeeze(-1)  # (batch_size,)
        
        # Determine update strategies
        high_confidence_mask = current_confidences > 0.7
        high_conflict_mask = conflicts > self.conflict_threshold
        
        # Initialize updated memories
        updated_memories = memories.clone()
        
        # Apply fine-tune updates (high confidence, low conflict)
        fine_tune_mask = high_confidence_mask & (~high_conflict_mask)
        if fine_tune_mask.any():
            idx = fine_tune_mask.nonzero().squeeze(-1)
            updated_memories[idx] = memories[idx] + revision_weights[idx].unsqueeze(-1) * (
                new_embeddings[idx] - memories[idx]
            )
            
        # Apply belief revision updates (high confidence, high conflict)
        revision_mask = high_confidence_mask & high_conflict_mask
        if revision_mask.any():
            idx = revision_mask.nonzero().squeeze(-1)
            revision_weight = 0.5 * (1 + conflicts[idx])
            updated_memories[idx] = (
                (1 - revision_weight).unsqueeze(-1) * memories[idx] +
                revision_weight.unsqueeze(-1) * new_embeddings[idx]
            )
            
        # Apply direct updates (low confidence)
        direct_update_mask = ~high_confidence_mask
        if direct_update_mask.any():
            idx = direct_update_mask.nonzero().squeeze(-1)
            updated_memories[idx] = new_embeddings[idx]
            
        # Update confidences
        new_confidences = self._batch_update_confidences(
            current_confidences,
            conflicts,
            observation_counts,
            memories,
            new_embeddings
        )
        
        return updated_memories, new_confidences
        
    def _batch_update_confidences(
        self,
        current_confidences: torch.Tensor,
        conflicts: torch.Tensor,
        observation_counts: torch.Tensor,
        memories: torch.Tensor,
        new_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch update of confidence scores.
        
        Args:
            current_confidences: Current confidence scores (batch_size,)
            conflicts: Conflict levels (batch_size,)
            observation_counts: Observation counts (batch_size,)
            memories: Current memory embeddings (batch_size, embedding_dim)
            new_embeddings: New observation embeddings (batch_size, embedding_dim)
            
        Returns:
            Updated confidence scores (batch_size,)
        """
        # Compute similarities
        similarities = F.cosine_similarity(memories, new_embeddings, dim=-1)
        
        # Prepare combined input for confidence updater
        batch_size = memories.shape[0]
        conflict_obs_tensor = torch.stack([conflicts, observation_counts.float()], dim=-1)
        combined = torch.cat([
            memories,
            new_embeddings,
            conflict_obs_tensor
        ], dim=-1)
        
        confidence_deltas = self.confidence_updater(combined).squeeze(-1)
        
        # Compute confidence adjustments
        confidence_adjustments = torch.where(
            conflicts > self.conflict_threshold,
            -0.1 * conflicts,
            torch.where(
                similarities > 0.8,
                0.05 * similarities,
                torch.tensor(0.01, device=memories.device)
            )
        )
        
        # Observation factor
        observation_factors = torch.minimum(
            torch.tensor(0.1, device=memories.device),
            0.02 * observation_counts.float()
        )
        
        # Compute new confidences
        new_confidences = (
            current_confidences +
            confidence_adjustments +
            observation_factors +
            confidence_deltas
        )
        
        # Clamp to valid range
        new_confidences = torch.clamp(new_confidences, 0.0, 1.0)
        
        return new_confidences
