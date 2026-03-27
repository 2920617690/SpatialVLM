import torch
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np


@dataclass
class ObjectNode:
    """
    Represents an object in the spatial scene graph.
    
    Attributes:
        position: 3D position tensor [x, y, z]
        orientation: Quaternion orientation tensor [w, x, y, z]
        appearance: Feature vector representing object appearance
        confidence: Confidence score for this observation (0.0 to 1.0)
        timestamp: Timestamp of the last observation
        observation_count: Number of times this object has been observed
    """
    position: torch.Tensor
    orientation: torch.Tensor
    appearance: torch.Tensor
    confidence: float
    timestamp: int
    observation_count: int
    
    def __post_init__(self):
        """Validate tensor shapes after initialization."""
        assert self.position.shape == (3,), f"position must have shape (3,), got {self.position.shape}"
        assert self.orientation.shape == (4,), f"orientation must have shape (4,), got {self.orientation.shape}"
        assert 0.0 <= self.confidence <= 1.0, f"confidence must be in [0, 1], got {self.confidence}"


@dataclass
class RelationEdge:
    """
    Represents a spatial relationship between two objects.
    
    Attributes:
        relative_direction: 3D direction vector from source to target
        relative_distance: Euclidean distance between objects
        confidence: Confidence score for this relationship
    """
    relative_direction: torch.Tensor
    relative_distance: float
    confidence: float
    
    def __post_init__(self):
        """Validate tensor shapes after initialization."""
        assert self.relative_direction.shape == (3,), f"relative_direction must have shape (3,), got {self.relative_direction.shape}"
        assert self.relative_distance >= 0.0, f"relative_distance must be non-negative, got {self.relative_distance}"
        assert 0.0 <= self.confidence <= 1.0, f"confidence must be in [0, 1], got {self.confidence}"


@dataclass
class AgentState:
    """
    Represents the agent's state in the environment.
    
    Attributes:
        position: Agent's 3D position [x, y, z]
        orientation: Agent's orientation as quaternion [w, x, y, z]
        timestamp: Timestamp of this state
    """
    position: torch.Tensor
    orientation: torch.Tensor
    timestamp: int
    
    def __post_init__(self):
        """Validate tensor shapes after initialization."""
        assert self.position.shape == (3,), f"position must have shape (3,), got {self.position.shape}"
        assert self.orientation.shape == (4,), f"orientation must have shape (4,), got {self.orientation.shape}"


class SpatialSceneGraph:
    """
    Spatial scene graph maintaining objects, relationships, and agent state.
    
    This graph structure represents the spatial belief memory, tracking objects
    and their relationships over time with confidence scores.
    """
    
    def __init__(self, embedding_dim: int = 256):
        """
        Initialize the spatial scene graph.
        
        Args:
            embedding_dim: Dimension of appearance embeddings
        """
        self.embedding_dim = embedding_dim
        self.nodes: Dict[str, ObjectNode] = {}
        self.edges: Dict[Tuple[str, str], RelationEdge] = {}
        self.agent_state: Optional[AgentState] = None
        
    def set_agent_state(self, position: torch.Tensor, orientation: torch.Tensor, timestamp: int):
        """
        Update the agent's state.
        
        Args:
            position: Agent's 3D position [x, y, z]
            orientation: Agent's orientation as quaternion [w, x, y, z]
            timestamp: Timestamp of this state
        """
        self.agent_state = AgentState(
            position=position,
            orientation=orientation,
            timestamp=timestamp
        )
        
    def add_node(
        self,
        object_id: str,
        position: torch.Tensor,
        orientation: torch.Tensor,
        appearance: torch.Tensor,
        confidence: float = 0.5,
        timestamp: int = 0
    ):
        """
        Add a new object node to the graph.
        
        Args:
            object_id: Unique identifier for the object
            position: 3D position [x, y, z]
            orientation: Quaternion [w, x, y, z]
            appearance: Appearance feature vector
            confidence: Initial confidence score
            timestamp: Observation timestamp
        """
        if object_id in self.nodes:
            raise ValueError(f"Object {object_id} already exists in the graph")
            
        self.nodes[object_id] = ObjectNode(
            position=position,
            orientation=orientation,
            appearance=appearance,
            confidence=confidence,
            timestamp=timestamp,
            observation_count=1
        )
        
    def update_node(
        self,
        object_id: str,
        position: Optional[torch.Tensor] = None,
        orientation: Optional[torch.Tensor] = None,
        appearance: Optional[torch.Tensor] = None,
        confidence: Optional[float] = None,
        timestamp: int = 0
    ):
        """
        Update an existing object node.
        
        Args:
            object_id: Identifier of the object to update
            position: New 3D position (optional)
            orientation: New quaternion (optional)
            appearance: New appearance features (optional)
            confidence: New confidence score (optional)
            timestamp: Update timestamp
        """
        if object_id not in self.nodes:
            raise ValueError(f"Object {object_id} not found in the graph")
            
        node = self.nodes[object_id]
        
        if position is not None:
            node.position = position
        if orientation is not None:
            node.orientation = orientation
        if appearance is not None:
            node.appearance = appearance
        if confidence is not None:
            node.confidence = confidence
        node.timestamp = timestamp
        node.observation_count += 1
        
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relative_direction: torch.Tensor,
        relative_distance: float,
        confidence: float = 0.5
    ):
        """
        Add a spatial relationship edge between two objects.
        
        Args:
            source_id: ID of the source object
            target_id: ID of the target object
            relative_direction: Direction vector from source to target
            relative_distance: Distance between objects
            confidence: Confidence score for this relationship
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Both objects must exist in the graph")
            
        edge_key = (source_id, target_id)
        self.edges[edge_key] = RelationEdge(
            relative_direction=relative_direction,
            relative_distance=relative_distance,
            confidence=confidence
        )
        
    def get_node(self, object_id: str) -> Optional[ObjectNode]:
        """
        Retrieve a node by its ID.
        
        Args:
            object_id: Identifier of the object
            
        Returns:
            ObjectNode if found, None otherwise
        """
        return self.nodes.get(object_id)
        
    def get_all_node_embeddings(self) -> torch.Tensor:
        """
        Get all node appearance embeddings as a tensor.
        
        Returns:
            Tensor of shape (num_nodes, embedding_dim)
        """
        if not self.nodes:
            return torch.empty(0, self.embedding_dim)
            
        embeddings = [node.appearance for node in self.nodes.values()]
        return torch.stack(embeddings)
        
    def get_all_confidences(self) -> torch.Tensor:
        """
        Get all node confidence scores as a tensor.
        
        Returns:
            Tensor of shape (num_nodes,)
        """
        if not self.nodes:
            return torch.empty(0)
            
        confidences = [node.confidence for node in self.nodes.values()]
        return torch.tensor(confidences)
        
    def get_all_positions(self) -> torch.Tensor:
        """
        Get all node positions as a tensor.
        
        Returns:
            Tensor of shape (num_nodes, 3)
        """
        if not self.nodes:
            return torch.empty(0, 3)
            
        positions = [node.position for node in self.nodes.values()]
        return torch.stack(positions)
        
    def to_tensor(self, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        Convert the scene graph to tensors suitable for attention mechanisms.
        
        Args:
            device: Device to place tensors on
            
        Returns:
            Dictionary containing:
                - 'node_embeddings': (num_nodes, embedding_dim)
                - 'node_positions': (num_nodes, 3)
                - 'node_confidences': (num_nodes,)
                - 'edge_indices': (num_edges, 2)
                - 'edge_features': (num_edges, 4) [direction_x, direction_y, direction_z, distance]
                - 'edge_confidences': (num_edges,)
        """
        if not self.nodes:
            return {
                'node_embeddings': torch.empty(0, self.embedding_dim),
                'node_positions': torch.empty(0, 3),
                'node_confidences': torch.empty(0),
                'edge_indices': torch.empty(0, 2, dtype=torch.long),
                'edge_features': torch.empty(0, 4),
                'edge_confidences': torch.empty(0)
            }
            
        # Get node information
        node_ids = list(self.nodes.keys())
        node_embeddings = self.get_all_node_embeddings()
        node_positions = self.get_all_positions()
        node_confidences = self.get_all_confidences()
        
        # Create edge indices and features
        edge_indices = []
        edge_features = []
        edge_confidences = []
        
        for (source_id, target_id), edge in self.edges.items():
            source_idx = node_ids.index(source_id)
            target_idx = node_ids.index(target_id)
            
            edge_indices.append([source_idx, target_idx])
            edge_features.append(torch.cat([
                edge.relative_direction,
                torch.tensor([edge.relative_distance])
            ]))
            edge_confidences.append(edge.confidence)
            
        edge_indices_tensor = torch.tensor(edge_indices, dtype=torch.long)
        edge_features_tensor = torch.stack(edge_features) if edge_features else torch.empty(0, 4)
        edge_confidences_tensor = torch.tensor(edge_confidences) if edge_confidences else torch.empty(0)
        
        # Move to device if specified
        if device is not None:
            node_embeddings = node_embeddings.to(device)
            node_positions = node_positions.to(device)
            node_confidences = node_confidences.to(device)
            edge_indices_tensor = edge_indices_tensor.to(device)
            edge_features_tensor = edge_features_tensor.to(device)
            edge_confidences_tensor = edge_confidences_tensor.to(device)
            
        return {
            'node_embeddings': node_embeddings,
            'node_positions': node_positions,
            'node_confidences': node_confidences,
            'edge_indices': edge_indices_tensor,
            'edge_features': edge_features_tensor,
            'edge_confidences': edge_confidences_tensor
        }
        
    def clear(self):
        """Clear all nodes and edges from the graph."""
        self.nodes.clear()
        self.edges.clear()
        self.agent_state = None
