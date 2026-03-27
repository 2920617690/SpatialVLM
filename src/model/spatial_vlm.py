"""Spatial VLM: Complete model assembly for spatial reasoning."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from transformers import PreTrainedModel, AutoModelForCausalLM

from .projector import SpatialProjector
from .three_d_aware_vit import ThreeDAwareViT
from .dual_coordinate_pe import DualCoordinatePE
from .spatial_belief_memory import SpatialSceneGraph, ConfidenceGatedWriter, UncertaintyAwareReader


class SpatialVLM(nn.Module):
    """Complete Spatial VLM model assembly.
    
    Components:
    - three_d_aware_vit: 3D-aware Vision Transformer for extracting spatial tokens
    - dual_coordinate_pe: Dual coordinate positional encoding
    - spatial_belief_memory: Spatial belief memory for cognitive mapping
    - projector: Projects visual tokens to LLM embedding space
    - llm_decoder: Language model decoder for text generation and reasoning
    """
    
    def __init__(
        self,
        three_d_aware_vit: ThreeDAwareViT,
        dual_coordinate_pe: DualCoordinatePE,
        spatial_belief_memory: SpatialBeliefMemory,
        projector: SpatialProjector,
        llm_decoder: PreTrainedModel,
        max_memory_size: int = 1000
    ):
        """Initialize Spatial VLM model.
        
        Args:
            three_d_aware_vit: 3D-aware Vision Transformer
            dual_coordinate_pe: Dual coordinate positional encoding module
            spatial_belief_memory: Spatial belief memory module
            projector: Visual-to-LLM projector
            llm_decoder: Language model decoder
            max_memory_size: Maximum size of spatial belief memory
        """
        super().__init__()
        self.three_d_aware_vit = three_d_aware_vit
        self.dual_coordinate_pe = dual_coordinate_pe
        self.spatial_belief_memory = spatial_belief_memory
        self.projector = projector
        self.llm_decoder = llm_decoder
        self.max_memory_size = max_memory_size
        
        # Exploration state
        self.exploration_step_count = 0
        self.current_position = None
        self.current_facing = None
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        camera_poses: Optional[torch.Tensor] = None,
        coordinates: Optional[torch.Tensor] = None,
        memory_read: bool = True,
        memory_write: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the complete Spatial VLM model.
        
        Pipeline: image → 3D tokens → PE → Memory Read/Write → LLM → output
        
        Args:
            images: Input images of shape (batch_size, num_frames, channels, height, width)
            input_ids: Token IDs for text input of shape (batch_size, seq_len)
            attention_mask: Attention mask for text input of shape (batch_size, seq_len)
            camera_poses: Optional camera poses for each frame
            coordinates: Optional coordinates for dual coordinate PE
            memory_read: Whether to read from spatial belief memory
            memory_write: Whether to write to spatial belief memory
        
        Returns:
            Dictionary containing:
            - logits: Language model output logits
            - memory_state: Current spatial belief memory state
            - spatial_tokens: Projected spatial tokens
        """
        batch_size = images.shape[0]
        
        # Extract 3D-aware visual tokens
        visual_tokens = self.three_d_aware_vit(images)
        
        # Apply dual coordinate positional encoding
        if coordinates is not None:
            visual_tokens = self.dual_coordinate_pe(visual_tokens, coordinates)
        
        # Spatial belief memory operations
        if memory_read:
            memory_tokens = self.spatial_belief_memory.read(
                visual_tokens,
                camera_poses=camera_poses
            )
        else:
            memory_tokens = torch.zeros_like(visual_tokens)
        
        # Combine visual and memory tokens
        combined_tokens = visual_tokens + memory_tokens
        
        # Project to LLM embedding space
        projected_tokens = self.projector(combined_tokens)
        
        # Get LLM embeddings for text input
        text_embeddings = self.llm_decoder.get_input_embeddings()(input_ids)
        
        # Concatenate projected tokens with text embeddings
        combined_embeddings = torch.cat([projected_tokens, text_embeddings], dim=1)
        
        # Extend attention mask for visual tokens
        if attention_mask is not None:
            num_visual_tokens = projected_tokens.shape[1]
            visual_attention_mask = torch.ones(
                batch_size, num_visual_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            combined_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)
        else:
            combined_attention_mask = None
        
        # Forward through LLM
        outputs = self.llm_decoder(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            output_hidden_states=True
        )
        
        # Write to memory if enabled
        if memory_write:
            self.spatial_belief_memory.write(
                combined_tokens,
                camera_poses=camera_poses
            )
        
        return {
            "logits": outputs.logits,
            "memory_state": self.spatial_belief_memory.get_state(),
            "spatial_tokens": projected_tokens,
            "hidden_states": outputs.hidden_states
        }
    
    def explore_step(
        self,
        image: torch.Tensor,
        position: Tuple[float, float, float],
        facing: Tuple[float, float, float],
        action_history: Optional[list] = None,
        memory_read: bool = True,
        memory_write: bool = True
    ) -> Dict[str, Any]:
        """Single step exploration: observe → update memory → decide next action.
        
        Args:
            image: Current observation image
            position: Current 3D position (x, y, z)
            facing: Current facing direction (dx, dy, dz)
            action_history: History of previous actions
            memory_read: Whether to read from memory
            memory_write: Whether to write to memory
        
        Returns:
            Dictionary containing:
            - next_action: Suggested next action
            - belief_map: Current belief map
            - uncertainty_map: Current uncertainty map
        """
        self.exploration_step_count += 1
        self.current_position = position
        self.current_facing = facing
        
        # Extract visual tokens from current observation
        image_batch = image.unsqueeze(0)
        visual_tokens = self.three_d_aware_vit(image_batch)
        
        # Apply coordinate PE based on position and facing
        coordinates = torch.tensor([position + facing], dtype=torch.float32, device=image.device)
        coordinates = coordinates.unsqueeze(0)
        visual_tokens = self.dual_coordinate_pe(visual_tokens, coordinates)
        
        # Read from memory
        if memory_read:
            memory_tokens = self.spatial_belief_memory.read(
                visual_tokens,
                camera_poses=coordinates
            )
        else:
            memory_tokens = torch.zeros_like(visual_tokens)
        
        # Update memory
        if memory_write:
            combined_tokens = visual_tokens + memory_tokens
            self.spatial_belief_memory.write(
                combined_tokens,
                camera_poses=coordinates
            )
        
        # Get current belief and uncertainty maps
        belief_map = self.spatial_belief_memory.get_belief_map()
        uncertainty_map = self.spatial_belief_memory.get_uncertainty_map()
        
        # Project tokens to LLM space for action decision
        projected_tokens = self.projector(combined_tokens)
        
        # Generate next action using LLM
        action_prompt = self._generate_action_prompt(action_history)
        action_input_ids = self._encode_action_prompt(action_prompt)
        
        action_embeddings = self.llm_decoder.get_input_embeddings()(action_input_ids)
        combined_embeddings = torch.cat([projected_tokens, action_embeddings], dim=1)
        
        outputs = self.llm_decoder.generate(
            inputs_embeds=combined_embeddings,
            max_length=50,
            do_sample=True,
            temperature=0.7
        )
        
        next_action = self._decode_action(outputs)
        
        return {
            "next_action": next_action,
            "belief_map": belief_map,
            "uncertainty_map": uncertainty_map,
            "exploration_step": self.exploration_step_count
        }
    
    def probe_belief(self) -> Dict[str, torch.Tensor]:
        """Output current cognitive map for evaluation.
        
        Returns:
            Dictionary containing:
            - belief_map: Current belief map
            - uncertainty_map: Current uncertainty map
            - memory_state: Full memory state
            - exploration_summary: Summary of exploration so far
        """
        belief_map = self.spatial_belief_memory.get_belief_map()
        uncertainty_map = self.spatial_belief_memory.get_uncertainty_map()
        memory_state = self.spatial_belief_memory.get_state()
        
        exploration_summary = {
            "step_count": self.exploration_step_count,
            "current_position": self.current_position,
            "current_facing": self.current_facing,
            "memory_size": self.spatial_belief_memory.get_memory_size()
        }
        
        return {
            "belief_map": belief_map,
            "uncertainty_map": uncertainty_map,
            "memory_state": memory_state,
            "exploration_summary": exploration_summary
        }
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_vlm_path: str,
        three_d_aware_vit: ThreeDAwareViT,
        dual_coordinate_pe: DualCoordinatePE,
        spatial_belief_memory: SpatialBeliefMemory,
        projector: SpatialProjector,
        max_memory_size: int = 1000
    ) -> "SpatialVLM":
        """Load from pretrained VLM and insert new spatial modules.
        
        Args:
            pretrained_vlm_path: Path to pretrained VLM model
            three_d_aware_vit: 3D-aware ViT module
            dual_coordinate_pe: Dual coordinate PE module
            spatial_belief_memory: Spatial belief memory module
            projector: Spatial projector module
            max_memory_size: Maximum memory size
        
        Returns:
            Initialized SpatialVLM model
        """
        llm_decoder = AutoModelForCausalLM.from_pretrained(pretrained_vlm_path)
        
        model = cls(
            three_d_aware_vit=three_d_aware_vit,
            dual_coordinate_pe=dual_coordinate_pe,
            spatial_belief_memory=spatial_belief_memory,
            projector=projector,
            llm_decoder=llm_decoder,
            max_memory_size=max_memory_size
        )
        
        return model
    
    def _generate_action_prompt(self, action_history: Optional[list]) -> str:
        """Generate action prompt for LLM."""
        if action_history is None:
            return "Based on current observation, decide next action:"
        
        history_str = "Previous actions: " + ", ".join(action_history)
        return f"{history_str}. Based on current observation, decide next action:"
    
    def _encode_action_prompt(self, prompt: str) -> torch.Tensor:
        """Encode action prompt to token IDs."""
        tokenizer = self.llm_decoder.get_input_embeddings().weight.device
        # Simplified encoding - in practice, use actual tokenizer
        tokens = [ord(c) for c in prompt]
        return torch.tensor([tokens], dtype=torch.long)
    
    def _decode_action(self, outputs: torch.Tensor) -> str:
        """Decode action from LLM outputs."""
        # Simplified decoding - in practice, use actual tokenizer
        action_tokens = outputs[0].cpu().numpy()
        return "".join([chr(t) if t < 128 else "" for t in action_tokens])
