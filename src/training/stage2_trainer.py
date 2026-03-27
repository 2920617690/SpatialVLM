"""Stage 2 trainer: Freeze ViT, train Memory + PE."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from accelerate import Accelerator
from transformers import get_scheduler
import wandb

from ..model.spatial_vlm import SpatialVLM


class Stage2Trainer:
    """Stage 2 training logic: Freeze ViT, train Memory + PE.
    
    Loss components:
    - memory_reconstruction_loss: Reconstruct memory from current observations
    - belief_revision_loss: Update beliefs based on new observations
    - uncertainty_calibration_loss: Calibrate uncertainty estimates
    """
    
    def __init__(
        self,
        model: SpatialVLM,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        config: Dict[str, Any],
        accelerator: Optional[Accelerator] = None
    ):
        """Initialize Stage 2 trainer.
        
        Args:
            model: Spatial VLM model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            config: Training configuration
            accelerator: Accelerate accelerator
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        
        # Freeze ViT and LLM
        self._freeze_modules()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Loss weights
        self.memory_reconstruction_weight = config.get("memory_reconstruction_weight", 1.0)
        self.belief_revision_weight = config.get("belief_revision_weight", 1.0)
        self.uncertainty_calibration_weight = config.get("uncertainty_calibration_weight", 0.5)
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        
        # Initialize wandb if enabled
        if config.get("use_wandb", False):
            wandb.init(
                project=config.get("wandb_project", "spatial_vlm"),
                name=config.get("wandb_run_name", "stage2_spatial_memory"),
                config=config
            )
    
    def _freeze_modules(self):
        """Freeze ViT and LLM modules."""
        # Freeze 3D-aware ViT
        for param in self.model.three_d_aware_vit.parameters():
            param.requires_grad = False
        
        # Freeze LLM decoder
        for param in self.model.llm_decoder.parameters():
            param.requires_grad = False
        
        # Freeze projector
        for param in self.model.projector.parameters():
            param.requires_grad = False
        
        # Ensure memory and PE are trainable
        for param in self.model.spatial_belief_memory.parameters():
            param.requires_grad = True
        for param in self.model.dual_coordinate_pe.parameters():
            param.requires_grad = True
    
    def _setup_optimizer(self):
        """Setup optimizer for Memory + PE."""
        trainable_params = []
        
        for param in self.model.spatial_belief_memory.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        for param in self.model.dual_coordinate_pe.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.get("learning_rate", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
            betas=(0.9, 0.999)
        )
        
        return self.accelerator.prepare_optimizer(optimizer)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        num_training_steps = len(self.train_dataloader) * self.config.get("num_epochs", 15)
        
        scheduler = get_scheduler(
            name=self.config.get("scheduler", "cosine"),
            optimizer=self.optimizer,
            num_warmup_steps=self.config.get("warmup_steps", 500),
            num_training_steps=num_training_steps
        )
        
        return self.accelerator.prepare_scheduler(scheduler)
    
    def compute_memory_reconstruction_loss(
        self,
        memory_tokens: torch.Tensor,
        current_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute memory reconstruction loss.
        
        Args:
            memory_tokens: Tokens retrieved from memory
            current_tokens: Current observation tokens
        
        Returns:
            Reconstruction loss
        """
        # MSE loss for reconstruction
        loss = nn.functional.mse_loss(memory_tokens, current_tokens, reduction='mean')
        return loss
    
    def compute_belief_revision_loss(
        self,
        belief_map: torch.Tensor,
        ground_truth_map: torch.Tensor,
        uncertainty_map: torch.Tensor
    ) -> torch.Tensor:
        """Compute belief revision loss.
        
        Args:
            belief_map: Current belief map
            ground_truth_map: Ground truth scene map
            uncertainty_map: Uncertainty estimates
        
        Returns:
            Belief revision loss
        """
        # Weighted MSE loss based on uncertainty
        weights = 1.0 / (uncertainty_map + 1e-8)
        loss = nn.functional.mse_loss(belief_map * weights, ground_truth_map * weights, reduction='mean')
        return loss
    
    def compute_uncertainty_calibration_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """Compute uncertainty calibration loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            uncertainty: Uncertainty estimates
        
        Returns:
            Calibration loss
        """
        # Compute prediction errors
        errors = torch.abs(predictions - targets)
        
        # Calibration loss: encourage uncertainty to match error magnitude
        loss = torch.abs(uncertainty - errors).mean()
        return loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_memory_loss = 0.0
        total_belief_loss = 0.0
        total_calibration_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            images = batch["images"].to(self.accelerator.device)
            positions = batch["positions"].to(self.accelerator.device)
            camera_poses = batch["camera_poses"].to(self.accelerator.device)
            ground_truth_nodes = batch["ground_truth_nodes"].to(self.accelerator.device)
            trajectory_lengths = batch["trajectory_lengths"]
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Extract visual tokens (frozen)
            with torch.no_grad():
                visual_tokens = self.model.three_d_aware_vit(images)
            
            # Apply coordinate PE (trainable)
            coordinates = torch.cat([positions, camera_poses[:, :, :3, :3].reshape(-1, 9)], dim=-1)
            visual_tokens = self.model.dual_coordinate_pe(visual_tokens, coordinates)
            
            # Memory operations (trainable)
            memory_tokens = self.model.spatial_belief_memory.read(
                visual_tokens,
                camera_poses=camera_poses
            )
            
            # Write to memory
            combined_tokens = visual_tokens + memory_tokens
            self.model.spatial_belief_memory.write(
                combined_tokens,
                camera_poses=camera_poses
            )
            
            # Get belief and uncertainty maps
            belief_map = self.model.spatial_belief_memory.get_belief_map()
            uncertainty_map = self.model.spatial_belief_memory.get_uncertainty_map()
            
            # Compute losses
            memory_reconstruction_loss = self.compute_memory_reconstruction_loss(
                memory_tokens, visual_tokens
            )
            
            belief_revision_loss = self.compute_belief_revision_loss(
                belief_map, ground_truth_nodes, uncertainty_map
            )
            
            uncertainty_calibration_loss = self.compute_uncertainty_calibration_loss(
                belief_map, ground_truth_nodes, uncertainty_map
            )
            
            # Total loss
            loss = (
                self.memory_reconstruction_weight * memory_reconstruction_loss +
                self.belief_revision_weight * belief_revision_loss +
                self.uncertainty_calibration_weight * uncertainty_calibration_loss
            )
            
            # Backward pass
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(
                list(self.model.spatial_belief_memory.parameters()) +
                list(self.model.dual_coordinate_pe.parameters()),
                self.config.get("max_grad_norm", 1.0)
            )
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_memory_loss += memory_reconstruction_loss.item()
            total_belief_loss += belief_revision_loss.item()
            total_calibration_loss += uncertainty_calibration_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Reset memory after each batch
            self.model.spatial_belief_memory.reset()
            
            # Log metrics
            if self.global_step % self.config.get("log_interval", 10) == 0:
                metrics = {
                    "train/loss": loss.item(),
                    "train/memory_reconstruction_loss": memory_reconstruction_loss.item(),
                    "train/belief_revision_loss": belief_revision_loss.item(),
                    "train/uncertainty_calibration_loss": uncertainty_calibration_loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0]
                }
                
                if self.config.get("use_wandb", False):
                    wandb.log(metrics, step=self.global_step)
        
        avg_metrics = {
            "train/loss": total_loss / num_batches,
            "train/memory_reconstruction_loss": total_memory_loss / num_batches,
            "train/belief_revision_loss": total_belief_loss / num_batches,
            "train/uncertainty_calibration_loss": total_calibration_loss / num_batches
        }
        
        return avg_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_memory_loss = 0.0
        total_belief_loss = 0.0
        total_calibration_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                images = batch["images"].to(self.accelerator.device)
                positions = batch["positions"].to(self.accelerator.device)
                camera_poses = batch["camera_poses"].to(self.accelerator.device)
                ground_truth_nodes = batch["ground_truth_nodes"].to(self.accelerator.device)
                
                # Forward pass
                visual_tokens = self.model.three_d_aware_vit(images)
                
                coordinates = torch.cat([positions, camera_poses[:, :, :3, :3].reshape(-1, 9)], dim=-1)
                visual_tokens = self.model.dual_coordinate_pe(visual_tokens, coordinates)
                
                memory_tokens = self.model.spatial_belief_memory.read(
                    visual_tokens,
                    camera_poses=camera_poses
                )
                
                combined_tokens = visual_tokens + memory_tokens
                self.model.spatial_belief_memory.write(
                    combined_tokens,
                    camera_poses=camera_poses
                )
                
                belief_map = self.model.spatial_belief_memory.get_belief_map()
                uncertainty_map = self.model.spatial_belief_memory.get_uncertainty_map()
                
                # Compute losses
                memory_reconstruction_loss = self.compute_memory_reconstruction_loss(
                    memory_tokens, visual_tokens
                )
                
                belief_revision_loss = self.compute_belief_revision_loss(
                    belief_map, ground_truth_nodes, uncertainty_map
                )
                
                uncertainty_calibration_loss = self.compute_uncertainty_calibration_loss(
                    belief_map, ground_truth_nodes, uncertainty_map
                )
                
                loss = (
                    self.memory_reconstruction_weight * memory_reconstruction_loss +
                    self.belief_revision_weight * belief_revision_loss +
                    self.uncertainty_calibration_weight * uncertainty_calibration_loss
                )
                
                total_loss += loss.item()
                total_memory_loss += memory_reconstruction_loss.item()
                total_belief_loss += belief_revision_loss.item()
                total_calibration_loss += uncertainty_calibration_loss.item()
                num_batches += 1
                
                self.model.spatial_belief_memory.reset()
        
        avg_metrics = {
            "val/loss": total_loss / num_batches,
            "val/memory_reconstruction_loss": total_memory_loss / num_batches,
            "val/belief_revision_loss": total_belief_loss / num_batches,
            "val/uncertainty_calibration_loss": total_calibration_loss / num_batches
        }
        
        return avg_metrics
    
    def train(self):
        """Run full training loop."""
        num_epochs = self.config.get("num_epochs", 15)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            if self.config.get("use_wandb", False):
                wandb.log(epoch_metrics, step=self.global_step)
            
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['train/loss']:.4f}")
            if val_metrics:
                print(f"Val Loss: {val_metrics['val/loss']:.4f}")
        
        if self.config.get("use_wandb", False):
            wandb.finish()
