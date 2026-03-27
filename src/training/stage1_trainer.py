"""Stage 1 trainer: Freeze LLM, train 3D-aware ViT."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from accelerate import Accelerator
from transformers import get_scheduler
import wandb

from ..model.spatial_vlm import SpatialVLM


class Stage1Trainer:
    """Stage 1 training logic: Freeze LLM, train 3D-aware ViT.
    
    Loss components:
    - depth_loss: Depth prediction loss
    - contrastive_loss: Contrastive learning loss for spatial features
    """
    
    def __init__(
        self,
        model: SpatialVLM,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        config: Dict[str, Any],
        accelerator: Optional[Accelerator] = None
    ):
        """Initialize Stage 1 trainer.
        
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
        
        # Freeze LLM and projector
        self._freeze_modules()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Loss weights
        self.depth_loss_weight = config.get("depth_loss_weight", 1.0)
        self.contrastive_loss_weight = config.get("contrastive_loss_weight", 0.5)
        self.temperature = config.get("temperature", 0.07)
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        
        # Initialize wandb if enabled
        if config.get("use_wandb", False):
            wandb.init(
                project=config.get("wandb_project", "spatial_vlm"),
                name=config.get("wandb_run_name", "stage1_3d_grounding"),
                config=config
            )
    
    def _freeze_modules(self):
        """Freeze LLM and projector modules."""
        # Freeze LLM decoder
        for param in self.model.llm_decoder.parameters():
            param.requires_grad = False
        
        # Freeze projector
        for param in self.model.projector.parameters():
            param.requires_grad = False
        
        # Freeze memory and PE
        for param in self.model.spatial_belief_memory.parameters():
            param.requires_grad = False
        for param in self.model.dual_coordinate_pe.parameters():
            param.requires_grad = False
        
        # Ensure 3D-aware ViT is trainable
        for param in self.model.three_d_aware_vit.parameters():
            param.requires_grad = True
    
    def _setup_optimizer(self):
        """Setup optimizer for 3D-aware ViT."""
        trainable_params = [
            p for p in self.model.three_d_aware_vit.parameters() if p.requires_grad
        ]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            betas=(0.9, 0.999)
        )
        
        return self.accelerator.prepare_optimizer(optimizer)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        num_training_steps = len(self.train_dataloader) * self.config.get("num_epochs", 10)
        
        scheduler = get_scheduler(
            name=self.config.get("scheduler", "cosine"),
            optimizer=self.optimizer,
            num_warmup_steps=self.config.get("warmup_steps", 1000),
            num_training_steps=num_training_steps
        )
        
        return self.accelerator.prepare_scheduler(scheduler)
    
    def compute_depth_loss(
        self,
        predicted_depth: torch.Tensor,
        ground_truth_depth: torch.Tensor
    ) -> torch.Tensor:
        """Compute depth prediction loss.
        
        Args:
            predicted_depth: Predicted depth maps
            ground_truth_depth: Ground truth depth maps
        
        Returns:
            Depth loss
        """
        # L1 loss for depth prediction
        loss = nn.functional.l1_loss(predicted_depth, ground_truth_depth, reduction='mean')
        return loss
    
    def compute_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive learning loss.
        
        Args:
            features: Feature embeddings
            labels: Labels for positive pairs
        
        Returns:
            Contrastive loss
        """
        # Normalize features
        features = nn.functional.normalize(features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.transpose(-2, -1)) / self.temperature
        
        # Create positive mask
        batch_size = features.shape[0]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.transpose(0, 1)).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=features.device)
        
        # Compute InfoNCE loss
        exp_sim = torch.exp(similarity)
        positive_sum = (exp_sim * mask).sum(dim=-1)
        all_sum = exp_sim.sum(dim=-1)
        
        loss = -torch.log(positive_sum / (all_sum + 1e-8))
        loss = loss.mean()
        
        return loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_depth_loss = 0.0
        total_contrastive_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            images = batch["images"].to(self.accelerator.device)
            positions = batch["positions"].to(self.accelerator.device)
            ground_truth_depth = batch.get("ground_truth_depth", None)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Extract visual tokens
            visual_tokens = self.model.three_d_aware_vit(images)
            
            # Compute depth loss if available
            depth_loss = torch.tensor(0.0, device=self.accelerator.device)
            if ground_truth_depth is not None:
                predicted_depth = self._predict_depth(visual_tokens)
                depth_loss = self.compute_depth_loss(predicted_depth, ground_truth_depth)
            
            # Compute contrastive loss
            contrastive_loss = self.compute_contrastive_loss(visual_tokens, positions)
            
            # Total loss
            loss = (
                self.depth_loss_weight * depth_loss +
                self.contrastive_loss_weight * contrastive_loss
            )
            
            # Backward pass
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(
                self.model.three_d_aware_vit.parameters(),
                self.config.get("max_grad_norm", 1.0)
            )
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_depth_loss += depth_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log metrics
            if self.global_step % self.config.get("log_interval", 10) == 0:
                metrics = {
                    "train/loss": loss.item(),
                    "train/depth_loss": depth_loss.item(),
                    "train/contrastive_loss": contrastive_loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0]
                }
                
                if self.config.get("use_wandb", False):
                    wandb.log(metrics, step=self.global_step)
        
        avg_metrics = {
            "train/loss": total_loss / num_batches,
            "train/depth_loss": total_depth_loss / num_batches,
            "train/contrastive_loss": total_contrastive_loss / num_batches
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
        total_depth_loss = 0.0
        total_contrastive_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                images = batch["images"].to(self.accelerator.device)
                positions = batch["positions"].to(self.accelerator.device)
                ground_truth_depth = batch.get("ground_truth_depth", None)
                
                # Forward pass
                visual_tokens = self.model.three_d_aware_vit(images)
                
                # Compute losses
                depth_loss = torch.tensor(0.0, device=self.accelerator.device)
                if ground_truth_depth is not None:
                    predicted_depth = self._predict_depth(visual_tokens)
                    depth_loss = self.compute_depth_loss(predicted_depth, ground_truth_depth)
                
                contrastive_loss = self.compute_contrastive_loss(visual_tokens, positions)
                
                loss = (
                    self.depth_loss_weight * depth_loss +
                    self.contrastive_loss_weight * contrastive_loss
                )
                
                total_loss += loss.item()
                total_depth_loss += depth_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                num_batches += 1
        
        avg_metrics = {
            "val/loss": total_loss / num_batches,
            "val/depth_loss": total_depth_loss / num_batches,
            "val/contrastive_loss": total_contrastive_loss / num_batches
        }
        
        return avg_metrics
    
    def _predict_depth(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Predict depth from visual tokens.
        
        Args:
            visual_tokens: Visual tokens from 3D-aware ViT
        
        Returns:
            Predicted depth maps
        """
        # Simple depth prediction head
        batch_size = visual_tokens.shape[0]
        depth_maps = visual_tokens.mean(dim=1)
        return depth_maps
    
    def train(self):
        """Run full training loop."""
        num_epochs = self.config.get("num_epochs", 10)
        
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
