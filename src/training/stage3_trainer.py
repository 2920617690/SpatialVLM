"""Stage 3 trainer: End-to-end fine-tuning with LoRA."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from accelerate import Accelerator
from transformers import get_scheduler
from peft import LoraConfig, get_peft_model
import wandb

from ..model.spatial_vlm import SpatialVLM


class Stage3Trainer:
    """Stage 3 training logic: End-to-end fine-tuning with LoRA.
    
    Loss components:
    - task_loss: Task-specific loss (e.g., navigation, object search)
    - exploration_reward: Reward for effective exploration
    """
    
    def __init__(
        self,
        model: SpatialVLM,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        config: Dict[str, Any],
        accelerator: Optional[Accelerator] = None
    ):
        """Initialize Stage 3 trainer.
        
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
        
        # Apply LoRA to LLM
        self._apply_lora()
        
        # Unfreeze all modules
        self._unfreeze_modules()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Loss weights
        self.task_loss_weight = config.get("task_loss_weight", 1.0)
        self.exploration_reward_weight = config.get("exploration_reward_weight", 0.1)
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        
        # Initialize wandb if enabled
        if config.get("use_wandb", False):
            wandb.init(
                project=config.get("wandb_project", "spatial_vlm"),
                name=config.get("wandb_run_name", "stage3_end2end"),
                config=config
            )
    
    def _apply_lora(self):
        """Apply LoRA to LLM decoder."""
        lora_config = LoraConfig(
            r=self.config.get("lora_r", 8),
            lora_alpha=self.config.get("lora_alpha", 32),
            target_modules=self.config.get("lora_target_modules", ["q_proj", "v_proj"]),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model.llm_decoder = get_peft_model(self.model.llm_decoder, lora_config)
    
    def _unfreeze_modules(self):
        """Unfreeze all modules for end-to-end training."""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def _setup_optimizer(self):
        """Setup optimizer for all trainable parameters."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.get("learning_rate", 1e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
            betas=(0.9, 0.999)
        )
        
        return self.accelerator.prepare_optimizer(optimizer)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        num_training_steps = len(self.train_dataloader) * self.config.get("num_epochs", 20)
        
        scheduler = get_scheduler(
            name=self.config.get("scheduler", "cosine"),
            optimizer=self.optimizer,
            num_warmup_steps=self.config.get("warmup_steps", 2000),
            num_training_steps=num_training_steps
        )
        
        return self.accelerator.prepare_scheduler(scheduler)
    
    def compute_task_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        task_type: str = "navigation"
    ) -> torch.Tensor:
        """Compute task-specific loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            task_type: Type of task (navigation, object_search, etc.)
        
        Returns:
            Task loss
        """
        if task_type == "navigation":
            # Cross-entropy loss for action prediction
            loss = nn.functional.cross_entropy(predictions, targets, reduction='mean')
        elif task_type == "object_search":
            # Binary cross-entropy for object presence
            loss = nn.functional.binary_cross_entropy_with_logits(predictions, targets, reduction='mean')
        else:
            # Default to MSE loss
            loss = nn.functional.mse_loss(predictions, targets, reduction='mean')
        
        return loss
    
    def compute_exploration_reward(
        self,
        belief_map: torch.Tensor,
        uncertainty_map: torch.Tensor,
        trajectory_length: int,
        success: bool
    ) -> torch.Tensor:
        """Compute exploration reward.
        
        Args:
            belief_map: Current belief map
            uncertainty_map: Current uncertainty map
            trajectory_length: Length of exploration trajectory
            success: Whether the task was completed successfully
        
        Returns:
            Exploration reward (negative for loss)
        """
        # Reward for reducing uncertainty
        uncertainty_reduction = uncertainty_map.mean()
        reward = -uncertainty_reduction
        
        # Reward for efficiency (shorter trajectories are better)
        efficiency_penalty = trajectory_length / 100.0
        reward -= efficiency_penalty
        
        # Reward for success
        if success:
            reward += 1.0
        
        # Convert to loss (negative reward)
        loss = -reward
        
        return loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_exploration_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            images = batch["images"].to(self.accelerator.device)
            input_ids = batch.get("input_ids", None)
            attention_mask = batch.get("attention_mask", None)
            positions = batch["positions"].to(self.accelerator.device)
            camera_poses = batch["camera_poses"].to(self.accelerator.device)
            actions = batch["actions"].to(self.accelerator.device)
            trajectory_lengths = batch["trajectory_lengths"]
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Full model forward pass
            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                camera_poses=camera_poses
            )
            
            logits = outputs["logits"]
            belief_map = outputs["memory_state"].get("belief_map", None)
            uncertainty_map = outputs["memory_state"].get("uncertainty_map", None)
            
            # Compute task loss (action prediction)
            task_loss = self.compute_task_loss(
                logits[:, -1, :4],  # Action logits
                actions[:, 0],  # First action
                task_type="navigation"
            )
            
            # Compute exploration reward
            if belief_map is not None and uncertainty_map is not None:
                exploration_loss = self.compute_exploration_reward(
                    belief_map,
                    uncertainty_map,
                    trajectory_lengths[0].item(),
                    success=False
                )
            else:
                exploration_loss = torch.tensor(0.0, device=self.accelerator.device)
            
            # Total loss
            loss = (
                self.task_loss_weight * task_loss +
                self.exploration_reward_weight * exploration_loss
            )
            
            # Backward pass
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("max_grad_norm", 1.0)
            )
            self.optimizer.step()
            self.scheduler.step()
            
            # Reset memory after each batch
            self.model.spatial_belief_memory.reset()
            
            # Accumulate losses
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_exploration_loss += exploration_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log metrics
            if self.global_step % self.config.get("log_interval", 10) == 0:
                metrics = {
                    "train/loss": loss.item(),
                    "train/task_loss": task_loss.item(),
                    "train/exploration_loss": exploration_loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0]
                }
                
                if self.config.get("use_wandb", False):
                    wandb.log(metrics, step=self.global_step)
        
        avg_metrics = {
            "train/loss": total_loss / num_batches,
            "train/task_loss": total_task_loss / num_batches,
            "train/exploration_loss": total_exploration_loss / num_batches
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
        total_task_loss = 0.0
        total_exploration_loss = 0.0
        num_batches = 0
        total_success = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                images = batch["images"].to(self.accelerator.device)
                input_ids = batch.get("input_ids", None)
                attention_mask = batch.get("attention_mask", None)
                positions = batch["positions"].to(self.accelerator.device)
                camera_poses = batch["camera_poses"].to(self.accelerator.device)
                actions = batch["actions"].to(self.accelerator.device)
                trajectory_lengths = batch["trajectory_lengths"]
                
                # Forward pass
                outputs = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    camera_poses=camera_poses
                )
                
                logits = outputs["logits"]
                belief_map = outputs["memory_state"].get("belief_map", None)
                uncertainty_map = outputs["memory_state"].get("uncertainty_map", None)
                
                # Compute task loss
                task_loss = self.compute_task_loss(
                    logits[:, -1, :4],
                    actions[:, 0],
                    task_type="navigation"
                )
                
                # Compute exploration reward
                if belief_map is not None and uncertainty_map is not None:
                    exploration_loss = self.compute_exploration_reward(
                        belief_map,
                        uncertainty_map,
                        trajectory_lengths[0].item(),
                        success=False
                    )
                else:
                    exploration_loss = torch.tensor(0.0, device=self.accelerator.device)
                
                loss = (
                    self.task_loss_weight * task_loss +
                    self.exploration_reward_weight * exploration_loss
                )
                
                total_loss += loss.item()
                total_task_loss += task_loss.item()
                total_exploration_loss += exploration_loss.item()
                num_batches += 1
                
                self.model.spatial_belief_memory.reset()
        
        avg_metrics = {
            "val/loss": total_loss / num_batches,
            "val/task_loss": total_task_loss / num_batches,
            "val/exploration_loss": total_exploration_loss / num_batches
        }
        
        return avg_metrics
    
    def train(self):
        """Run full training loop."""
        num_epochs = self.config.get("num_epochs", 20)
        
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
