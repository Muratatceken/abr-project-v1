import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
from pathlib import Path
import time
from datetime import datetime
import json
import pickle
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.cvae_model import ConditionalVAEWithMasking
from src.models.advanced_cvae_model import AdvancedConditionalVAE
from configs.m1_optimization_config import M1OptimizationConfig
from src.utils.training_logger import TrainingHistoryLogger

logger = logging.getLogger(__name__)

class CVAETrainer:
    """
    Enhanced trainer for Conditional Variational Autoencoder with improved loss tracking.
    
    Features:
    - Better loss component tracking
    - Custom beta scheduling to prevent KL collapse
    - Enhanced debugging and monitoring
    - Improved configuration handling
    """
    
    def __init__(self,
                 model,  # Union[ConditionalVAEWithMasking, AdvancedConditionalVAE]
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None,
                 use_m1_optimizations: bool = True,
                 notes: Optional[str] = None):
        """
        Initialize the enhanced CVAE trainer.
        
        Args:
            model: ConditionalVAE model
            config: Training configuration
            device: Device to use for training
            use_m1_optimizations: Whether to apply M1 optimizations
            notes: Optional notes about this training run
        """
        self.config = config
        self.model = model
        self.notes = notes
        
        # Setup device and M1 optimizations
        if use_m1_optimizations and device is None:
            self.m1_config = M1OptimizationConfig()
            self.device = self.m1_config.get_optimal_device()
            self.m1_config.setup_torch_optimizations()
        else:
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if use_m1_optimizations and self.device.type == 'mps':
                self.m1_config = M1OptimizationConfig()
                self.m1_config.setup_torch_optimizations()
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup logging
        self.setup_logging()
        
        # Initialize training history logger
        self.history_logger = TrainingHistoryLogger(base_dir=str(self.output_dir))
        
        # Enhanced beta scheduling
        self.beta_scheduler = self._setup_beta_scheduler()
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.get('advanced', {}).get('gradient_accumulation_steps', 1)
        
        # Debug settings
        self.debug_config = config.get('debug', {})
        
        # Training run tracking
        self.run_id = None
        self.training_start_time = None
        
        # Detect model type
        self.model_type = 'advanced' if isinstance(model, AdvancedConditionalVAE) else 'original'
        
        logger.info(f"Enhanced CVAE Trainer initialized on device: {self.device}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Log configuration summary
        self._log_config_summary()
    
    def _setup_beta_scheduler(self):
        """Setup enhanced beta scheduler to prevent KL collapse."""
        training_config = self.config.get('training', {})
        beta_config = self.config.get('beta_schedule', {})
        
        if beta_config.get('type') == 'custom':
            return CustomBetaScheduler(beta_config.get('schedule', []))
        else:
            return BetaScheduler(
                initial_beta=training_config.get('initial_beta', 0.0),
                final_beta=training_config.get('final_beta', 0.01),
                annealing_epochs=training_config.get('beta_annealing_epochs', 150)
            )
    
    def _log_config_summary(self):
        """Log important configuration settings."""
        model_config = self.config.get('model', {})
        training_config = self.config.get('training', {})
        
        logger.info("=== CONFIGURATION SUMMARY ===")
        logger.info(f"Model: Latent={model_config.get('latent_dim', 32)}, Hidden={model_config.get('hidden_dim', 64)}")
        logger.info(f"Training: LR={training_config.get('optimizer', {}).get('lr', 0.0001)}, Batch={training_config.get('batch_size', 8)}")
        logger.info(f"Beta: {training_config.get('initial_beta', 0.0)} → {training_config.get('final_beta', 0.01)} over {training_config.get('beta_annealing_epochs', 150)} epochs")
        logger.info("=" * 30)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with enhanced configuration support."""
        optimizer_config = self.config.get('training', {}).get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        
        # Helper function to convert string values to float
        def safe_float(value, default):
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return default
            return value if value is not None else default
        
        if optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=safe_float(optimizer_config.get('lr'), 1e-4),
                weight_decay=safe_float(optimizer_config.get('weight_decay'), 0.0),
                betas=optimizer_config.get('betas', (0.9, 0.999)),
                eps=safe_float(optimizer_config.get('eps'), 1e-8)
            )
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=safe_float(optimizer_config.get('lr'), 1e-4),
                weight_decay=safe_float(optimizer_config.get('weight_decay'), 0.0),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler with enhanced options."""
        scheduler_config = self.config.get('training', {}).get('scheduler', {})
        if not scheduler_config.get('enabled', True):
            return None
        
        scheduler_type = scheduler_config.get('type', 'step')
        
        # Helper function to convert string values to float
        def safe_float(value, default):
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return default
            return value if value is not None else default
        
        if scheduler_type == 'cosine':
            min_lr = safe_float(scheduler_config.get('min_lr'), 1e-6)
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('training', {}).get('epochs', 200),
                eta_min=min_lr
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 50),
                gamma=scheduler_config.get('gamma', 0.8)
            )
        elif scheduler_type == 'plateau':
            factor = safe_float(scheduler_config.get('factor'), 0.5)
            min_lr = safe_float(scheduler_config.get('min_lr'), 1e-6)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=scheduler_config.get('patience', 10),
                min_lr=min_lr
            )
        else:
            return None
    
    def setup_logging(self):
        """Setup logging and tensorboard."""
        # Create output directories
        self.output_dir = Path(self.config.get('training', {}).get('output_dir', 'outputs'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.results_dir = self.output_dir / 'results'
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _get_model_architecture_info(self) -> Dict[str, Any]:
        """Get model architecture information for logging."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_class': self.model.__class__.__name__,
            'latent_dim': getattr(self.model, 'latent_dim', None),
            'condition_dim': getattr(self.model, 'condition_dim', None),
            'sequence_length': getattr(self.model, 'sequence_length', None),
            'device': str(self.device)
        }
    
    def start_training_run(self, data_path: Optional[str] = None) -> str:
        """Start a new training run and initialize logging."""
        # Get model architecture info
        model_arch = self._get_model_architecture_info()
        
        # Start training run in history logger
        self.run_id = self.history_logger.start_training_run(
            config=self.config,
            model_architecture=model_arch,
            data_path=data_path,
            notes=self.notes
        )
        
        self.training_start_time = time.time()
        logger.info(f"Started training run: {self.run_id}")
        return self.run_id
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Enhanced training epoch with better loss tracking."""
        self.model.train()
        
        # Update beta for KL annealing
        current_beta = self.beta_scheduler.get_beta(epoch)
        
        # Handle different model types for beta setting
        if hasattr(self.model, 'beta'):
            self.model.beta = current_beta
        elif hasattr(self.model, 'beta_waveform') and hasattr(self.model, 'beta_peak'):
            # Advanced CVAE with hierarchical betas
            # Scale the betas proportionally
            original_beta_waveform = getattr(self.model, '_original_beta_waveform', self.model.beta_waveform)
            original_beta_peak = getattr(self.model, '_original_beta_peak', self.model.beta_peak)
            
            if not hasattr(self.model, '_original_beta_waveform'):
                self.model._original_beta_waveform = self.model.beta_waveform
                self.model._original_beta_peak = self.model.beta_peak
            
            self.model.beta_waveform = current_beta * original_beta_waveform
            self.model.beta_peak = current_beta * original_beta_peak
        
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'beta': current_beta
        }
        
        num_batches = len(train_loader)
        
        # Debug tracking
        if self.debug_config.get('loss_component_tracking', False):
            detailed_losses = []
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)
                    elif isinstance(batch[key], dict):
                        for subkey in batch[key]:
                            batch[key][subkey] = batch[key][subkey].to(self.device, non_blocking=True)
                
                # Forward pass
                loss_dict = self.model.compute_loss(batch)
                
                # Debug: Check for NaN or extreme values
                if self.debug_config.get('check_nan', False):
                    self._check_for_nan(loss_dict, batch_idx, epoch)
                
                # Scale loss for gradient accumulation
                loss = loss_dict['total_loss'] / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.get('training', {}).get('gradient_clipping', {}).get('enabled', True):
                        max_norm = self.config.get('training', {}).get('gradient_clipping', {}).get('max_norm', 0.1)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update epoch losses with correct key mapping
                epoch_losses['total_loss'] += loss_dict['total_loss'].item()
                epoch_losses['reconstruction_loss'] += loss_dict.get('abr_recon_loss', torch.tensor(0.0)).item()
                epoch_losses['kl_loss'] += loss_dict.get('kl_loss', torch.tensor(0.0)).item()
                
                # Track additional losses for Advanced CVAE
                if 'waveform_kl_loss' in loss_dict:
                    if 'waveform_kl_loss' not in epoch_losses:
                        epoch_losses['waveform_kl_loss'] = 0.0
                    epoch_losses['waveform_kl_loss'] += loss_dict['waveform_kl_loss'].item()
                
                if 'peak_kl_loss' in loss_dict:
                    if 'peak_kl_loss' not in epoch_losses:
                        epoch_losses['peak_kl_loss'] = 0.0
                    epoch_losses['peak_kl_loss'] += loss_dict['peak_kl_loss'].item()
                
                if 'total_static_loss' in loss_dict:
                    if 'static_loss' not in epoch_losses:
                        epoch_losses['static_loss'] = 0.0
                    epoch_losses['static_loss'] += loss_dict['total_static_loss'].item()
                
                if 'total_peak_loss' in loss_dict:
                    if 'peak_loss' not in epoch_losses:
                        epoch_losses['peak_loss'] = 0.0
                    epoch_losses['peak_loss'] += loss_dict['total_peak_loss'].item()
                
                # Debug tracking
                if self.debug_config.get('loss_component_tracking', False):
                    detailed_losses.append({
                        'batch': batch_idx,
                        'total': loss_dict['total_loss'].item(),
                        'recon': loss_dict.get('abr_recon_loss', torch.tensor(0.0)).item(),
                        'kl': loss_dict.get('kl_loss', torch.tensor(0.0)).item(),
                        'beta': current_beta
                    })
                
                # Update progress bar with correct values
                postfix_dict = {
                    'Loss': f"{loss_dict['total_loss'].item():.4f}",
                    'Recon': f"{loss_dict.get('abr_recon_loss', torch.tensor(0.0)).item():.4f}",
                    'KL': f"{loss_dict.get('kl_loss', torch.tensor(0.0)).item():.4f}",
                    'Beta': f"{current_beta:.6f}"
                }
                
                # Add advanced model specific losses
                if 'waveform_kl_loss' in loss_dict and 'peak_kl_loss' in loss_dict:
                    postfix_dict['WaveKL'] = f"{loss_dict['waveform_kl_loss'].item():.4f}"
                    postfix_dict['PeakKL'] = f"{loss_dict['peak_kl_loss'].item():.4f}"
                
                if 'total_static_loss' in loss_dict:
                    postfix_dict['Static'] = f"{loss_dict['total_static_loss'].item():.4f}"
                
                pbar.set_postfix(postfix_dict)
                
                # Log to tensorboard
                log_interval = self.config.get('training', {}).get('log_interval', 25)
                if batch_idx % log_interval == 0:
                    global_step = epoch * num_batches + batch_idx
                    self.writer.add_scalar('Train/BatchLoss', loss_dict['total_loss'].item(), global_step)
                    self.writer.add_scalar('Train/BatchRecon', loss_dict.get('abr_recon_loss', torch.tensor(0.0)).item(), global_step)
                    self.writer.add_scalar('Train/BatchKL', loss_dict.get('kl_loss', torch.tensor(0.0)).item(), global_step)
                    self.writer.add_scalar('Train/Beta', current_beta, global_step)
        
        # Average losses
        for key in epoch_losses:
            if key != 'beta':
                epoch_losses[key] /= num_batches
        
        # Save detailed losses for debugging
        if self.debug_config.get('loss_component_tracking', False):
            debug_path = self.output_dir / f'debug_losses_epoch_{epoch}.json'
            with open(debug_path, 'w') as f:
                json.dump(detailed_losses, f, indent=2)
        
        return epoch_losses
    
    def _check_for_nan(self, loss_dict: Dict[str, torch.Tensor], batch_idx: int, epoch: int):
        """Check for NaN values in loss components."""
        for key, value in loss_dict.items():
            if torch.isnan(value).any():
                logger.warning(f"NaN detected in {key} at epoch {epoch}, batch {batch_idx}")
                logger.warning(f"Loss dict: {loss_dict}")
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Enhanced validation epoch with better loss tracking."""
        self.model.eval()
        
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0
        }
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)
                    elif isinstance(batch[key], dict):
                        for subkey in batch[key]:
                            batch[key][subkey] = batch[key][subkey].to(self.device, non_blocking=True)
                
                # Forward pass
                loss_dict = self.model.compute_loss(batch)
                
                # Update epoch losses with correct key mapping
                epoch_losses['total_loss'] += loss_dict['total_loss'].item()
                epoch_losses['reconstruction_loss'] += loss_dict.get('abr_recon_loss', torch.tensor(0.0)).item()
                epoch_losses['kl_loss'] += loss_dict.get('kl_loss', torch.tensor(0.0)).item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', epoch_losses['total_loss'], epoch)
        self.writer.add_scalar('Val/ReconLoss', epoch_losses['reconstruction_loss'], epoch)
        self.writer.add_scalar('Val/KLLoss', epoch_losses['kl_loss'], epoch)
        
        return epoch_losses
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              epochs: Optional[int] = None,
              data_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Main training loop with comprehensive logging.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (overrides config)
            data_path: Path to training data for reproducibility tracking
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.get('training', {}).get('epochs', 100)
        
        # Start training run
        if self.run_id is None:
            self.start_training_run(data_path)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
            'train_kl_loss': [],
            'val_kl_loss': [],
            'beta_values': [],
            'epochs': []
        }
        
        # Early stopping
        patience = self.config.get('training', {}).get('early_stopping_patience', 20)
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Training run ID: {self.run_id}")
        
        try:
            for epoch in range(epochs):
                # Training
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Validation
                val_metrics = self.validate_epoch(val_loader, epoch)
                
                # Update learning rate
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['total_loss'])
                    else:
                        self.scheduler.step()
                
                # Store metrics
                history['epochs'].append(epoch)
                history['train_loss'].append(train_metrics['total_loss'])
                history['val_loss'].append(val_metrics['total_loss'])
                history['train_recon_loss'].append(train_metrics['reconstruction_loss'])
                history['val_recon_loss'].append(val_metrics['reconstruction_loss'])
                history['train_kl_loss'].append(train_metrics['kl_loss'])
                history['val_kl_loss'].append(val_metrics['kl_loss'])
                history['beta_values'].append(train_metrics['beta'])
                
                # Update training history logger
                additional_metrics = {
                    'train_recon_loss': train_metrics['reconstruction_loss'],
                    'val_recon_loss': val_metrics['reconstruction_loss'],
                    'train_kl_loss': train_metrics['kl_loss'],
                    'val_kl_loss': val_metrics['kl_loss'],
                    'beta': train_metrics['beta'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                self.history_logger.update_training_progress(
                    self.run_id, epoch, 
                    train_metrics['total_loss'], 
                    val_metrics['total_loss'],
                    additional_metrics
                )
                
                # Log epoch summary
                logger.info(f"Epoch {epoch+1}/{epochs}")
                logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f}")
                logger.info(f"  Val Loss: {val_metrics['total_loss']:.4f}")
                logger.info(f"  Beta: {train_metrics['beta']:.4f}")
                logger.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Check for improvement
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self.patience_counter = 0
                    
                    # Save best checkpoint
                    self.save_checkpoint(epoch, is_best=True)
                    logger.info(f"  ✅ New best model saved! (Val Loss: {self.best_val_loss:.4f})")
                else:
                    self.patience_counter += 1
                    logger.info(f"  No improvement ({self.patience_counter}/{patience})")
                
                # Early stopping
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break
                
                # Save checkpoint periodically
                if (epoch + 1) % self.config.get('training', {}).get('checkpoint_interval', 10) == 0:
                    self.save_checkpoint(epoch)
                
                # Generate samples periodically
                if (epoch + 1) % self.config.get('training', {}).get('sample_interval', 20) == 0:
                    self.generate_and_log_samples(val_loader, epoch)
                
                # Log training curves
                self.writer.add_scalar('Train/Loss', train_metrics['total_loss'], epoch)
                self.writer.add_scalar('Train/ReconLoss', train_metrics['reconstruction_loss'], epoch)
                self.writer.add_scalar('Train/KLLoss', train_metrics['kl_loss'], epoch)
                self.writer.add_scalar('Train/Beta', train_metrics['beta'], epoch)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Calculate training time
            training_time = time.time() - self.training_start_time if self.training_start_time else 0
            
            # Save final checkpoint
            final_epoch = len(history['epochs']) - 1 if history['epochs'] else 0
            self.save_checkpoint(final_epoch, is_final=True)
            
            # Prepare final metrics
            final_metrics = {
                'best_val_loss': self.best_val_loss,
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else float('inf'),
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
                'total_epochs': len(history['epochs']),
                'training_time': training_time
            }
            
            # Finish training run in history logger
            model_path = str(self.checkpoint_dir / 'best_checkpoint.pth')
            self.history_logger.finish_training_run(
                self.run_id,
                final_metrics,
                training_time,
                model_path=model_path if os.path.exists(model_path) else None
            )
            
            # Save training history
            self.save_training_history(history)
            
            # Plot training curves
            self.plot_training_curves(history)
            
            logger.info(f"Training completed!")
            logger.info(f"Run ID: {self.run_id}")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            logger.info(f"Total training time: {training_time/3600:.2f} hours")
        
        return history
    
    def generate_and_log_samples(self, val_loader: DataLoader, epoch: int):
        """Generate and log sample reconstructions and generations."""
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch from validation set
            batch = next(iter(val_loader))
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device, non_blocking=True)
                elif isinstance(batch[key], dict):
                    for subkey in batch[key]:
                        batch[key][subkey] = batch[key][subkey].to(self.device, non_blocking=True)
            
            # Get first 4 samples
            sample_batch = {}
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    sample_batch[key] = batch[key][:4]
                elif isinstance(batch[key], dict):
                    sample_batch[key] = {}
                    for subkey in batch[key]:
                        sample_batch[key][subkey] = batch[key][subkey][:4]
                else:
                    sample_batch[key] = batch[key]
            
            # Forward pass for reconstruction
            outputs = self.model(sample_batch)
            reconstructions = outputs['abr_reconstruction']
            
            # Generate new samples - generate enough samples for visualization
            num_viz_samples = min(4, original.shape[0])
            generated_outputs = self.model.generate(sample_batch, num_samples=num_viz_samples)
            generated = generated_outputs['abr_waveforms']
            
            # Get original data
            original = sample_batch['time_series']
            
            # Create visualization
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            
            for i in range(num_viz_samples):
                # Original
                if len(original.shape) == 3:  # [batch, seq, channels]
                    orig_data = original[i, :, 0].cpu().numpy()
                    recon_data = reconstructions[i, :, 0].cpu().numpy()
                    # Handle generated data shape - it might be [num_samples, seq] or [num_samples, seq, 1]
                    if len(generated.shape) == 3:
                        gen_data = generated[i, :, 0].cpu().numpy()
                    else:
                        gen_data = generated[i, :].cpu().numpy()
                else:  # [batch, seq]
                    orig_data = original[i, :].cpu().numpy()
                    recon_data = reconstructions[i, :].cpu().numpy()
                    gen_data = generated[i, :].cpu().numpy()
                
                axes[0, i].plot(orig_data)
                axes[0, i].set_title(f'Original {i+1}')
                axes[0, i].set_ylabel('Amplitude')
                
                # Reconstruction
                axes[1, i].plot(recon_data)
                axes[1, i].set_title(f'Reconstruction {i+1}')
                axes[1, i].set_ylabel('Amplitude')
                
                # Generated
                axes[2, i].plot(gen_data)
                axes[2, i].set_title(f'Generated {i+1}')
                axes[2, i].set_ylabel('Amplitude')
                axes[2, i].set_xlabel('Time')
            
            plt.tight_layout()
            
            # Save and log to tensorboard
            sample_path = self.results_dir / f'samples_epoch_{epoch+1}.png'
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            self.writer.add_figure('Samples', fig, epoch)
            plt.close()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint with unique identifiers."""
        # Create timestamp for this training session if not exists
        if not hasattr(self, '_training_timestamp'):
            self._training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create unique identifier for this training run
        run_identifier = f"{self._training_timestamp}_{self.run_id}" if self.run_id else self._training_timestamp
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'run_id': self.run_id,
            'training_timestamp': self._training_timestamp,
            'run_identifier': run_identifier
        }
        
        # Create run-specific checkpoint directory
        run_checkpoint_dir = self.checkpoint_dir / f"run_{run_identifier}"
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = run_checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model for this run
        if is_best:
            best_path = run_checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            
            # Also save in main checkpoint dir for backward compatibility
            main_best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, main_best_path)
            
            logger.info(f"New best model saved for run {run_identifier} with validation loss: {self.best_val_loss:.4f}")
        
        # Save final model for this run
        if is_final:
            final_path = run_checkpoint_dir / 'final_model.pth'
            torch.save(checkpoint, final_path)
            
            # Also save in main checkpoint dir for backward compatibility
            main_final_path = self.checkpoint_dir / 'final_model.pth'
            torch.save(checkpoint, main_final_path)
            
            # Save run summary
            self._save_run_summary(run_checkpoint_dir, run_identifier)
    
    def _save_run_summary(self, run_dir: Path, run_identifier: str):
        """Save a summary of this training run."""
        summary = {
            'run_id': self.run_id,
            'run_identifier': run_identifier,
            'training_timestamp': self._training_timestamp,
            'best_val_loss': self.best_val_loss,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'config': self.config,
            'notes': self.notes,
            'training_duration': time.time() - self.training_start_time if self.training_start_time else None
        }
        
        summary_path = run_dir / 'run_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Run summary saved to {summary_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']
    
    def save_training_history(self, history: Dict[str, List[float]]):
        """Save training history."""
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves(history)
    
    def plot_training_curves(self, history: Dict[str, List[float]]):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(history['train_loss'], label='Train')
        axes[0, 0].plot(history['val_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        axes[0, 1].plot(history['train_recon_loss'], label='Train')
        axes[0, 1].plot(history['val_recon_loss'], label='Validation')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # KL loss
        axes[1, 0].plot(history['train_kl_loss'], label='Train')
        axes[1, 0].plot(history['val_kl_loss'], label='Validation')
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Beta schedule
        epochs = len(history['train_loss'])
        betas = [self.beta_scheduler.get_beta(epoch) for epoch in range(epochs)]
        axes[1, 1].plot(betas)
        axes[1, 1].set_title('Beta Schedule (KL Annealing)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Beta')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        curves_path = self.results_dir / 'training_curves.png'
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close()

class CustomBetaScheduler:
    """Custom beta scheduler with predefined schedule to prevent KL collapse."""
    
    def __init__(self, schedule: List[Dict[str, float]]):
        """
        Initialize custom beta scheduler.
        
        Args:
            schedule: List of {"epoch": int, "beta": float} dictionaries
        """
        self.schedule = sorted(schedule, key=lambda x: x['epoch'])
    
    def get_beta(self, epoch: int) -> float:
        """Get beta value for current epoch."""
        if not self.schedule:
            return 0.0
        
        # Find the appropriate beta value
        for i, point in enumerate(self.schedule):
            if epoch <= point['epoch']:
                if i == 0:
                    return point['beta']
                else:
                    # Linear interpolation between points
                    prev_point = self.schedule[i-1]
                    progress = (epoch - prev_point['epoch']) / (point['epoch'] - prev_point['epoch'])
                    return prev_point['beta'] + progress * (point['beta'] - prev_point['beta'])
        
        # If epoch is beyond last point, use last beta value
        return self.schedule[-1]['beta']

class BetaScheduler:
    """Enhanced beta scheduler with better control."""
    
    def __init__(self, initial_beta: float = 0.0, final_beta: float = 0.01, annealing_epochs: int = 150):
        """Initialize beta scheduler with very conservative settings."""
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.annealing_epochs = annealing_epochs
    
    def get_beta(self, epoch: int) -> float:
        """Get beta value for current epoch with very gradual increase."""
        if epoch >= self.annealing_epochs:
            return self.final_beta
        
        # Very gradual linear increase
        progress = epoch / self.annealing_epochs
        return self.initial_beta + progress * (self.final_beta - self.initial_beta) 