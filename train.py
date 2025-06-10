#!/usr/bin/env python3
"""
ABR CVAE Training Script
=======================

Enhanced training script for the Conditional Variational Autoencoder on ABR data
with comprehensive logging, generation capabilities, and clinical parameter tracking.

Usage:
    python train.py [--config config.yaml] [--output-dir outputs] [--device cuda] [--notes "experiment notes"]

Example:
    python train.py --config configs/cvae_config.yaml --device cuda --notes "Testing new beta annealing"
"""

import argparse
import logging
import json
import yaml
from pathlib import Path
import sys

import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from models.cvae_model import ConditionalVAEWithMasking
from models.advanced_cvae_model import AdvancedConditionalVAE
from data.dataset import ABRMaskedDataset, create_abr_datasets, create_abr_dataloaders
from training.trainer import CVAETrainer
# from src.utils.training_logger import get_training_logger  # Not needed here

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON or YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)


def create_datasets_and_loaders(config: dict):
    """Create datasets and data loaders using existing functions."""
    try:
        # Extract data configuration
        data_config = config.get('data', {})
        training_config = config.get('training', {})
        
        # Create datasets using existing function
        train_dataset, val_dataset, test_dataset = create_abr_datasets(
            data_dir=config.get('data_dir', 'data/processed'),
            train_split=data_config.get('train_split', 0.7),
            val_split=data_config.get('val_split', 0.15),
            test_split=data_config.get('test_split', 0.15),
            random_seed=data_config.get('random_state', 42),
            augment_train=data_config.get('augment_train', True),
            augmentation_params={
                'noise_std': data_config.get('noise_std', 0.01),
                'time_shift_max': data_config.get('time_shift_max', 10)
            }
        )
        
        # Create data loaders using existing function
        train_loader, val_loader, test_loader = create_abr_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=training_config.get('batch_size', 8),
            num_workers=config.get('num_workers', 0),
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, "data/processed"
        
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        logger.error(f"Make sure preprocessed data exists in data/processed/")
        sys.exit(1)


def create_model(config: dict, device: torch.device):
    """Create the CVAE model (original or advanced)."""
    try:
        # Extract model configuration
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        training_config = config.get('training', {})
        
        # Model selection
        model_type = model_config.get('type', 'original').lower()
        
        if model_type == 'advanced':
            logger.info("Creating Advanced CVAE model with hierarchical latents and conditional priors...")
            model = AdvancedConditionalVAE(
                static_dim=model_config.get('static_dim', 4),
                masked_features_dim=64,  # Fixed for latency/amplitude features
                waveform_latent_dim=model_config.get('waveform_latent_dim', 64),
                peak_latent_dim=model_config.get('peak_latent_dim', 32),
                condition_dim=model_config.get('condition_dim', 128),
                hidden_dim=model_config.get('hidden_dim', 256),
                num_encoder_layers=model_config.get('num_encoder_layers', 4),
                num_decoder_layers=model_config.get('num_decoder_layers', 6),
                num_heads=model_config.get('num_heads', 8),
                dropout=model_config.get('dropout', 0.1),
                sequence_length=data_config.get('sequence_length', 200),
                beta_waveform=training_config.get('beta_waveform', 1.0),
                beta_peak=training_config.get('beta_peak', 0.5),
                use_conditional_prior=model_config.get('use_conditional_prior', True),
                noise_augmentation=model_config.get('noise_augmentation', 0.1)
            ).to(device)
        else:
            logger.info("Creating Original CVAE model...")
            model = ConditionalVAEWithMasking(
                static_dim=model_config.get('static_dim', 4),
                masked_features_dim=64,  # Fixed for latency/amplitude features
                latent_dim=model_config.get('latent_dim', 128),
                condition_dim=model_config.get('condition_dim', 128),
                hidden_dim=model_config.get('hidden_dim', 256),
                num_encoder_layers=model_config.get('num_encoder_layers', 4),
                num_decoder_layers=model_config.get('num_decoder_layers', 4),
                num_heads=model_config.get('num_heads', 8),
                dropout=model_config.get('dropout', 0.1),
                sequence_length=data_config.get('sequence_length', 200),
                beta=training_config.get('initial_beta', 0.0),
                reconstruct_masked_features=True
            ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created successfully!")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size: {total_params * 4 / 1e6:.1f} MB")
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        sys.exit(1)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ABR CVAE Model with Enhanced Features')
    parser.add_argument('--config', type=str, default='configs/cvae_config.yaml',
                       help='Path to configuration file (JSON or YAML)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/mps/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--notes', type=str, default=None,
                       help='Notes about this training run')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--check-history', action='store_true',
                       help='Show training history and exit')
    parser.add_argument('--best', type=int, default=5,
                       help='Number of best runs to show (used with --check-history)')
    parser.add_argument('--model', type=str, default='original', choices=['original', 'advanced'],
                       help='Model type to use (original or advanced)')
    parser.add_argument('--use-conditional-prior', action='store_true',
                       help='Use conditional prior for advanced model (overrides config)')
    parser.add_argument('--noise-augmentation', type=float, default=None,
                       help='Noise augmentation level for advanced model (overrides config)')
    
    args = parser.parse_args()
    
    # Check training history if requested
    if args.check_history:
        from src.utils.training_logger import TrainingHistoryLogger
        history_logger = TrainingHistoryLogger()
        
        # Get and display training history
        runs_df = history_logger.list_all_runs()
        if len(runs_df) == 0:
            print("No training runs found.")
            return
        
        # Sort by validation loss and show best runs
        if 'best_val_loss' in runs_df.columns:
            runs_df = runs_df.sort_values('best_val_loss')
        
        print(f"\n{'='*80}")
        print("ABR CVAE TRAINING HISTORY")
        print(f"{'='*80}")
        
        # Show top runs
        top_runs = runs_df.head(args.best)
        print(top_runs.to_string(index=False))
        
        print(f"\nTotal runs: {len(runs_df)}")
        if 'best_val_loss' in runs_df.columns:
            best_loss = runs_df['best_val_loss'].min()
            best_run = runs_df.loc[runs_df['best_val_loss'].idxmin(), 'run_id']
            print(f"Best validation loss: {best_loss:.4f} (Run: {best_run})")
        
        print(f"\nQuick commands:")
        print(f"  python evaluate.py --test-generation  # Test enhanced generation")
        print(f"  python train.py --epochs 50 --notes 'Longer training'  # Continue training")
        
        return
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"🚀 Starting ABR CVAE Training")
    logger.info(f"Device: {device}")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        if 'training' not in config:
            config['training'] = {}
        if 'optimizer' not in config['training']:
            config['training']['optimizer'] = {}
        config['training']['optimizer']['lr'] = args.learning_rate
    
    # Model selection and advanced model options
    if 'model' not in config:
        config['model'] = {}
    config['model']['type'] = args.model
    
    if args.model == 'advanced':
        if args.use_conditional_prior:
            config['model']['use_conditional_prior'] = True
        if args.noise_augmentation is not None:
            config['model']['noise_augmentation'] = args.noise_augmentation
    
    # Add device to config
    config['device'] = str(device)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save updated config
    config_save_path = output_path / 'config.json'
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to: {config_save_path}")
    
    # Create datasets and data loaders
    logger.info("📊 Creating datasets and data loaders...")
    train_loader, val_loader, test_loader, data_path = create_datasets_and_loaders(config)
    
    # Create model
    logger.info("🏗️ Creating enhanced CVAE model...")
    model = create_model(config, device)
    
    # Create trainer with enhanced logging
    logger.info("🎯 Creating trainer with comprehensive logging...")
    trainer = CVAETrainer(
        model=model,
        config=config,
        device=device,
        use_m1_optimizations=True,
        notes=args.notes
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"📂 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Display training information
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Epochs: {training_config.get('epochs', 100)}")
    logger.info(f"Batch Size: {training_config.get('batch_size', 32)}")
    logger.info(f"Learning Rate: {training_config.get('optimizer', {}).get('lr', 1e-3)}")
    logger.info(f"Latent Dimension: {model_config.get('latent_dim', 128)}")
    logger.info(f"Sequence Length: {data_config.get('sequence_length', 200)}")
    logger.info(f"Beta Annealing: {training_config.get('initial_beta', 0.0)} → {training_config.get('final_beta', 1.0)}")
    if args.notes:
        logger.info(f"Notes: {args.notes}")
    logger.info("=" * 80)
    
    # Train model with comprehensive logging
    logger.info("🚀 Starting enhanced training with automatic logging...")
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            data_path=data_path
        )
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📊 Training run ID: {trainer.run_id}")
        logger.info(f"💾 Results saved to: {output_path}")
        
        # Display final results
        if history:
            best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
            total_epochs = len(history['epochs']) if history['epochs'] else 0
            
            logger.info("=" * 80)
            logger.info("FINAL RESULTS")
            logger.info("=" * 80)
            logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
            logger.info(f"Total Epochs: {total_epochs}")
            logger.info(f"Training Run ID: {trainer.run_id}")
            logger.info("=" * 80)
        
        # Show how to check training history
        logger.info("\n📈 To check training history and compare runs:")
        logger.info("   python train.py --check-history")
        logger.info("   python train.py --check-history --best 10")
        logger.info("   python evaluate.py --test-generation  # Test enhanced generation")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 