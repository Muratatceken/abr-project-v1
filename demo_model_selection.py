#!/usr/bin/env python3
"""
ABR CVAE Model Selection Demo
============================

This script demonstrates how to use both the original and advanced CVAE models
for ABR data generation and analysis.

Usage Examples:
    # Train original CVAE model
    python demo_model_selection.py --action train --model original --epochs 50
    
    # Train advanced CVAE model with conditional priors
    python demo_model_selection.py --action train --model advanced --epochs 100 --use-conditional-prior
    
    # Evaluate a trained model
    python demo_model_selection.py --action evaluate --model-path outputs/best_checkpoint.pth
    
    # Compare both models
    python demo_model_selection.py --action compare --original-path outputs/original_model.pth --advanced-path outputs/advanced_model.pth
"""

import argparse
import logging
import json
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(model_type: str, epochs: int = 100, use_conditional_prior: bool = False, 
                noise_augmentation: float = 0.1, notes: str = None):
    """Train a model with the specified configuration."""
    
    logger.info(f"🚀 Training {model_type.upper()} CVAE model")
    
    if model_type == 'original':
        config_file = 'configs/cvae_config.yaml'
        cmd_args = [
            'python', 'train.py',
            '--config', config_file,
            '--model', 'original',
            '--epochs', str(epochs),
            '--output-dir', f'outputs_{model_type}',
        ]
        
        if notes:
            cmd_args.extend(['--notes', f'Original CVAE: {notes}'])
        else:
            cmd_args.extend(['--notes', 'Original CVAE training'])
            
    elif model_type == 'advanced':
        config_file = 'configs/advanced_cvae_config.yaml'
        cmd_args = [
            'python', 'train.py',
            '--config', config_file,
            '--model', 'advanced',
            '--epochs', str(epochs),
            '--output-dir', f'outputs_{model_type}',
        ]
        
        if use_conditional_prior:
            cmd_args.append('--use-conditional-prior')
            
        if noise_augmentation != 0.1:
            cmd_args.extend(['--noise-augmentation', str(noise_augmentation)])
            
        if notes:
            cmd_args.extend(['--notes', f'Advanced CVAE: {notes}'])
        else:
            features = []
            if use_conditional_prior:
                features.append('conditional priors')
            if noise_augmentation > 0:
                features.append(f'noise aug {noise_augmentation}')
            feature_str = ', '.join(features) if features else 'default settings'
            cmd_args.extend(['--notes', f'Advanced CVAE with {feature_str}'])
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Training command: {' '.join(cmd_args)}")
    logger.info("=" * 60)
    
    # Import and run training
    import subprocess
    result = subprocess.run(cmd_args, capture_output=False)
    
    if result.returncode == 0:
        logger.info(f"✅ {model_type.upper()} model training completed successfully!")
    else:
        logger.error(f"❌ {model_type.upper()} model training failed!")
        
    return result.returncode == 0

def evaluate_model(model_path: str, comprehensive: bool = False, test_generation: bool = True):
    """Evaluate a trained model."""
    
    logger.info(f"🔍 Evaluating model: {model_path}")
    
    cmd_args = [
        'python', 'evaluate.py',
        '--model', model_path,
        '--output-dir', 'evaluation_results',
    ]
    
    if comprehensive:
        cmd_args.append('--comprehensive')
        
    if test_generation:
        cmd_args.append('--test-generation')
    
    logger.info(f"Evaluation command: {' '.join(cmd_args)}")
    logger.info("=" * 60)
    
    # Import and run evaluation
    import subprocess
    result = subprocess.run(cmd_args, capture_output=False)
    
    if result.returncode == 0:
        logger.info("✅ Model evaluation completed successfully!")
    else:
        logger.error("❌ Model evaluation failed!")
        
    return result.returncode == 0

def compare_models(original_path: str, advanced_path: str):
    """Compare original and advanced models."""
    
    logger.info("🔬 Comparing Original vs Advanced CVAE models")
    logger.info("=" * 60)
    
    # Evaluate both models
    logger.info("📊 Evaluating Original CVAE model...")
    original_success = evaluate_model(original_path, comprehensive=True, test_generation=True)
    
    logger.info("📊 Evaluating Advanced CVAE model...")
    advanced_success = evaluate_model(advanced_path, comprehensive=True, test_generation=True)
    
    if original_success and advanced_success:
        logger.info("✅ Both models evaluated successfully!")
        logger.info("📈 Check the evaluation_results directory for detailed comparisons")
    else:
        logger.warning("⚠️ Some evaluations failed. Check the logs above.")

def show_training_history():
    """Show training history for all models."""
    
    logger.info("📈 Showing training history...")
    
    import subprocess
    result = subprocess.run(['python', 'train.py', '--check-history'], capture_output=False)
    
    return result.returncode == 0

def quick_demo():
    """Run a quick demonstration of both models."""
    
    logger.info("🎯 Running Quick Demo of ABR CVAE Models")
    logger.info("=" * 60)
    
    # Train original model (short)
    logger.info("1️⃣ Training Original CVAE (10 epochs)...")
    original_success = train_model('original', epochs=10, notes='Quick demo')
    
    # Train advanced model (short)
    logger.info("2️⃣ Training Advanced CVAE (10 epochs)...")
    advanced_success = train_model('advanced', epochs=10, use_conditional_prior=True, notes='Quick demo')
    
    if original_success and advanced_success:
        logger.info("3️⃣ Both models trained! Check outputs_original and outputs_advanced directories")
        logger.info("4️⃣ You can now evaluate them using:")
        logger.info("   python demo_model_selection.py --action evaluate --model-path outputs_original/checkpoints/best_model.pth")
        logger.info("   python demo_model_selection.py --action evaluate --model-path outputs_advanced/checkpoints/best_model.pth")
    else:
        logger.warning("⚠️ Some training failed. Check the logs above.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='ABR CVAE Model Selection Demo')
    parser.add_argument('--action', type=str, required=True,
                       choices=['train', 'evaluate', 'compare', 'history', 'demo'],
                       help='Action to perform')
    
    # Training arguments
    parser.add_argument('--model', type=str, choices=['original', 'advanced'],
                       help='Model type to train (required for train action)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--use-conditional-prior', action='store_true',
                       help='Use conditional prior for advanced model')
    parser.add_argument('--noise-augmentation', type=float, default=0.1,
                       help='Noise augmentation level for advanced model')
    parser.add_argument('--notes', type=str, default=None,
                       help='Training notes')
    
    # Evaluation arguments
    parser.add_argument('--model-path', type=str,
                       help='Path to model checkpoint (required for evaluate action)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive evaluation')
    parser.add_argument('--test-generation', action='store_true', default=True,
                       help='Test generation capabilities')
    
    # Comparison arguments
    parser.add_argument('--original-path', type=str,
                       help='Path to original model checkpoint (required for compare action)')
    parser.add_argument('--advanced-path', type=str,
                       help='Path to advanced model checkpoint (required for compare action)')
    
    args = parser.parse_args()
    
    # Validate arguments based on action
    if args.action == 'train':
        if not args.model:
            parser.error("--model is required for train action")
            
    elif args.action == 'evaluate':
        if not args.model_path:
            parser.error("--model-path is required for evaluate action")
            
    elif args.action == 'compare':
        if not args.original_path or not args.advanced_path:
            parser.error("--original-path and --advanced-path are required for compare action")
    
    # Execute action
    try:
        if args.action == 'train':
            success = train_model(
                model_type=args.model,
                epochs=args.epochs,
                use_conditional_prior=args.use_conditional_prior,
                noise_augmentation=args.noise_augmentation,
                notes=args.notes
            )
            sys.exit(0 if success else 1)
            
        elif args.action == 'evaluate':
            success = evaluate_model(
                model_path=args.model_path,
                comprehensive=args.comprehensive,
                test_generation=args.test_generation
            )
            sys.exit(0 if success else 1)
            
        elif args.action == 'compare':
            compare_models(args.original_path, args.advanced_path)
            
        elif args.action == 'history':
            success = show_training_history()
            sys.exit(0 if success else 1)
            
        elif args.action == 'demo':
            quick_demo()
            
    except KeyboardInterrupt:
        logger.info("⏹️ Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🧬 ABR CVAE Model Selection Demo")
    print("=" * 40)
    print("Available Models:")
    print("  • Original CVAE: Standard conditional VAE with masking")
    print("  • Advanced CVAE: Hierarchical latents + conditional priors + FiLM + temporal decoder")
    print()
    print("Quick Start:")
    print("  python demo_model_selection.py --action demo")
    print()
    
    main() 