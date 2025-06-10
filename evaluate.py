#!/usr/bin/env python3
"""
ABR CVAE Evaluation Script
==========================

Main script for evaluating the trained Conditional Variational Autoencoder.

Usage:
    python evaluate.py [--model outputs/best_checkpoint.pth] [--output-dir results]

Example:
    python evaluate.py --model outputs/best_checkpoint.pth --comprehensive
"""

import argparse
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from src.models.cvae_model import ConditionalVAEWithMasking
from src.models.advanced_cvae_model import AdvancedConditionalVAE
from src.data.dataset import ABRMaskedDataset
from src.evaluation import CVAEEvaluator, ABRSyntheticDataEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_model_identifier(checkpoint_path: str, checkpoint: dict) -> str:
    """Extract unique identifier for the model being evaluated."""
    # Try to get run identifier from checkpoint
    if 'run_identifier' in checkpoint:
        return checkpoint['run_identifier']
    elif 'run_id' in checkpoint and 'training_timestamp' in checkpoint:
        return f"{checkpoint['training_timestamp']}_{checkpoint['run_id']}"
    elif 'training_timestamp' in checkpoint:
        return checkpoint['training_timestamp']
    else:
        # Fallback: use model file timestamp or current timestamp
        model_path = Path(checkpoint_path)
        if model_path.exists():
            # Use file modification time
            timestamp = datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
            return f"model_{timestamp}"
        else:
            # Use current timestamp
            return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_evaluation_output_dir(base_output_dir: str, model_identifier: str) -> Path:
    """Create unique output directory for this evaluation."""
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir_name = f"eval_{eval_timestamp}_{model_identifier}"
    
    output_path = Path(base_output_dir) / eval_dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path


def save_evaluation_metadata(output_dir: Path, model_path: str, checkpoint: dict, args):
    """Save metadata about this evaluation run."""
    metadata = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'model_identifier': extract_model_identifier(model_path, checkpoint),
        'evaluation_args': vars(args),
        'model_info': {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
            'run_id': checkpoint.get('run_id', 'unknown'),
            'training_timestamp': checkpoint.get('training_timestamp', 'unknown'),
            'total_parameters': sum(p.numel() for p in checkpoint['model_state_dict'].values()) if 'model_state_dict' in checkpoint else 'unknown'
        }
    }
    
    metadata_path = output_dir / 'evaluation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Evaluation metadata saved to {metadata_path}")


def test_enhanced_generation_capabilities(model, output_dir: Path, num_samples: int = 5):
    """Test the enhanced generation capabilities of the CVAE model."""
    logger.info(f"\n🧪 Testing Enhanced Generation Capabilities")
    logger.info("=" * 60)
    
    # Create test conditions
    test_conditions = [
        {"age": 25, "intensity": 80, "stimulus_rate": 11.1, "hear_loss": 0},
        {"age": 45, "intensity": 90, "stimulus_rate": 21.1, "hear_loss": 1},
        {"age": 65, "intensity": 100, "stimulus_rate": 11.1, "hear_loss": 2},
        {"age": 35, "intensity": 70, "stimulus_rate": 41.1, "hear_loss": 0},
        {"age": 55, "intensity": 85, "stimulus_rate": 21.1, "hear_loss": 1},
    ]
    
    results = []
    
    with torch.no_grad():
        for i, condition in enumerate(test_conditions[:num_samples]):
            logger.info(f"\n📋 Test Sample {i+1}")
            logger.info(f"   Condition: Age={condition['age']}, Intensity={condition['intensity']}, "
                      f"Rate={condition['stimulus_rate']}, Hearing Loss={condition['hear_loss']}")
            
            # Generate without conditioning (pure generation mode)
            generation_output = model.generate(
                batch=None,
                num_samples=1,
                use_conditioning=False
            )
            
            logger.info(f"   ✅ Generation successful!")
            logger.info(f"   📊 Generated components:")
            for key, value in generation_output.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"      - {key}: shape {value.shape}")
                else:
                    logger.info(f"      - {key}: {type(value)}")
            
            # Extract and display generated parameters
            if 'generated_static_params' in generation_output:
                gen_static = generation_output['generated_static_params']
                logger.info(f"   🎯 Generated Static Parameters:")
                logger.info(f"      Age: {gen_static['age'][0]:.1f}, Intensity: {gen_static['intensity'][0]:.1f}")
                logger.info(f"      Rate: {gen_static['stimulus_rate'][0]:.1f}, Hearing Loss: {gen_static['hear_loss'][0]:.1f}")
            
            if 'generated_latencies' in generation_output:
                latencies = generation_output['generated_latencies'][0].cpu().numpy()
                logger.info(f"   ⏱️  Generated Latencies (ms):")
                logger.info(f"      Wave I: {latencies[0]:.2f}, Wave III: {latencies[1]:.2f}, Wave V: {latencies[2]:.2f}")
            
            if 'generated_amplitudes' in generation_output:
                amplitudes = generation_output['generated_amplitudes'][0].cpu().numpy()
                logger.info(f"   📈 Generated Amplitudes (μV):")
                logger.info(f"      Wave I: {amplitudes[0]:.3f}, Wave III: {amplitudes[1]:.3f}, Wave V: {amplitudes[2]:.3f}")
            
            # Store results
            result = {
                'sample_id': i+1,
                'input_age': condition['age'],
                'input_intensity': condition['intensity'],
                'input_rate': condition['stimulus_rate'],
                'input_hear_loss': condition['hear_loss'],
            }
            
            if 'generated_static_params' in generation_output:
                gen_static = generation_output['generated_static_params']
                result.update({
                    'gen_age': gen_static['age'][0].item(),
                    'gen_intensity': gen_static['intensity'][0].item(),
                    'gen_rate': gen_static['stimulus_rate'][0].item(),
                    'gen_hear_loss': gen_static['hear_loss'][0].item(),
                })
            
            if 'generated_latencies' in generation_output:
                latencies = generation_output['generated_latencies'][0].cpu().numpy()
                result.update({
                    'gen_lat_I': latencies[0],
                    'gen_lat_III': latencies[1],
                    'gen_lat_V': latencies[2],
                })
            
            if 'generated_amplitudes' in generation_output:
                amplitudes = generation_output['generated_amplitudes'][0].cpu().numpy()
                result.update({
                    'gen_amp_I': amplitudes[0],
                    'gen_amp_III': amplitudes[1],
                    'gen_amp_V': amplitudes[2],
                })
            
            results.append(result)
    
    # Test peak detection capabilities
    logger.info(f"\n🔍 Testing Peak Detection Capabilities")
    logger.info("=" * 60)
    
    peak_test_conditions = test_conditions[:3]  # Test with first 3 conditions
    
    with torch.no_grad():
        for i, condition in enumerate(peak_test_conditions):
            logger.info(f"\n📋 Peak Detection Test {i+1}")
            logger.info(f"   Condition: Age={condition['age']}, Intensity={condition['intensity']}")
            
            # Create a dummy batch for peak detection testing
            dummy_batch = {
                'static_params': {
                    'age': torch.tensor([condition['age']], dtype=torch.float32),
                    'intensity': torch.tensor([condition['intensity']], dtype=torch.float32),
                    'stimulus_rate': torch.tensor([condition['stimulus_rate']], dtype=torch.float32),
                    'hear_loss': torch.tensor([condition['hear_loss']], dtype=torch.float32)
                },
                'latency_data': {
                    'I Latancy': torch.zeros(1),
                    'III Latancy': torch.zeros(1),
                    'V Latancy': torch.zeros(1)
                },
                'latency_masks': {
                    'I Latancy': torch.zeros(1),
                    'III Latancy': torch.zeros(1),
                    'V Latancy': torch.zeros(1)
                },
                'amplitude_data': {
                    'I Amplitude': torch.zeros(1),
                    'III Amplitude': torch.zeros(1),
                    'V Amplitude': torch.zeros(1)
                },
                'amplitude_masks': {
                    'I Amplitude': torch.zeros(1),
                    'III Amplitude': torch.zeros(1),
                    'V Amplitude': torch.zeros(1)
                }
            }
            
            # Generate with peak detection
            try:
                generation_output = model.generate_with_peak_detection(
                    batch=dummy_batch,
                    num_samples=1
                )
                
                logger.info(f"   ✅ Peak detection successful!")
                
                if 'extracted_latencies' in generation_output:
                    det_lat = generation_output['extracted_latencies'][0].cpu().numpy()
                    logger.info(f"   🎯 Extracted Latencies (ms):")
                    logger.info(f"      Wave I: {det_lat[0]:.2f}, Wave III: {det_lat[1]:.2f}, Wave V: {det_lat[2]:.2f}")
                
                if 'extracted_amplitudes' in generation_output:
                    det_amp = generation_output['extracted_amplitudes'][0].cpu().numpy()
                    logger.info(f"   📊 Extracted Amplitudes (μV):")
                    logger.info(f"      Wave I: {det_amp[0]:.3f}, Wave III: {det_amp[1]:.3f}, Wave V: {det_amp[2]:.3f}")
                
                if 'peak_confidence' in generation_output:
                    conf = generation_output['peak_confidence'][0].cpu().numpy()
                    logger.info(f"   🎯 Peak Confidence Scores:")
                    logger.info(f"      Wave I: {conf[0]:.3f}, Wave III: {conf[1]:.3f}, Wave V: {conf[2]:.3f}")
                
            except Exception as e:
                logger.info(f"   ❌ Peak detection failed: {e}")
    
    # Create and save summary
    if results:
        logger.info(f"\n📊 Generation Results Summary")
        logger.info("=" * 80)
        
        df = pd.DataFrame(results)
        
        # Display input vs generated static parameters
        if all(col in df.columns for col in ['input_age', 'gen_age']):
            logger.info(f"\n🎯 Static Parameters (Input vs Generated):")
            logger.info("-" * 50)
            for _, row in df.iterrows():
                logger.info(f"Sample {row['sample_id']}:")
                logger.info(f"  Age: {row['input_age']:.1f} → {row.get('gen_age', 'N/A'):.1f}")
                logger.info(f"  Intensity: {row['input_intensity']:.1f} → {row.get('gen_intensity', 'N/A'):.1f}")
                logger.info(f"  Rate: {row['input_rate']:.1f} → {row.get('gen_rate', 'N/A'):.1f}")
                logger.info(f"  Hearing Loss: {row['input_hear_loss']:.1f} → {row.get('gen_hear_loss', 'N/A'):.1f}")
        
        # Display generated clinical parameters
        if all(col in df.columns for col in ['gen_lat_I', 'gen_lat_III', 'gen_lat_V']):
            logger.info(f"\n⏱️  Generated Latencies (ms):")
            logger.info("-" * 30)
            for _, row in df.iterrows():
                logger.info(f"Sample {row['sample_id']}: I={row['gen_lat_I']:.2f}, III={row['gen_lat_III']:.2f}, V={row['gen_lat_V']:.2f}")
        
        if all(col in df.columns for col in ['gen_amp_I', 'gen_amp_III', 'gen_amp_V']):
            logger.info(f"\n📈 Generated Amplitudes (μV):")
            logger.info("-" * 30)
            for _, row in df.iterrows():
                logger.info(f"Sample {row['sample_id']}: I={row['gen_amp_I']:.3f}, III={row['gen_amp_III']:.3f}, V={row['gen_amp_V']:.3f}")
        
        # Save to CSV
        output_file = output_dir / "generation_test_results.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\n💾 Results saved to: {output_file}")
        
        return df
    
    return None


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate ABR CVAE Model')
    parser.add_argument('--model', type=str, default='outputs/checkpoints/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='outputs/config.json',
                       help='Path to model configuration file')
    parser.add_argument('--data', type=str, default='data/processed/abr_processed_data.pkl',
                       help='Path to processed data file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Base output directory for evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive synthetic data quality evaluation')
    parser.add_argument('--test-generation', action='store_true',
                       help='Test enhanced generation capabilities (static params, clinical features, peak detection)')
    parser.add_argument('--n-samples', type=int, default=100,
                       help='Number of samples to evaluate')
    parser.add_argument('--generation-samples', type=int, default=5,
                       help='Number of samples for generation testing')
    parser.add_argument('--model-type', type=str, default='auto', choices=['auto', 'original', 'advanced'],
                       help='Model type to use (auto detects from config, or override)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for generation sampling (advanced model only)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Load checkpoint first to extract model identifier
    logger.info(f"Loading model checkpoint from {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Extract model identifier and create unique output directory
    model_identifier = extract_model_identifier(args.model, checkpoint)
    output_path = create_evaluation_output_dir(args.output_dir, model_identifier)
    
    logger.info(f"Model identifier: {model_identifier}")
    logger.info(f"Evaluation results will be saved to: {output_path}")
    
    # Save evaluation metadata
    save_evaluation_metadata(output_path, args.model, checkpoint, args)
    
    # Load and create model based on type
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    # Override model type if specified
    if args.model_type != 'auto':
        model_type = args.model_type.lower()
    else:
        model_type = model_config.get('type', 'original').lower()
    
    if model_type == 'advanced':
        logger.info("Creating Advanced CVAE model for evaluation...")
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
        logger.info("Creating Original CVAE model for evaluation...")
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
            reconstruct_masked_features=True,
            generate_static_params=True
        ).to(device)
    
    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded successfully! Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load test data
    logger.info(f"Loading test data from {args.data}")
    with open(args.data, 'rb') as f:
        processed_data = pickle.load(f)
    
    # Create test dataset (use last 10% as test)
    data_config = config.get('data', {})
    
    # The processed_data is a dictionary with arrays, we need to slice each array
    total_samples = len(processed_data['time_series'])
    test_size = int(total_samples * 0.1)
    test_start = total_samples - test_size
    
    # Helper function to slice data recursively
    def slice_data(data, start_idx):
        if hasattr(data, 'shape') and hasattr(data, '__getitem__'):
            return data[start_idx:]
        elif isinstance(data, dict):
            return {k: slice_data(v, start_idx) for k, v in data.items()}
        else:
            return data
    
    # Create test data by slicing each array
    test_data = {
        'time_series': processed_data['time_series'][test_start:],
        'static_params': slice_data(processed_data['static_params'], test_start),
        'latency_data': slice_data(processed_data['latency_data'], test_start),
        'amplitude_data': slice_data(processed_data['amplitude_data'], test_start),
        'masks': slice_data(processed_data['masks'], test_start),
        'metadata': processed_data['metadata']
    }
    
    test_dataset = ABRMaskedDataset(
        time_series=test_data['time_series'],
        static_params=test_data['static_params'],
        latency_data=test_data['latency_data'],
        amplitude_data=test_data['amplitude_data'],
        masks=test_data['masks']
    )
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False
    )
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Basic evaluation
    logger.info("Running basic model evaluation...")
    evaluator = CVAEEvaluator(model, device, str(output_path))
    
    # Generate evaluation report
    report = evaluator.evaluate_reconstruction(test_loader, num_samples=args.n_samples)
    
    # Test generation capabilities
    logger.info("Testing generation capabilities...")
    generation_report = evaluator.evaluate_generation(test_dataset, num_samples=50, samples_per_condition=3)
    
    # Test enhanced generation capabilities if requested
    generation_test_results = None
    if args.test_generation:
        logger.info("Testing enhanced generation capabilities...")
        generation_test_results = test_enhanced_generation_capabilities(
            model, output_path, num_samples=args.generation_samples
        )
    
    # Comprehensive evaluation if requested
    if args.comprehensive:
        logger.info("Running comprehensive synthetic data quality evaluation...")
        
        # Create comprehensive output directory
        comp_output_dir = output_path / "comprehensive_evaluation"
        comp_output_dir.mkdir(exist_ok=True)
        
        # Initialize comprehensive evaluator
        comp_evaluator = ABRSyntheticDataEvaluator(sampling_rate=1000.0)
        
        logger.info("Comprehensive evaluation framework is ready!")
    
    # Print summary
    print("\n" + "="*60)
    print("ABR CVAE EVALUATION COMPLETED")
    print("="*60)
    print(f"Model: {model_identifier}")
    print(f"Evaluation ID: {output_path.name}")
    
    if report:
        print(f"\n📊 Reconstruction Performance:")
        print(f"   Mean Squared Error: {report.get('mse', 'N/A'):.4f}" if 'mse' in report else "   MSE: N/A")
        print(f"   Mean Absolute Error: {report.get('mae', 'N/A'):.4f}" if 'mae' in report else "   MAE: N/A")
        print(f"   Correlation: {report.get('correlation', 'N/A'):.4f}" if 'correlation' in report else "   Correlation: N/A")
    
    if generation_report:
        print(f"\n🎯 Generation Performance:")
        print(f"   Samples Generated: {generation_report.get('total_samples', 'N/A')}")
        print(f"   Generation Quality: {generation_report.get('generation_quality', 'N/A'):.4f}" if 'generation_quality' in generation_report else "   Quality: N/A")
    
    if generation_test_results is not None:
        print(f"\n🧪 Enhanced Generation Testing:")
        print(f"   Test Samples: {len(generation_test_results)}")
        if 'gen_age' in generation_test_results.columns:
            age_range = f"{generation_test_results['gen_age'].min():.1f}-{generation_test_results['gen_age'].max():.1f}"
            intensity_range = f"{generation_test_results['gen_intensity'].min():.1f}-{generation_test_results['gen_intensity'].max():.1f}"
            print(f"   Generated Age Range: {age_range} years")
            print(f"   Generated Intensity Range: {intensity_range} dB")
        if 'gen_lat_I' in generation_test_results.columns:
            lat_i_range = f"{generation_test_results['gen_lat_I'].min():.2f}-{generation_test_results['gen_lat_I'].max():.2f}"
            lat_v_range = f"{generation_test_results['gen_lat_V'].min():.2f}-{generation_test_results['gen_lat_V'].max():.2f}"
            print(f"   Wave I Latency Range: {lat_i_range} ms")
            print(f"   Wave V Latency Range: {lat_v_range} ms")
    
    print(f"\n📁 Results saved to: {output_path}")
    print("   • Reconstruction evaluation plots and metrics")
    print("   • Generation evaluation and clinical parameter analysis")
    print("   • Evaluation metadata and model information")
    if args.test_generation:
        print("   • Enhanced generation testing results and CSV export")
    if args.comprehensive:
        print("   • Comprehensive evaluation framework ready")
    
    print(f"\n🎯 Next Steps:")
    print("   • Review generated plots for model quality")
    print("   • Analyze clinical parameter generation accuracy")
    print("   • Compare results across different training runs")
    print("   • Consider model improvements based on metrics")


if __name__ == "__main__":
    main() 