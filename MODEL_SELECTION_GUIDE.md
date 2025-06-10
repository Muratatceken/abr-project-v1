# ABR CVAE Model Selection Guide

This guide explains how to use both the **Original CVAE** and **Advanced CVAE** models for ABR data generation and analysis.

## Available Models

### 1. Original CVAE
- Standard conditional VAE with masking
- Single latent space
- Basic encoder-decoder architecture
- Good for reconstruction tasks

### 2. Advanced CVAE
- **Hierarchical latent variables** (waveform + peak latents)
- **Conditional priors** instead of standard normal
- **FiLM layers** for dynamic conditioning
- **Temporal decoder** with Transformer architecture
- **Explicit peak detection** and multi-task learning
- **Noise augmentation** during training
- Better for generation from scratch

## Quick Start

### Training Models

```bash
# Train Original CVAE
python train.py --config configs/cvae_config.yaml --model original --epochs 100

# Train Advanced CVAE with conditional priors
python train.py --config configs/advanced_cvae_config.yaml --model advanced --epochs 200 --use-conditional-prior

# Train Advanced CVAE with custom noise augmentation
python train.py --config configs/advanced_cvae_config.yaml --model advanced --epochs 150 --noise-augmentation 0.15
```

### Evaluating Models

```bash
# Evaluate any trained model (auto-detects type)
python evaluate.py --model outputs/checkpoints/best_model.pth --test-generation

# Force model type if needed
python evaluate.py --model outputs/checkpoints/best_model.pth --model-type advanced --temperature 0.8

# Comprehensive evaluation
python evaluate.py --model outputs/checkpoints/best_model.pth --comprehensive --test-generation
```

### Using the Demo Script

```bash
# Quick demo of both models
python demo_model_selection.py --action demo

# Train specific model
python demo_model_selection.py --action train --model advanced --epochs 50 --use-conditional-prior

# Evaluate a model
python demo_model_selection.py --action evaluate --model-path outputs/best_model.pth --comprehensive

# Compare two models
python demo_model_selection.py --action compare --original-path outputs_original/checkpoints/best_model.pth --advanced-path outputs_advanced/checkpoints/best_model.pth

# Check training history
python demo_model_selection.py --action history
```

## Configuration Files

### Original CVAE Config (`configs/cvae_config.yaml`)
```yaml
model:
  type: "original"
  latent_dim: 128
  hidden_dim: 256
  # ... other original model settings
```

### Advanced CVAE Config (`configs/advanced_cvae_config.yaml`)
```yaml
model:
  type: "advanced"
  waveform_latent_dim: 64
  peak_latent_dim: 32
  use_conditional_prior: true
  noise_augmentation: 0.1
  # ... other advanced model settings
```

## Key Differences

| Feature | Original CVAE | Advanced CVAE |
|---------|---------------|---------------|
| **Latent Space** | Single latent (128D) | Hierarchical (64D + 32D) |
| **Prior** | Standard Normal | Conditional Prior |
| **Conditioning** | Concatenation | FiLM Layers |
| **Decoder** | Standard | Temporal Transformer |
| **Peak Detection** | Post-processing | Explicit Multi-task |
| **Generation Quality** | Good reconstruction | Better pure generation |
| **Training Time** | Faster | Slower |
| **Model Size** | Smaller | Larger |

## Advanced Features

### Conditional Prior Generation
The Advanced CVAE learns conditional priors `p(z|c)` instead of using standard normal priors:

```python
# This addresses the "generation from scratch" problem
# by learning meaningful priors conditioned on clinical parameters
```

### Hierarchical Latent Variables
- **Waveform latent**: Captures general ABR signal structure
- **Peak latent**: Captures specific peak characteristics
- Better disentanglement and controllable generation

### Multi-task Learning
The Advanced CVAE simultaneously learns to:
1. Reconstruct ABR waveforms
2. Predict peak presence/absence
3. Estimate peak latencies and amplitudes
4. Generate consistent static parameters

### FiLM Conditioning
Feature-wise Linear Modulation provides dynamic conditioning:
```python
# Instead of simple concatenation:
# output = decoder(concat(z, condition))

# FiLM uses multiplicative and additive conditioning:
# output = decoder(z * gamma(condition) + beta(condition))
```

## Performance Expectations

### Original CVAE
- ✅ Excellent reconstruction quality
- ✅ Fast training and inference
- ✅ Stable training
- ❌ Poor generation from scratch
- ❌ Limited peak detection

### Advanced CVAE
- ✅ Good reconstruction quality
- ✅ **Much better generation from scratch**
- ✅ Explicit peak detection
- ✅ Causal static parameter generation
- ❌ Slower training
- ❌ More complex architecture
- ❌ Requires more hyperparameter tuning

## Troubleshooting

### Common Issues

1. **"Generation from scratch is poor"** → Use Advanced CVAE with conditional priors
2. **"Training is unstable"** → Reduce learning rate, enable gradient clipping
3. **"Peak detection fails"** → Use Advanced CVAE with explicit peak head
4. **"Model too slow"** → Use Original CVAE or reduce model size

### Hyperparameter Tuning

**For Advanced CVAE:**
- `beta_waveform`: Controls waveform latent regularization (default: 1.0)
- `beta_peak`: Controls peak latent regularization (default: 0.5)
- `noise_augmentation`: Training noise level (default: 0.1)
- `use_conditional_prior`: Enable conditional priors (recommended: true)

## Example Workflows

### Research Workflow
```bash
# 1. Train both models
python demo_model_selection.py --action train --model original --epochs 100
python demo_model_selection.py --action train --model advanced --epochs 200 --use-conditional-prior

# 2. Compare performance
python demo_model_selection.py --action compare --original-path outputs_original/checkpoints/best_model.pth --advanced-path outputs_advanced/checkpoints/best_model.pth

# 3. Check training history
python demo_model_selection.py --action history
```

### Production Workflow
```bash
# 1. Train Advanced CVAE for best generation quality
python train.py --config configs/advanced_cvae_config.yaml --model advanced --epochs 300 --use-conditional-prior --notes "Production model"

# 2. Comprehensive evaluation
python evaluate.py --model outputs/checkpoints/best_model.pth --comprehensive --test-generation --generation-samples 20

# 3. Generate synthetic data
python -c "
import torch
from src.models.advanced_cvae_model import AdvancedConditionalVAE
# ... load model and generate samples
"
```

## Tips for Best Results

1. **Use Advanced CVAE** for generation tasks
2. **Enable conditional priors** for better generation quality
3. **Tune noise augmentation** based on your data characteristics
4. **Monitor both reconstruction and generation metrics**
5. **Use temperature scaling** during generation for diversity control
6. **Check training history** to compare different runs

## Getting Help

- Check `python train.py --help` for all training options
- Check `python evaluate.py --help` for all evaluation options
- Check `python demo_model_selection.py --help` for demo script options
- Review training logs in the output directories
- Use `--check-history` to see past training runs 