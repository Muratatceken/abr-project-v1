# ABR CVAE Architecture Diagrams

This directory contains visual representations of the ABR CVAE model architecture.

## Diagrams

### 1. Main Architecture (abr_cvae_main_architecture.png)
- Complete model overview showing data flow from inputs to outputs
- Input processing: Time series, static parameters, latency/amplitude data with masking
- VAE core: Encoder, latent space, decoder
- Output reconstruction: ABR waveforms and masked clinical features
- Loss computation: Multi-component loss with β-annealing

### 2. Transformer Details (abr_cvae_transformer_details.png)
- Detailed view of transformer encoder/decoder architecture
- Attention mechanisms and layer structure
- Model statistics and parameters
- Sequence processing pipeline

### 3. Training Flow (abr_cvae_training_flow.png)
- Training pipeline and optimization process
- Current training status and progress
- Loss components and early stopping
- Checkpointing and monitoring

## Model Specifications

- **Parameters**: 520,335 (2.1 MB)
- **Latent Dimension**: 32
- **Sequence Length**: 200
- **Transformer Layers**: 2 encoder + 2 decoder
- **Attention Heads**: 4
- **Training Speed**: ~1.7 min/epoch
- **Current Status**: Stable convergence with consistent improvement

## Generated on
2025-06-09 06:24:01