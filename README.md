# ABR CVAE Project - Modular Framework

A comprehensive framework for Auditory Brainstem Response (ABR) data analysis using Conditional Variational Autoencoders (CVAE) with masking for missing clinical data.

## 🚀 Quick Start

### 1. Data Preprocessing
```bash
python preprocess.py --input dataset/abr_data_preprocessed.xlsx --output-dir data/processed
```

### 2. Model Training  
```bash
python train.py --config outputs/config.json --device cuda
```

### 3. Model Evaluation
```bash
python evaluate.py --model outputs/models/best_checkpoint.pth --comprehensive
```

## 📁 Project Structure

```
abr_project/
├── src/                        # Main source code
│   ├── data/                   # Data handling modules
│   │   ├── dataset.py          # PyTorch dataset classes
│   │   ├── preprocessing.py    # Data preprocessing utilities
│   │   └── __init__.py
│   ├── models/                 # Model architectures
│   │   ├── cvae_model.py       # Conditional VAE with masking
│   │   └── __init__.py
│   ├── training/               # Training utilities
│   │   ├── trainer.py          # Training loop and optimization
│   │   └── __init__.py
│   ├── evaluation/             # Evaluation and metrics
│   │   ├── evaluator.py        # Model performance evaluation
│   │   ├── metrics.py          # Synthetic data quality metrics
│   │   └── __init__.py
│   ├── visualization/          # Plotting and diagrams
│   │   ├── architecture.py     # Model architecture diagrams
│   │   └── __init__.py
│   ├── utils/                  # Utility functions
│   │   └── __init__.py
│   └── __init__.py
├── train.py                    # Main training script
├── evaluate.py                 # Main evaluation script
├── preprocess.py               # Data preprocessing script
├── data/                       # Data storage
│   ├── processed/              # Processed data files
│   └── raw/                    # Raw data files
├── outputs/                    # Training outputs and results
│   ├── models/                 # Saved model checkpoints
│   ├── plots/                  # Generated visualizations
│   ├── evaluation/             # Evaluation results
│   ├── logs/                   # Training logs
│   └── config.json             # Model configuration
├── dataset/                    # Original dataset files
├── configs/                    # Configuration templates
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
├── cleanup_project.py          # Project cleanup script
└── README.md                   # This file
```

## 🔧 Installation

### Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate abr_project
```

### Using Pip
```bash
pip install -r requirements.txt
```

## 📊 Features

### Core Capabilities
- **Conditional VAE**: Advanced neural architecture with masking for missing data
- **Clinical Integration**: Handles ABR-specific parameters (latency, amplitude, static features)
- **Quality Assessment**: Comprehensive synthetic data evaluation metrics
- **Modular Design**: Clean, maintainable codebase with clear separation of concerns

### Evaluation Metrics
- **Reconstruction Quality**: RMSE, correlation, PSNR, SSIM
- **Distributional Similarity**: Wasserstein distance, Jensen-Shannon divergence, MMD
- **Frequency Domain**: Power spectral density, spectral centroid analysis
- **Clinical Metrics**: Wave latency preservation, morphological similarity
- **Diversity Assessment**: Coverage, precision, mode collapse detection

## 🎯 Usage Examples

### Basic Training
```bash
# Train with default configuration
python train.py

# Train with custom settings
python train.py --config custom_config.json --device cuda --output-dir experiments/run1
```

### Comprehensive Evaluation
```bash
# Basic evaluation
python evaluate.py --model outputs/models/best_checkpoint.pth

# Full synthetic data quality assessment
python evaluate.py --model outputs/models/best_checkpoint.pth --comprehensive --n-samples 2000
```

### Custom Preprocessing
```bash
# Preprocess with custom parameters
python preprocess.py \
    --input dataset/abr_data.xlsx \
    --output-dir data/custom_processed \
    --sequence-length 250 \
    --min-fmp 3.0
```

## 📈 Model Architecture

The CVAE model features:
- **Input Processing**: 200-point ABR time series with static parameters
- **Masking Mechanism**: Handles missing latency/amplitude data
- **Encoder**: 1D CNN + Transformer encoder (4 layers, 8 heads)
- **Latent Space**: 128-dimensional with reparameterization trick
- **Decoder**: Transformer decoder + transposed CNN
- **Conditioning**: Age, intensity, stimulus rate, hearing loss type

## 🔬 Scientific Applications

### Clinical Use Cases
- ABR waveform synthesis for research
- Data augmentation for machine learning
- Missing data imputation
- Hearing loss pattern analysis

### Research Applications
- Synthetic dataset generation
- Model comparison and benchmarking
- Statistical analysis of ABR characteristics
- Quality assessment of generative models

## 📋 Configuration

Key configuration parameters:

```json
{
    "static_dim": 4,
    "masked_features_dim": 64,
    "latent_dim": 128,
    "sequence_length": 200,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "epochs": 10,
    "beta_start": 0.0,
    "beta_end": 0.1
}
```

## 🧹 Project Maintenance

### Clean Up Project
```bash
# Remove unnecessary files and reorganize structure
python cleanup_project.py
```

This will:
- Remove redundant and legacy files
- Organize outputs into subdirectories
- Create clean modular structure
- Preserve essential components

## 📚 Documentation

### Key Classes
- `ConditionalVAEWithMasking`: Main CVAE model class
- `ABRDataset`: PyTorch dataset with masking support
- `ABRCVAETrainer`: Training loop with logging and checkpointing
- `ABRModelEvaluator`: Basic model evaluation
- `ABRSyntheticDataEvaluator`: Comprehensive synthetic data quality assessment

### Best Practices
1. Use the modular entry points (`train.py`, `evaluate.py`, `preprocess.py`)
2. Configure via JSON files rather than hardcoding parameters
3. Monitor training with comprehensive evaluation metrics
4. Validate synthetic data quality before use in applications
5. Keep processed data and outputs organized in subdirectories

## 🤝 Contributing

The project follows a modular architecture for easy extension:
1. Add new model components in `src/models/`
2. Extend evaluation metrics in `src/evaluation/metrics.py`
3. Add visualization tools in `src/visualization/`
4. Create utility functions in `src/utils/`

## 📄 License

This project is developed for research purposes in auditory neuroscience and biomedical signal processing.

## 🎯 Next Steps

1. **Run project cleanup**: `python cleanup_project.py`
2. **Test modular structure**: Verify imports work correctly
3. **Update documentation**: Add specific usage examples
4. **Performance optimization**: Profile and optimize critical paths
5. **Extended evaluation**: Add more clinical validation metrics 