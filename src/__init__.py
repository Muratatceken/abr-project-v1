"""
ABR CVAE Project - Source Package
==================================

A comprehensive framework for Auditory Brainstem Response (ABR) data analysis
using Conditional Variational Autoencoders (CVAE) with masking for missing data.

Modules:
    data: Data loading, preprocessing, and dataset management
    models: CVAE model architectures and components  
    training: Training loops and utilities
    evaluation: Model evaluation and synthetic data quality metrics
    utils: Utility functions and helpers
    visualization: Plotting and visualization tools
"""

__version__ = "1.0.0"
__author__ = "ABR Research Team"

# Core imports for easy access
from .models.cvae_model import ConditionalVAEWithMasking
from .data.dataset import ABRMaskedDataset, create_abr_datasets
from .training.trainer import CVAETrainer
from .evaluation.evaluator import CVAEEvaluator
from .evaluation.metrics import ABRSyntheticDataEvaluator

__all__ = [
    'ConditionalVAEWithMasking',
    'ABRMaskedDataset', 
    'create_abr_datasets',
    'CVAETrainer',
    'CVAEEvaluator',
    'ABRSyntheticDataEvaluator'
] 