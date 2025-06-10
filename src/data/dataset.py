#!/usr/bin/env python3
"""
ABR Dataset with Masking Support
===============================

This module provides dataset classes for ABR data with proper handling of:
- Time series data (first 200 timestamps)
- Static parameters (age, intensity, stimulus rate, hear loss type)
- Latency and amplitude values with masking for missing data
- Efficient data loading from preprocessed files
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path
import pickle
import os

logger = logging.getLogger(__name__)


class ABRMaskedDataset(Dataset):
    """
    PyTorch Dataset for ABR data with masking support for latency and amplitude.
    """
    
    def __init__(self, 
                 time_series: np.ndarray,
                 static_params: Dict[str, np.ndarray],
                 latency_data: Dict[str, np.ndarray],
                 amplitude_data: Dict[str, np.ndarray],
                 masks: Dict[str, Dict[str, np.ndarray]],
                 transform=None):
        """
        Initialize the dataset.
        
        Args:
            time_series: Time series data of shape (n_samples, sequence_length)
            static_params: Dictionary of static parameters
            latency_data: Dictionary of latency values
            amplitude_data: Dictionary of amplitude values
            masks: Dictionary containing latency and amplitude masks
            transform: Optional transform to apply to time series data
        """
        self.time_series = torch.FloatTensor(time_series)
        self.transform = transform
        
        # Static parameters
        self.age = torch.FloatTensor(static_params['age'])
        self.intensity = torch.FloatTensor(static_params['intensity'])
        self.stimulus_rate = torch.FloatTensor(static_params['stimulus_rate'])
        self.hear_loss = torch.LongTensor(static_params['hear_loss'])
        
        # Latency data and masks
        self.latency_data = {}
        self.latency_masks = {}
        for key in latency_data:
            self.latency_data[key] = torch.FloatTensor(latency_data[key])
            self.latency_masks[key] = torch.FloatTensor(masks['latency_masks'][key])
        
        # Amplitude data and masks
        self.amplitude_data = {}
        self.amplitude_masks = {}
        for key in amplitude_data:
            self.amplitude_data[key] = torch.FloatTensor(amplitude_data[key])
            self.amplitude_masks[key] = torch.FloatTensor(masks['amplitude_masks'][key])
        
        self.n_samples = len(self.time_series)
        
        # Validate data consistency
        self._validate_data()
    
    def _validate_data(self):
        """Validate that all data arrays have consistent length."""
        expected_length = self.n_samples
        
        # Check static parameters
        assert len(self.age) == expected_length, f"Age length mismatch: {len(self.age)} vs {expected_length}"
        assert len(self.intensity) == expected_length, f"Intensity length mismatch: {len(self.intensity)} vs {expected_length}"
        assert len(self.stimulus_rate) == expected_length, f"Stimulus rate length mismatch: {len(self.stimulus_rate)} vs {expected_length}"
        assert len(self.hear_loss) == expected_length, f"Hear loss length mismatch: {len(self.hear_loss)} vs {expected_length}"
        
        # Check latency data
        for key in self.latency_data:
            assert len(self.latency_data[key]) == expected_length, f"Latency {key} length mismatch"
            assert len(self.latency_masks[key]) == expected_length, f"Latency mask {key} length mismatch"
        
        # Check amplitude data
        for key in self.amplitude_data:
            assert len(self.amplitude_data[key]) == expected_length, f"Amplitude {key} length mismatch"
            assert len(self.amplitude_masks[key]) == expected_length, f"Amplitude mask {key} length mismatch"
    
    def __len__(self):
        """Return the number of samples."""
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing all data for the sample
        """
        # Get time series data
        time_series = self.time_series[idx]
        if self.transform:
            time_series = self.transform(time_series)
        
        # Static parameters
        static_params = {
            'age': self.age[idx],
            'intensity': self.intensity[idx],
            'stimulus_rate': self.stimulus_rate[idx],
            'hear_loss': self.hear_loss[idx]
        }
        
        # Latency data with masks
        latency_data = {}
        latency_masks = {}
        for key in self.latency_data:
            latency_data[key] = self.latency_data[key][idx]
            latency_masks[key] = self.latency_masks[key][idx]
        
        # Amplitude data with masks
        amplitude_data = {}
        amplitude_masks = {}
        for key in self.amplitude_data:
            amplitude_data[key] = self.amplitude_data[key][idx]
            amplitude_masks[key] = self.amplitude_masks[key][idx]
        
        return {
            'time_series': time_series,
            'static_params': static_params,
            'latency_data': latency_data,
            'latency_masks': latency_masks,
            'amplitude_data': amplitude_data,
            'amplitude_masks': amplitude_masks
        }
    
    def get_batch_data(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Get batch data for multiple indices (useful for custom batching).
        
        Args:
            indices: List of sample indices
            
        Returns:
            Dictionary containing batched data
        """
        batch_data = {
            'time_series': self.time_series[indices],
            'static_params': {
                'age': self.age[indices],
                'intensity': self.intensity[indices],
                'stimulus_rate': self.stimulus_rate[indices],
                'hear_loss': self.hear_loss[indices]
            },
            'latency_data': {},
            'latency_masks': {},
            'amplitude_data': {},
            'amplitude_masks': {}
        }
        
        # Batch latency data
        for key in self.latency_data:
            batch_data['latency_data'][key] = self.latency_data[key][indices]
            batch_data['latency_masks'][key] = self.latency_masks[key][indices]
        
        # Batch amplitude data
        for key in self.amplitude_data:
            batch_data['amplitude_data'][key] = self.amplitude_data[key][indices]
            batch_data['amplitude_masks'][key] = self.amplitude_masks[key][indices]
        
        return batch_data


class ABRTimeSeriesAugmentation:
    """Augmentation transforms for ABR time series data."""
    
    def __init__(self, noise_std: float = 0.01, time_shift_max: int = 10):
        """
        Initialize augmentation.
        
        Args:
            noise_std: Standard deviation for Gaussian noise
            time_shift_max: Maximum time shift in samples
        """
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
    
    def __call__(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to time series.
        
        Args:
            time_series: Input time series tensor
            
        Returns:
            Augmented time series tensor
        """
        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(time_series) * self.noise_std
            time_series = time_series + noise
        
        # Apply time shift
        if self.time_shift_max > 0:
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
            if shift != 0:
                if shift > 0:
                    # Shift right, pad left with zeros
                    shifted = torch.cat([torch.zeros(shift), time_series[:-shift]])
                else:
                    # Shift left, pad right with zeros
                    shifted = torch.cat([time_series[-shift:], torch.zeros(-shift)])
                time_series = shifted
        
        return time_series


def custom_collate_fn(batch):
    """
    Custom collate function for ABR dataset with masking.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data dictionary
    """
    batch_size = len(batch)
    
    # Stack time series
    time_series = torch.stack([item['time_series'] for item in batch])
    
    # Stack static parameters
    static_params = {
        'age': torch.stack([item['static_params']['age'] for item in batch]),
        'intensity': torch.stack([item['static_params']['intensity'] for item in batch]),
        'stimulus_rate': torch.stack([item['static_params']['stimulus_rate'] for item in batch]),
        'hear_loss': torch.stack([item['static_params']['hear_loss'] for item in batch])
    }
    
    # Get all latency and amplitude keys from first item
    latency_keys = list(batch[0]['latency_data'].keys())
    amplitude_keys = list(batch[0]['amplitude_data'].keys())
    
    # Stack latency data and masks
    latency_data = {}
    latency_masks = {}
    for key in latency_keys:
        latency_data[key] = torch.stack([item['latency_data'][key] for item in batch])
        latency_masks[key] = torch.stack([item['latency_masks'][key] for item in batch])
    
    # Stack amplitude data and masks
    amplitude_data = {}
    amplitude_masks = {}
    for key in amplitude_keys:
        amplitude_data[key] = torch.stack([item['amplitude_data'][key] for item in batch])
        amplitude_masks[key] = torch.stack([item['amplitude_masks'][key] for item in batch])
    
    return {
        'time_series': time_series,
        'static_params': static_params,
        'latency_data': latency_data,
        'latency_masks': latency_masks,
        'amplitude_data': amplitude_data,
        'amplitude_masks': amplitude_masks
    }


def create_abr_datasets(data_dir: str = "data/processed",
                       train_split: float = 0.7,
                       val_split: float = 0.15,
                       test_split: float = 0.15,
                       random_seed: int = 42,
                       augment_train: bool = True,
                       augmentation_params: Optional[Dict] = None) -> Tuple[ABRMaskedDataset, ABRMaskedDataset, ABRMaskedDataset]:
    """
    Create train, validation, and test datasets from processed data.
    
    Args:
        data_dir: Directory containing processed data
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducible splits
        augment_train: Whether to apply augmentation to training data
        augmentation_params: Parameters for augmentation
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load processed data
    try:
        from .preprocessing import load_processed_data
    except ImportError:
        # Fallback for when running as script
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from preprocessing import load_processed_data
    data = load_processed_data(data_dir)
    
    # Extract data components
    time_series = data['time_series']
    static_params = data['static_params']
    latency_data = data['latency_data']
    amplitude_data = data['amplitude_data']
    masks = data['masks']
    
    n_samples = len(time_series)
    
    # Create train/val/test splits
    np.random.seed(random_seed)
    indices = np.random.permutation(n_samples)
    
    train_end = int(train_split * n_samples)
    val_end = train_end + int(val_split * n_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    logger.info(f"Data splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Create augmentation transform for training
    augmentation = None
    if augment_train:
        aug_params = augmentation_params or {'noise_std': 0.01, 'time_shift_max': 10}
        augmentation = ABRTimeSeriesAugmentation(**aug_params)
    
    # Create datasets
    def create_subset(indices):
        return {
            'time_series': time_series[indices],
            'static_params': {k: v[indices] for k, v in static_params.items() if k != 'hear_loss_mapping'},
            'latency_data': {k: v[indices] for k, v in latency_data.items()},
            'amplitude_data': {k: v[indices] for k, v in amplitude_data.items()},
            'masks': {
                'latency_masks': {k: v[indices] for k, v in masks['latency_masks'].items()},
                'amplitude_masks': {k: v[indices] for k, v in masks['amplitude_masks'].items()}
            }
        }
    
    train_subset = create_subset(train_indices)
    val_subset = create_subset(val_indices)
    test_subset = create_subset(test_indices)
    
    train_dataset = ABRMaskedDataset(
        train_subset['time_series'],
        train_subset['static_params'],
        train_subset['latency_data'],
        train_subset['amplitude_data'],
        train_subset['masks'],
        transform=augmentation
    )
    
    val_dataset = ABRMaskedDataset(
        val_subset['time_series'],
        val_subset['static_params'],
        val_subset['latency_data'],
        val_subset['amplitude_data'],
        val_subset['masks']
    )
    
    test_dataset = ABRMaskedDataset(
        test_subset['time_series'],
        test_subset['static_params'],
        test_subset['latency_data'],
        test_subset['amplitude_data'],
        test_subset['masks']
    )
    
    return train_dataset, val_dataset, test_dataset


def create_abr_dataloaders(train_dataset: ABRMaskedDataset,
                          val_dataset: ABRMaskedDataset,
                          test_dataset: ABRMaskedDataset,
                          batch_size: int = 32,
                          num_workers: int = 4,
                          pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for the datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    try:
        # Create datasets (adjust path for when running from src/data directory)
        data_dir = "../../data/processed" if os.path.exists("../../data/processed") else "data/processed"
        train_dataset, val_dataset, test_dataset = create_abr_datasets(data_dir=data_dir)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_abr_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=16
        )
        
        # Test data loading
        print("Testing data loading...")
        for batch_idx, batch in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Time series shape: {batch['time_series'].shape}")
            print(f"  Static params: {[(k, v.shape) for k, v in batch['static_params'].items()]}")
            print(f"  Latency data: {[(k, v.shape) for k, v in batch['latency_data'].items()]}")
            print(f"  Latency masks: {[(k, v.shape) for k, v in batch['latency_masks'].items()]}")
            print(f"  Amplitude data: {[(k, v.shape) for k, v in batch['amplitude_data'].items()]}")
            print(f"  Amplitude masks: {[(k, v.shape) for k, v in batch['amplitude_masks'].items()]}")
            
            if batch_idx >= 2:  # Test only first few batches
                break
        
        print("\nDataset creation and loading successful!")
        
    except FileNotFoundError:
        print("Processed data not found. Please run data_preparation.py first.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 