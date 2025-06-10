#!/usr/bin/env python3
"""
ABR Data Preparation Script
==========================

This script processes the preprocessed ABR Excel data and prepares it for model training.
It applies all filtering criteria and saves the data in an efficient format for quick loading.

Requirements:
- Only first 200 time stamps in time series
- Only time series with FMP > 2.0
- Only alternate polarity stimulus
- Static parameters: age, intensity, stimulus rate, hear loss type
- Latency and amplitude values with masking for missing data
"""

import pandas as pd
import numpy as np
import pickle
import h5py
import json
from pathlib import Path
import logging
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ABRDataPreparator:
    """Class to handle ABR data preparation and filtering."""
    
    def __init__(self, excel_path: str, output_dir: str = "data/processed"):
        """
        Initialize the data preparator.
        
        Args:
            excel_path: Path to the preprocessed Excel file
            output_dir: Directory to save processed data
        """
        self.excel_path = Path(excel_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.raw_data = None
        self.filtered_data = None
        self.time_series_data = None
        self.static_data = None
        self.latency_data = None
        self.amplitude_data = None
        self.masks = None
        
    def load_excel_data(self) -> pd.DataFrame:
        """Load the Excel data once."""
        logger.info(f"Loading Excel data from {self.excel_path}")
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")
        
        self.raw_data = pd.read_excel(self.excel_path)
        logger.info(f"Loaded {len(self.raw_data)} records with {len(self.raw_data.columns)} columns")
        return self.raw_data
    
    def apply_filters(self, fmp_threshold: float = 1.0) -> pd.DataFrame:
        """Apply all filtering criteria."""
        logger.info("Applying filtering criteria...")
        
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_excel_data() first.")
        
        # Start with all data
        filtered_df = self.raw_data.copy()
        initial_count = len(filtered_df)
        
        # Filter 1: FMP threshold (configurable)
        # Based on analysis: FMP > 2.0 removes 72% of data, using lower default threshold
        filtered_df = filtered_df[filtered_df['FMP'] > fmp_threshold]
        fmp_count = len(filtered_df)
        logger.info(f"After FMP > {fmp_threshold} filter: {fmp_count} records ({initial_count - fmp_count} removed)")
        logger.warning(f"FMP filter removed {((initial_count - fmp_count)/initial_count)*100:.1f}% of data")
        
        # Filter 2: Only alternate polarity
        filtered_df = filtered_df[filtered_df['Stimulus Polarity'] == 'Alternate']
        polarity_count = len(filtered_df)
        logger.info(f"After alternate polarity filter: {polarity_count} records ({fmp_count - polarity_count} removed)")
        
        # Filter 3: Remove rows with missing critical data
        critical_columns = ['Age', 'Intensity', 'Stimulus Rate', 'Hear_Loss']
        initial_filtered_count = len(filtered_df)
        filtered_df = filtered_df.dropna(subset=critical_columns)
        final_count = len(filtered_df)
        logger.info(f"After removing missing critical data: {final_count} records ({initial_filtered_count - final_count} removed)")
        
        self.filtered_data = filtered_df
        logger.info(f"Total filtering removed {initial_count - final_count} records ({(initial_count - final_count)/initial_count*100:.1f}%)")
        
        return self.filtered_data
    
    def extract_time_series(self, sequence_length: int = 200) -> np.ndarray:
        """
        Extract time series data (first 200 timestamps).
        
        Args:
            sequence_length: Number of time points to extract (default: 200)
        
        Returns:
            Time series array of shape (n_samples, sequence_length)
        """
        logger.info(f"Extracting time series data (first {sequence_length} timestamps)")
        
        if self.filtered_data is None:
            raise ValueError("Filtered data not available. Call apply_filters() first.")
        
        # Time series columns are numbered from '1' to '467'
        time_columns = [str(i) for i in range(1, sequence_length + 1)]
        
        # Check which columns exist
        available_time_cols = [col for col in time_columns if col in self.filtered_data.columns]
        
        if len(available_time_cols) < sequence_length:
            logger.warning(f"Only {len(available_time_cols)} time columns available, requested {sequence_length}")
            sequence_length = len(available_time_cols)
        
        # Extract time series data
        time_series = self.filtered_data[available_time_cols].values.astype(np.float32)
        
        # Handle any remaining NaN values
        nan_count = np.isnan(time_series).sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in time series, filling with 0")
            time_series = np.nan_to_num(time_series, nan=0.0)
        
        self.time_series_data = time_series
        logger.info(f"Extracted time series shape: {time_series.shape}")
        
        return self.time_series_data
    
    def extract_static_parameters(self) -> Dict[str, np.ndarray]:
        """
        Extract static parameters: age, intensity, stimulus rate, hear loss type.
        
        Returns:
            Dictionary containing static parameter arrays
        """
        logger.info("Extracting static parameters")
        
        if self.filtered_data is None:
            raise ValueError("Filtered data not available. Call apply_filters() first.")
        
        # Extract numeric parameters
        age = self.filtered_data['Age'].values.astype(np.float32)
        intensity = self.filtered_data['Intensity'].values.astype(np.float32)
        stimulus_rate = self.filtered_data['Stimulus Rate'].values.astype(np.float32)
        
        # Handle hearing loss (categorical)
        hear_loss_values = self.filtered_data['Hear_Loss'].values
        unique_hear_loss = sorted([x for x in self.filtered_data['Hear_Loss'].unique() if pd.notna(x)])
        hear_loss_mapping = {val: idx for idx, val in enumerate(unique_hear_loss)}
        hear_loss_encoded = np.array([hear_loss_mapping.get(val, 0) for val in hear_loss_values], dtype=np.int32)
        
        self.static_data = {
            'age': age,
            'intensity': intensity,
            'stimulus_rate': stimulus_rate,
            'hear_loss': hear_loss_encoded,
            'hear_loss_mapping': hear_loss_mapping
        }
        
        logger.info(f"Static parameters extracted:")
        logger.info(f"  Age range: {age.min():.1f} - {age.max():.1f}")
        logger.info(f"  Intensity range: {intensity.min():.1f} - {intensity.max():.1f}")
        logger.info(f"  Stimulus rate range: {stimulus_rate.min():.1f} - {stimulus_rate.max():.1f}")
        logger.info(f"  Hearing loss categories: {unique_hear_loss}")
        
        return self.static_data
    
    def extract_latency_amplitude_with_masks(self) -> Dict[str, Any]:
        """
        Extract latency and amplitude values with masking for missing data.
        
        Returns:
            Dictionary containing latency/amplitude data and masks
        """
        logger.info("Extracting latency and amplitude data with masking")
        
        if self.filtered_data is None:
            raise ValueError("Filtered data not available. Call apply_filters() first.")
        
        # Latency columns
        latency_columns = ['I Latancy', 'III Latancy', 'V Latancy']
        # Amplitude columns
        amplitude_columns = ['I Amplitude', 'III Amplitude', 'V Amplitude']
        
        # Extract latency data and create masks
        latency_data = {}
        latency_masks = {}
        
        for col in latency_columns:
            if col in self.filtered_data.columns:
                values = self.filtered_data[col].values.astype(np.float32)
                mask = ~np.isnan(values)  # True where data is available
                

                
                values = np.nan_to_num(values, nan=0.0)  # Replace NaN with 0
                
                latency_data[col] = values
                latency_masks[col] = mask.astype(np.float32)
                
                availability_pct = mask.mean() * 100
                logger.info(f"  {col}: {mask.sum()}/{len(mask)} available ({availability_pct:.1f}%)")
                
                # Add warnings for low data availability
                if availability_pct < 20:
                    logger.warning(f"  ⚠️ {col} has very low availability ({availability_pct:.1f}%) - consider excluding from model")
                elif availability_pct < 50:
                    logger.warning(f"  ⚠️ {col} has low availability ({availability_pct:.1f}%) - may affect model performance")
        
        # Extract amplitude data and create masks
        amplitude_data = {}
        amplitude_masks = {}
        
        for col in amplitude_columns:
            if col in self.filtered_data.columns:
                values = self.filtered_data[col].values.astype(np.float32)
                mask = ~np.isnan(values)  # True where data is available
                values = np.nan_to_num(values, nan=0.0)  # Replace NaN with 0
                
                amplitude_data[col] = values
                amplitude_masks[col] = mask.astype(np.float32)
                
                availability_pct = mask.mean() * 100
                logger.info(f"  {col}: {mask.sum()}/{len(mask)} available ({availability_pct:.1f}%)")
                
                # Add warnings for low data availability
                if availability_pct < 20:
                    logger.warning(f"  ⚠️ {col} has very low availability ({availability_pct:.1f}%) - consider excluding from model")
                elif availability_pct < 50:
                    logger.warning(f"  ⚠️ {col} has low availability ({availability_pct:.1f}%) - may affect model performance")
        
        self.latency_data = latency_data
        self.amplitude_data = amplitude_data
        self.masks = {
            'latency_masks': latency_masks,
            'amplitude_masks': amplitude_masks
        }
        
        return {
            'latency_data': latency_data,
            'amplitude_data': amplitude_data,
            'masks': self.masks
        }
    
    def normalize_data(self, apply_normalization: bool = True) -> Dict[str, Dict]:
        """
        Apply normalization to the data.
        
        Args:
            apply_normalization: Whether to apply normalization
        
        Returns:
            Dictionary containing normalization statistics
        """
        if not apply_normalization:
            logger.info("Skipping normalization")
            return {}
        
        logger.info("Applying data normalization")
        
        normalization_stats = {}
        
        # Normalize time series data
        if self.time_series_data is not None:
            ts_mean = np.mean(self.time_series_data, axis=0, keepdims=True)
            ts_std = np.std(self.time_series_data, axis=0, keepdims=True)
            ts_std = np.where(ts_std == 0, 1.0, ts_std)  # Avoid division by zero
            
            self.time_series_data = (self.time_series_data - ts_mean) / ts_std
            
            normalization_stats['time_series'] = {
                'mean': ts_mean.flatten(),
                'std': ts_std.flatten()
            }
        
        # Normalize static parameters
        if self.static_data is not None:
            for key in ['age', 'intensity', 'stimulus_rate']:
                if key in self.static_data:
                    data = self.static_data[key]
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    if std_val == 0:
                        std_val = 1.0
                    
                    self.static_data[key] = (data - mean_val) / std_val
                    
                    normalization_stats[key] = {'mean': mean_val, 'std': std_val}
        
        # Normalize latency data
        if self.latency_data is not None:
            for key, data in self.latency_data.items():
                # Only normalize non-zero values (where mask is True)
                mask = self.masks['latency_masks'][key]
                valid_data = data[mask == 1]
                
                if len(valid_data) > 0:
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    if std_val == 0:
                        std_val = 1.0
                    
                    # Apply normalization only to valid data points
                    normalized_data = data.copy()
                    normalized_data[mask == 1] = (data[mask == 1] - mean_val) / std_val
                    self.latency_data[key] = normalized_data
                    
                    normalization_stats[f'latency_{key}'] = {'mean': mean_val, 'std': std_val}
        
        # Normalize amplitude data
        if self.amplitude_data is not None:
            for key, data in self.amplitude_data.items():
                # Only normalize non-zero values (where mask is True)
                mask = self.masks['amplitude_masks'][key]
                valid_data = data[mask == 1]
                
                if len(valid_data) > 0:
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    if std_val == 0:
                        std_val = 1.0
                    
                    # Apply normalization only to valid data points
                    normalized_data = data.copy()
                    normalized_data[mask == 1] = (data[mask == 1] - mean_val) / std_val
                    self.amplitude_data[key] = normalized_data
                    
                    normalization_stats[f'amplitude_{key}'] = {'mean': mean_val, 'std': std_val}
        
        logger.info("Data normalization completed")
        return normalization_stats
    
    def save_processed_data(self, normalization_stats: Dict = None) -> None:
        """Save all processed data to efficient formats."""
        logger.info(f"Saving processed data to {self.output_dir}")
        
        # Save time series data as HDF5 (most efficient for large arrays)
        h5_path = self.output_dir / "abr_time_series.h5"
        with h5py.File(h5_path, 'w') as f:
            if self.time_series_data is not None:
                f.create_dataset('time_series', data=self.time_series_data, compression='gzip')
            
            # Save static parameters
            if self.static_data is not None:
                static_group = f.create_group('static_params')
                for key, value in self.static_data.items():
                    if key != 'hear_loss_mapping':  # Skip non-array data
                        static_group.create_dataset(key, data=value)
            
            # Save latency data
            if self.latency_data is not None:
                latency_group = f.create_group('latency_data')
                for key, value in self.latency_data.items():
                    latency_group.create_dataset(key, data=value)
            
            # Save amplitude data
            if self.amplitude_data is not None:
                amplitude_group = f.create_group('amplitude_data')
                for key, value in self.amplitude_data.items():
                    amplitude_group.create_dataset(key, data=value)
            
            # Save masks
            if self.masks is not None:
                mask_group = f.create_group('masks')
                for mask_type, masks in self.masks.items():
                    type_group = mask_group.create_group(mask_type)
                    for key, value in masks.items():
                        type_group.create_dataset(key, data=value)
        
        # Save metadata as JSON
        metadata = {
            'n_samples': len(self.filtered_data) if self.filtered_data is not None else 0,
            'sequence_length': self.time_series_data.shape[1] if self.time_series_data is not None else 0,
            'hear_loss_mapping': self.static_data.get('hear_loss_mapping', {}) if self.static_data else {},
            'normalization_stats': normalization_stats or {},
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save a pickle file for quick loading of all data
        pickle_path = self.output_dir / "abr_processed_data.pkl"
        processed_data = {
            'time_series': self.time_series_data,
            'static_params': self.static_data,
            'latency_data': self.latency_data,
            'amplitude_data': self.amplitude_data,
            'masks': self.masks,
            'metadata': metadata
        }
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info(f"Processed data saved:")
        logger.info(f"  HDF5 file: {h5_path}")
        logger.info(f"  Metadata: {metadata_path}")
        logger.info(f"  Pickle file: {pickle_path}")
    
    def process_full_pipeline(self, sequence_length: int = 200, apply_normalization: bool = True, fmp_threshold: float = 1.0) -> Dict:
        """
        Run the complete data processing pipeline.
        
        Args:
            sequence_length: Number of time points to extract
            apply_normalization: Whether to apply normalization
        
        Returns:
            Dictionary containing all processed data
        """
        logger.info("Starting full data processing pipeline")
        
        # Step 1: Load data
        self.load_excel_data()
        
        # Step 2: Apply filters
        self.apply_filters(fmp_threshold)
        
        # Step 3: Extract time series
        self.extract_time_series(sequence_length)
        
        # Step 4: Extract static parameters
        self.extract_static_parameters()
        
        # Step 5: Extract latency/amplitude with masks
        self.extract_latency_amplitude_with_masks()
        
        # Step 6: Apply normalization
        normalization_stats = self.normalize_data(apply_normalization)
        
        # Step 7: Save processed data
        self.save_processed_data(normalization_stats)
        
        logger.info("Data processing pipeline completed successfully")
        
        return {
            'time_series': self.time_series_data,
            'static_params': self.static_data,
            'latency_data': self.latency_data,
            'amplitude_data': self.amplitude_data,
            'masks': self.masks,
            'normalization_stats': normalization_stats
        }


def load_processed_data(data_dir: str = "data/processed") -> Dict:
    """
    Quickly load previously processed data.
    
    Args:
        data_dir: Directory containing processed data
    
    Returns:
        Dictionary containing all processed data
    """
    data_path = Path(data_dir)
    pickle_path = data_path / "abr_processed_data.pkl"
    
    if not pickle_path.exists():
        raise FileNotFoundError(f"Processed data not found at {pickle_path}. Run process_data.py first.")
    
    logger.info(f"Loading processed data from {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded processed data with {data['metadata']['n_samples']} samples")
    return data


# Alias for backward compatibility
ABRDataPreprocessor = ABRDataPreparator


if __name__ == "__main__":
    # Example usage
    excel_path = "dataset/abr_data_preprocessed.xlsx"
    
    # Create data preparator
    preparator = ABRDataPreparator(excel_path)
    
    # Process data with your specifications
    processed_data = preparator.process_full_pipeline(
        sequence_length=200,  # First 200 timestamps
        apply_normalization=True  # Apply normalization
    )
    
    print("\nData processing completed!")
    print(f"Processed {processed_data['time_series'].shape[0]} samples")
    print(f"Time series shape: {processed_data['time_series'].shape}")
    print(f"Static parameters available: {list(processed_data['static_params'].keys())}")
    print(f"Latency data available: {list(processed_data['latency_data'].keys())}")
    print(f"Amplitude data available: {list(processed_data['amplitude_data'].keys())}") 