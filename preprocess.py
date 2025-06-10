#!/usr/bin/env python3
"""
ABR Data Preprocessing Script
============================

Main script for preprocessing raw ABR data for CVAE training.

Usage:
    python preprocess.py [--input data/raw] [--output data/processed]

Example:
    python preprocess.py --input dataset/abr_data.xlsx --output data/processed
"""

import argparse
import logging
from pathlib import Path

from src.data.preprocessing import ABRDataPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess ABR Data')
    parser.add_argument('--input', type=str, default='data/abr_data_preprocessed.xlsx',
                       help='Path to input data file')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--sequence-length', type=int, default=200,
                       help='Length of time series to extract')
    parser.add_argument('--min-fmp', type=float, default=2.0,
                       help='Minimum FMP value for filtering')
    parser.add_argument('--polarity', type=str, default='alternate',
                       help='Stimulus polarity to include')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Preprocessing ABR data from {args.input}")
    logger.info(f"Output directory: {output_path}")
    
    # Initialize preprocessor
    preprocessor = ABRDataPreprocessor(args.input, args.output_dir)
    
    # Use the full pipeline method which handles all steps
    logger.info("Running full data processing pipeline...")
    processed_data = preprocessor.process_full_pipeline(
        sequence_length=args.sequence_length,
        apply_normalization=True
    )
    
    # Print summary
    print("\n" + "="*60)
    print("ABR DATA PREPROCESSING COMPLETED")
    print("="*60)
    
    print(f"\n📊 Data Statistics:")
    print(f"   Total samples: {processed_data['time_series'].shape[0]}")
    print(f"   Time series length: {processed_data['time_series'].shape[1]}")
    print(f"   Static parameters: {len(processed_data['static_params'])}")
    print(f"   Latency features: {len(processed_data['latency_data'])}")
    print(f"   Amplitude features: {len(processed_data['amplitude_data'])}")
    
    print(f"\n📁 Output files saved to: {output_path}")
    print(f"   • abr_time_series.h5")
    print(f"   • metadata.json") 
    print(f"   • abr_processed_data.pkl")
    
    print(f"\n🎯 Next Steps:")
    print("   • Review data statistics and distributions")
    print("   • Train CVAE model with processed data")
    print("   • Validate preprocessing quality")


if __name__ == "__main__":
    main() 