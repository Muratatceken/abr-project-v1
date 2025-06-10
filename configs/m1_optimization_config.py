"""
M1 Chip Optimization Configuration for ABR Project

This module provides M1-specific optimizations for:
- Device selection (MPS vs CPU)
- Memory management
- Model architecture optimizations
- Training configurations
- Data loading optimizations
"""

import torch
import platform
import psutil
from typing import Dict, Any, Optional
import warnings

class M1OptimizationConfig:
    """Configuration class optimized for Apple M1/M2/M3 chips."""
    
    def __init__(self):
        self.device_info = self._detect_device_capabilities()
        self.memory_info = self._get_memory_info()
        self.optimal_config = self._generate_optimal_config()
    
    def _detect_device_capabilities(self) -> Dict[str, Any]:
        """Detect M1 chip capabilities and available devices."""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'is_apple_silicon': platform.machine() == 'arm64',
            'mps_available': torch.backends.mps.is_available(),
            'mps_built': torch.backends.mps.is_built(),
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': torch.get_num_threads(),
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True)
        }
        return info
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percentage': memory.percent
        }
    
    def get_optimal_device(self) -> torch.device:
        """Get the optimal device for M1 chip."""
        # Temporarily disable MPS due to compatibility issues with some operations
        # TODO: Re-enable when PyTorch MPS support is more stable
        if False and self.device_info['mps_available'] and self.device_info['is_apple_silicon']:
            print("🚀 Using MPS (Metal Performance Shaders) for M1 acceleration")
            return torch.device('mps')
        elif self.device_info['cuda_available']:
            print("🔥 Using CUDA")
            return torch.device('cuda')
        else:
            print("💻 Using CPU (MPS disabled for stability)")
            return torch.device('cpu')
    
    def _generate_optimal_config(self) -> Dict[str, Any]:
        """Generate optimal configuration for M1 chip."""
        device = self.get_optimal_device()
        
        # Base configuration optimized for M1
        config = {
            'device': device,
            'device_type': str(device),
            
            # Memory optimizations for M1
            'memory': {
                'pin_memory': device.type == 'cpu',  # Only for CPU, not needed for MPS
                'non_blocking': device.type in ['cuda', 'mps'],
                'max_memory_fraction': 0.8,  # Use 80% of available memory
                'empty_cache_frequency': 10,  # Clear cache every 10 batches
            },
            
            # Data loading optimizations
            'data_loading': {
                'num_workers': min(4, self.device_info['physical_cores']),  # Optimal for M1
                'prefetch_factor': 2,
                'persistent_workers': True,
                'drop_last': True,  # For consistent batch sizes
            },
            
            # Model optimizations
            'model': {
                'mixed_precision': device.type in ['cuda', 'mps'],
                'compile_model': True,  # Use torch.compile for M1 optimization
                'channels_last': False,  # Not always beneficial for time series
                'gradient_checkpointing': True,  # Save memory
            },
            
            # Training optimizations
            'training': {
                'batch_size': self._get_optimal_batch_size(),
                'accumulation_steps': 1,
                'max_grad_norm': 1.0,
                'warmup_steps': 1000,
                'scheduler_type': 'cosine_with_restarts',
                'optimizer': 'adamw',
                'weight_decay': 0.01,
                'learning_rate': 3e-4,
                'beta1': 0.9,
                'beta2': 0.999,
                'eps': 1e-8,
            },
            
            # M1-specific optimizations
            'm1_specific': {
                'use_metal_performance_shaders': device.type == 'mps',
                'optimize_for_inference': False,
                'use_torch_compile': True,
                'enable_nested_tensor': False,  # Not always stable
                'use_fused_adam': True,
            }
        }
        
        return config
    
    def _get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory."""
        available_memory_gb = self.memory_info['available_gb']
        
        # Conservative batch size calculation for M1
        if available_memory_gb >= 16:
            return 32
        elif available_memory_gb >= 8:
            return 16
        else:
            return 8
    
    def setup_torch_optimizations(self):
        """Setup PyTorch optimizations for M1."""
        # Set number of threads for optimal performance
        torch.set_num_threads(self.device_info['physical_cores'])
        
        # Enable optimizations
        if hasattr(torch.backends, 'mps'):
            # MPS optimizations
            torch.backends.mps.allow_tf32 = True
        
        # General optimizations
        torch.backends.cudnn.benchmark = False  # Not applicable for MPS
        torch.backends.cudnn.deterministic = True
        
        # Set memory allocation strategy
        if self.optimal_config['device'].type == 'mps':
            # MPS-specific memory management (only if available)
            try:
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            except (AttributeError, RuntimeError):
                # MPS cache clearing not available in this PyTorch version
                pass
        
        print("🔧 M1 optimizations applied successfully!")
    
    def get_model_config_overrides(self) -> Dict[str, Any]:
        """Get model configuration overrides for M1 optimization."""
        return {
            'static_encoder': {
                'input_dim': 10,
                'output_dim': 256,
                'use_layer_norm': True,  # Better for M1
                'activation': 'gelu',    # Optimized for M1
            },
            
            'abr_transformer': {
                'input_dim': 2,
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'max_len': 2048,
                'use_flash_attention': False,  # Not available on MPS
                'use_rotary_embeddings': True,
                'layer_norm_eps': 1e-6,
            },
            
            'state_space': {
                'hidden_dim': 256,
                'input_dim': 256,
                'ts_dim': 2,
                'dropout': 0.1,
                'nonlinear_layers': 2,
                'use_glu': True,  # Gated Linear Units work well on M1
            },
            
            'diffusion': {
                'hidden_dim': 256,
                'timesteps': 1000,
                'min_beta': 1e-4,
                'max_beta': 0.02,
                'schedule_type': 'cosine',
                'dropout': 0.1,
                'groups': 8,
                'use_group_norm': True,  # Better than batch norm for M1
            },
            
            'peak_detection': {
                'hidden_dim': 256,
                'num_peaks': 3,
                'dropout': 0.1,
                'conv_kernel_size': 5,
                'use_separable_conv': True,  # More efficient on M1
            }
        }
    
    def get_training_config_overrides(self) -> Dict[str, Any]:
        """Get training configuration overrides for M1."""
        return {
            'device': str(self.optimal_config['device']),
            'batch_size': self.optimal_config['training']['batch_size'],
            'learning_rate': self.optimal_config['training']['learning_rate'],
            'weight_decay': self.optimal_config['training']['weight_decay'],
            'max_epochs': 100,
            'grad_clip': self.optimal_config['training']['max_grad_norm'],
            'warmup_steps': self.optimal_config['training']['warmup_steps'],
            'use_amp': self.optimal_config['model']['mixed_precision'],
            'compile_model': self.optimal_config['model']['compile_model'],
            
            # M1-specific training settings
            'dataloader_config': {
                'num_workers': self.optimal_config['data_loading']['num_workers'],
                'pin_memory': self.optimal_config['memory']['pin_memory'],
                'persistent_workers': self.optimal_config['data_loading']['persistent_workers'],
                'prefetch_factor': self.optimal_config['data_loading']['prefetch_factor'],
            },
            
            'optimizer_config': {
                'type': 'adamw',
                'betas': (self.optimal_config['training']['beta1'], 
                         self.optimal_config['training']['beta2']),
                'eps': self.optimal_config['training']['eps'],
                'fused': self.optimal_config['m1_specific']['use_fused_adam'],
            },
            
            'scheduler_config': {
                'type': self.optimal_config['training']['scheduler_type'],
                'min_lr': 1e-6,
                'restart_period': 10,
            }
        }
    
    def print_system_info(self):
        """Print detailed system information."""
        print("🖥️  SYSTEM INFORMATION")
        print("=" * 50)
        print(f"Platform: {self.device_info['platform']}")
        print(f"Processor: {self.device_info['processor']}")
        print(f"Architecture: {self.device_info['machine']}")
        print(f"Apple Silicon: {self.device_info['is_apple_silicon']}")
        print(f"Physical Cores: {self.device_info['physical_cores']}")
        print(f"Logical Cores: {self.device_info['logical_cores']}")
        
        print(f"\n💾 MEMORY INFORMATION")
        print("=" * 50)
        print(f"Total Memory: {self.memory_info['total_gb']:.1f} GB")
        print(f"Available Memory: {self.memory_info['available_gb']:.1f} GB")
        print(f"Used Memory: {self.memory_info['used_gb']:.1f} GB")
        print(f"Memory Usage: {self.memory_info['percentage']:.1f}%")
        
        print(f"\n🔥 PYTORCH DEVICE INFO")
        print("=" * 50)
        print(f"MPS Available: {self.device_info['mps_available']}")
        print(f"MPS Built: {self.device_info['mps_built']}")
        print(f"CUDA Available: {self.device_info['cuda_available']}")
        print(f"Optimal Device: {self.optimal_config['device']}")
        print(f"Recommended Batch Size: {self.optimal_config['training']['batch_size']}")
        print(f"Recommended Workers: {self.optimal_config['data_loading']['num_workers']}")

def get_m1_config() -> M1OptimizationConfig:
    """Get M1 optimization configuration."""
    return M1OptimizationConfig()

def setup_m1_environment():
    """Setup environment for optimal M1 performance."""
    config = get_m1_config()
    config.setup_torch_optimizations()
    return config

# M1-specific utility functions
def move_to_device(data, device):
    """Move data to device with M1 optimizations."""
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    elif hasattr(data, 'to'):
        return data.to(device, non_blocking=(device.type in ['cuda', 'mps']))
    else:
        return data

def clear_memory_cache(device):
    """Clear memory cache for the given device."""
    if device.type == 'mps':
        try:
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        except (AttributeError, RuntimeError):
            # MPS cache clearing not available in this PyTorch version
            pass
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

def get_memory_usage(device):
    """Get current memory usage for the device."""
    if device.type == 'mps':
        # MPS doesn't have direct memory query, use system memory
        memory = psutil.virtual_memory()
        return {
            'allocated': memory.used / (1024**3),
            'cached': 0,  # Not available for MPS
            'total': memory.total / (1024**3)
        }
    elif device.type == 'cuda':
        return {
            'allocated': torch.cuda.memory_allocated(device) / (1024**3),
            'cached': torch.cuda.memory_reserved(device) / (1024**3),
            'total': torch.cuda.get_device_properties(device).total_memory / (1024**3)
        }
    else:
        memory = psutil.virtual_memory()
        return {
            'allocated': memory.used / (1024**3),
            'cached': 0,
            'total': memory.total / (1024**3)
        }

if __name__ == "__main__":
    # Test the M1 configuration
    config = setup_m1_environment()
    config.print_system_info()
    
    print(f"\n🚀 M1 OPTIMIZATION READY!")
    print("Use this configuration in your ABR model training for optimal M1 performance.") 