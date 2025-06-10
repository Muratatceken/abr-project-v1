import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.signal import find_peaks, correlate
import pandas as pd
from tqdm import tqdm

from ..models.cvae_model import ConditionalVAEWithMasking
from ..models.advanced_cvae_model import AdvancedConditionalVAE
from ..data.dataset import ABRMaskedDataset

logger = logging.getLogger(__name__)

class CVAEEvaluator:
    """
    Comprehensive evaluator for Conditional VAE models.
    
    Provides various metrics and visualizations for:
    - Reconstruction quality
    - Generation quality
    - Latent space analysis
    - Condition interpolation
    - Statistical comparisons
    """
    
    def __init__(self, 
                 model,  # Union[ConditionalVAEWithMasking, AdvancedConditionalVAE]
                 device: torch.device,
                 output_dir: str = "evaluation_results"):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained CVAE model (original or advanced)
            device: Device for computation
            output_dir: Directory to save results
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect model type
        self.model_type = 'advanced' if isinstance(model, AdvancedConditionalVAE) else 'original'
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.metrics_dir = self.output_dir / "metrics"
        self.samples_dir = self.output_dir / "samples"
        
        for dir_path in [self.plots_dir, self.metrics_dir, self.samples_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        logger.info(f"CVAE Evaluator initialized for {self.model_type} model, output dir: {self.output_dir}")
    
    def evaluate_reconstruction(self, 
                              test_loader: torch.utils.data.DataLoader,
                              num_samples: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate reconstruction quality on test set.
        
        Args:
            test_loader: Test data loader
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary of reconstruction metrics
        """
        logger.info("Evaluating reconstruction quality...")
        
        self.model.eval()
        
        all_originals = []
        all_reconstructions = []
        all_static_params = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Reconstruction evaluation")):
                if num_samples and len(all_originals) >= num_samples:
                    break
                
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                    elif isinstance(batch[key], dict):
                        for subkey in batch[key]:
                            batch[key][subkey] = batch[key][subkey].to(self.device)
                
                abr_data = batch['time_series']
                
                # Forward pass
                outputs = self.model(batch)
                reconstructions = outputs['abr_reconstruction']
                
                # Store results
                all_originals.append(abr_data.cpu().numpy())
                all_reconstructions.append(reconstructions.cpu().numpy())
                # Extract static params for storage
                static_tensor = torch.stack([
                    batch['static_params']['age'],
                    batch['static_params']['intensity'], 
                    batch['static_params']['stimulus_rate'],
                    batch['static_params']['hear_loss']
                ], dim=1)
                all_static_params.append(static_tensor.cpu().numpy())
        
        # Concatenate all results
        originals = np.concatenate(all_originals, axis=0)
        reconstructions = np.concatenate(all_reconstructions, axis=0)
        static_params = np.concatenate(all_static_params, axis=0)
        
        # Compute metrics
        metrics = self._compute_reconstruction_metrics(originals, reconstructions)
        
        # Save metrics
        self._save_metrics(metrics, "reconstruction_metrics.json")
        
        # Create visualizations
        self._plot_reconstruction_examples(originals, reconstructions, static_params)
        self._plot_reconstruction_distribution(originals, reconstructions)
        
        # Create static parameter visualization table
        self._create_static_parameter_visualization_table(test_loader, num_samples=min(50, len(originals)))
        
        logger.info("Reconstruction evaluation completed")
        return metrics
    
    def evaluate_generation(self,
                          test_dataset: ABRMaskedDataset,
                          num_samples: int = 1000,
                          samples_per_condition: int = 5) -> Dict[str, float]:
        """
        Evaluate generation quality with comprehensive clinical parameter tracking.
        
        Args:
            test_dataset: Test dataset for condition sampling
            num_samples: Number of samples to generate
            samples_per_condition: Number of samples per condition
            
        Returns:
            Dictionary of generation metrics
        """
        logger.info("Evaluating generation quality with clinical parameter tracking...")
        
        self.model.eval()
        
        # Sample conditions from test set
        condition_indices = np.random.choice(len(test_dataset), 
                                           size=min(num_samples // samples_per_condition, len(test_dataset)),
                                           replace=False)
        
        all_generated_data = []
        all_real_data = []
        generation_summary = []
        
        with torch.no_grad():
            for idx in tqdm(condition_indices, desc="Generation evaluation"):
                sample = test_dataset[idx]
                
                # Prepare batch for generation
                batch = {
                    'static_params': {
                        'age': sample['static_params']['age'].unsqueeze(0),
                        'intensity': sample['static_params']['intensity'].unsqueeze(0),
                        'stimulus_rate': sample['static_params']['stimulus_rate'].unsqueeze(0),
                        'hear_loss': sample['static_params']['hear_loss'].unsqueeze(0)
                    },
                    'latency_data': {key: val.unsqueeze(0) for key, val in sample['latency_data'].items()},
                    'latency_masks': {key: val.unsqueeze(0) for key, val in sample['latency_masks'].items()},
                    'amplitude_data': {key: val.unsqueeze(0) for key, val in sample['amplitude_data'].items()},
                    'amplitude_masks': {key: val.unsqueeze(0) for key, val in sample['amplitude_masks'].items()}
                }
                
                # Move to device
                for key in batch:
                    if isinstance(batch[key], dict):
                        for subkey in batch[key]:
                            batch[key][subkey] = batch[key][subkey].to(self.device)
                
                # Generate comprehensive samples with causal static parameters
                generated_results = self.model.generate(batch, num_samples=samples_per_condition, use_conditioning=True)
                
                # Extract peaks from generated waveforms
                waveforms = generated_results['abr_waveforms'].cpu().numpy()
                
                # Handle peak extraction based on model type
                if self.model_type == 'advanced' and 'peak_predictions' in generated_results:
                    # Advanced model has explicit peak predictions
                    peak_predictions = generated_results['peak_predictions']
                    extracted_peaks = {
                        'latencies': peak_predictions['latencies'].cpu().numpy(),
                        'amplitudes': peak_predictions['amplitudes'].cpu().numpy(),
                        'confidence': peak_predictions['presence_probs'].cpu().numpy()
                    }
                elif hasattr(self.model, '_extract_abr_peaks'):
                    # Original model with peak extraction method
                    extracted_peaks = self.model._extract_abr_peaks(waveforms)
                else:
                    # Fallback: simple peak detection
                    extracted_peaks = self._detect_abr_peaks(waveforms)
                
                # Store results
                all_generated_data.append({
                    'waveforms': waveforms,
                    'generated_static_params': {
                        'age': generated_results['generated_static_params']['age'].cpu().numpy(),
                        'intensity': generated_results['generated_static_params']['intensity'].cpu().numpy(),
                        'stimulus_rate': generated_results['generated_static_params']['stimulus_rate'].cpu().numpy(),
                        'hear_loss': generated_results['generated_static_params']['hear_loss'].cpu().numpy()
                    },
                    'generated_latencies': generated_results.get('generated_latencies', torch.zeros(samples_per_condition, 3)).cpu().numpy(),
                    'generated_amplitudes': generated_results.get('generated_amplitudes', torch.zeros(samples_per_condition, 3)).cpu().numpy(),
                    'extracted_latencies': extracted_peaks['latencies'],
                    'extracted_amplitudes': extracted_peaks['amplitudes'],
                    'peak_confidence': extracted_peaks['confidence']
                })
                
                all_real_data.append({
                    'waveform': sample['time_series'].numpy(),
                    'static_params': np.array([sample['static_params']['age'], sample['static_params']['intensity'], 
                                           sample['static_params']['stimulus_rate'], sample['static_params']['hear_loss']]),
                    'real_latencies': np.array([sample['latency_data']['I Latancy'], 
                                              sample['latency_data']['III Latancy'],
                                              sample['latency_data']['V Latancy']]),
                    'real_amplitudes': np.array([sample['amplitude_data']['I Amplitude'],
                                               sample['amplitude_data']['III Amplitude'], 
                                               sample['amplitude_data']['V Amplitude']])
                })
                
                # Create detailed summary for this condition
                condition_summary = self._create_generation_summary(
                    generated_results, sample, idx, samples_per_condition, extracted_peaks
                )
                generation_summary.extend(condition_summary)
        
        # Compute comprehensive metrics
        metrics = self._compute_comprehensive_generation_metrics(all_generated_data, all_real_data)
        
        # Save detailed generation table
        self._save_generation_table(generation_summary)
        
        # Save metrics
        self._save_metrics(metrics, "comprehensive_generation_metrics.json")
        
        # Create enhanced visualizations
        self._plot_comprehensive_generation_examples(all_generated_data, all_real_data)
        self._plot_clinical_parameter_comparison(all_generated_data, all_real_data)
        
        logger.info("Comprehensive generation evaluation completed")
        return metrics
    
    def evaluate_pure_generation(self,
                               test_dataset: ABRMaskedDataset,
                               num_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate pure generation quality (causal static parameter generation).
        
        Args:
            test_dataset: Test dataset for comparison (not used for conditioning)
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary of generation metrics
        """
        logger.info(f"Evaluating pure generation quality with {num_samples} samples...")
        logger.info("Using causal static parameter generation (no conditioning)")
        
        self.model.eval()
        
        all_generated_data = []
        all_real_data = []
        generation_summary = []
        
        with torch.no_grad():
            # Generate samples in batches for efficiency
            batch_size = 50
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(range(num_batches), desc="Pure generation"):
                current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
                
                # Generate samples without conditioning
                generated_results = self.model.generate(
                    num_samples=current_batch_size, 
                    use_conditioning=False
                )
                
                # Extract peaks from generated waveforms
                waveforms = generated_results['abr_waveforms'].cpu().numpy()
                
                # Handle peak extraction based on model type
                if self.model_type == 'advanced' and 'peak_predictions' in generated_results:
                    # Advanced model has explicit peak predictions
                    peak_predictions = generated_results['peak_predictions']
                    extracted_peaks = {
                        'latencies': peak_predictions['latencies'].cpu().numpy(),
                        'amplitudes': peak_predictions['amplitudes'].cpu().numpy(),
                        'confidence': peak_predictions['presence_probs'].cpu().numpy()
                    }
                elif hasattr(self.model, '_extract_abr_peaks'):
                    # Original model with peak extraction method
                    extracted_peaks = self.model._extract_abr_peaks(waveforms)
                else:
                    # Fallback: simple peak detection
                    extracted_peaks = self._detect_abr_peaks(waveforms)
                
                # Store generated data
                for i in range(current_batch_size):
                    gen_data = {
                        'waveform': waveforms[i],
                        'generated_static_params': {
                            'age': generated_results['generated_static_params']['age'][i].cpu().numpy(),
                            'intensity': generated_results['generated_static_params']['intensity'][i].cpu().numpy(),
                            'stimulus_rate': generated_results['generated_static_params']['stimulus_rate'][i].cpu().numpy(),
                            'hear_loss': generated_results['generated_static_params']['hear_loss'][i].cpu().numpy()
                        },
                        'latent_code': generated_results['latent_codes'][i].cpu().numpy(),
                        'extracted_latencies': extracted_peaks['latencies'][i],
                        'extracted_amplitudes': extracted_peaks['amplitudes'][i],
                        'peak_confidence': extracted_peaks['confidence'][i]
                    }
                    
                    # Add generated clinical features if available
                    if 'generated_latencies' in generated_results:
                        gen_data['generated_latencies'] = generated_results['generated_latencies'][i].cpu().numpy()
                        gen_data['generated_amplitudes'] = generated_results['generated_amplitudes'][i].cpu().numpy()
                    
                    all_generated_data.append(gen_data)
            
            # Sample real data for comparison
            num_real_samples = min(num_samples, len(test_dataset))
            real_indices = np.random.choice(len(test_dataset), size=num_real_samples, replace=False)
            
            for idx in real_indices:
                sample = test_dataset[idx]
                all_real_data.append({
                    'waveform': sample['time_series'].numpy(),
                    'static_params': np.array([sample['static_params']['age'], sample['static_params']['intensity'],
                                           sample['static_params']['stimulus_rate'], sample['static_params']['hear_loss']]),
                    'real_latencies': np.array([sample['latency_data']['I Latancy'], 
                                              sample['latency_data']['III Latancy'],
                                              sample['latency_data']['V Latancy']]),
                    'real_amplitudes': np.array([sample['amplitude_data']['I Amplitude'],
                                               sample['amplitude_data']['III Amplitude'], 
                                               sample['amplitude_data']['V Amplitude']])
                })
        
        # Compute comprehensive metrics for pure generation
        metrics = self._compute_pure_generation_metrics(all_generated_data, all_real_data)
        
        # Save detailed generation table
        self._save_pure_generation_table(all_generated_data)
        
        # Save metrics
        self._save_metrics(metrics, "pure_generation_metrics.json")
        
        # Create enhanced visualizations
        self._plot_pure_generation_examples(all_generated_data, all_real_data)
        self._plot_generated_static_parameters(all_generated_data, all_real_data)
        
        logger.info("Pure generation evaluation completed")
        return metrics
    
    def analyze_latent_space(self,
                           test_loader: torch.utils.data.DataLoader,
                           num_samples: int = 1000) -> Dict[str, Any]:
        """
        Analyze the learned latent space.
        
        Args:
            test_loader: Test data loader
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary of latent space analysis results
        """
        logger.info("Analyzing latent space...")
        
        self.model.eval()
        
        all_latents = []
        all_conditions = []
        all_static_params = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Latent space analysis"):
                if len(all_latents) * batch['abr_waveform'].size(0) >= num_samples:
                    break
                
                abr_data = batch['abr_waveform'].to(self.device)
                static_params = batch['static_params'].to(self.device)
                
                # Encode to latent space
                mu, logvar = self.model.encoder(abr_data)
                z = self.model.reparameterize(mu, logvar)
                condition = self.model.condition_encoder(static_params)
                
                all_latents.append(z.cpu().numpy())
                all_conditions.append(condition.cpu().numpy())
                all_static_params.append(static_params.cpu().numpy())
        
        # Concatenate results
        latents = np.concatenate(all_latents, axis=0)[:num_samples]
        conditions = np.concatenate(all_conditions, axis=0)[:num_samples]
        static_params = np.concatenate(all_static_params, axis=0)[:num_samples]
        
        # Analyze latent space
        analysis = self._analyze_latent_distributions(latents, conditions, static_params)
        
        # Create visualizations
        self._plot_latent_space(latents, conditions, static_params)
        self._plot_latent_interpolation()
        
        logger.info("Latent space analysis completed")
        return analysis
    
    def evaluate_condition_interpolation(self,
                                       test_dataset: ABRMaskedDataset,
                                       num_interpolations: int = 10) -> Dict[str, Any]:
        """
        Evaluate condition interpolation quality.
        
        Args:
            test_dataset: Test dataset
            num_interpolations: Number of interpolation examples
            
        Returns:
            Interpolation analysis results
        """
        logger.info("Evaluating condition interpolation...")
        
        self.model.eval()
        
        interpolation_results = []
        
        for i in range(num_interpolations):
            # Sample two random conditions
            idx1, idx2 = np.random.choice(len(test_dataset), size=2, replace=False)
            
            sample1 = test_dataset[idx1]
            sample2 = test_dataset[idx2]
            
            static1 = sample1['static_params'].unsqueeze(0).to(self.device)
            static2 = sample2['static_params'].unsqueeze(0).to(self.device)
            
            # Perform interpolation
            interpolated = self.model.interpolate(static1, static2, num_steps=10)
            
            interpolation_results.append({
                'condition1': static1.cpu().numpy(),
                'condition2': static2.cpu().numpy(),
                'interpolated': interpolated.cpu().numpy()
            })
        
        # Analyze interpolation smoothness
        analysis = self._analyze_interpolation_smoothness(interpolation_results)
        
        # Create visualizations
        self._plot_interpolation_examples(interpolation_results)
        
        logger.info("Condition interpolation evaluation completed")
        return analysis
    
    def _compute_reconstruction_metrics(self, 
                                      originals: np.ndarray, 
                                      reconstructions: np.ndarray) -> Dict[str, float]:
        """Compute reconstruction quality metrics."""
        metrics = {}
        
        # Mean Squared Error
        mse = mean_squared_error(originals.flatten(), reconstructions.flatten())
        metrics['mse'] = float(mse)
        
        # Mean Absolute Error
        mae = mean_absolute_error(originals.flatten(), reconstructions.flatten())
        metrics['mae'] = float(mae)
        
        # Peak Signal-to-Noise Ratio
        psnr = 20 * np.log10(np.max(originals) / np.sqrt(mse))
        metrics['psnr'] = float(psnr)
        
        # Structural Similarity (simplified)
        correlations = []
        for i in range(len(originals)):
            if originals.ndim == 3:  # Multi-channel data
                for ch in range(originals.shape[2]):
                    corr = np.corrcoef(originals[i, :, ch], reconstructions[i, :, ch])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            else:  # Single channel data
                corr = np.corrcoef(originals[i, :], reconstructions[i, :])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        metrics['mean_correlation'] = float(np.mean(correlations))
        metrics['std_correlation'] = float(np.std(correlations))
        
        # Spectral similarity
        spectral_similarities = []
        for i in range(min(100, len(originals))):  # Sample for efficiency
            if originals.ndim == 3:  # Multi-channel data
                for ch in range(originals.shape[2]):
                    orig_fft = np.abs(np.fft.fft(originals[i, :, ch]))
                    recon_fft = np.abs(np.fft.fft(reconstructions[i, :, ch]))
                    
                    # Normalize
                    orig_fft = orig_fft / (np.sum(orig_fft) + 1e-8)
                    recon_fft = recon_fft / (np.sum(recon_fft) + 1e-8)
                    
                    # Compute similarity
                    similarity = 1 - wasserstein_distance(orig_fft, recon_fft)
                    spectral_similarities.append(similarity)
            else:  # Single channel data
                orig_fft = np.abs(np.fft.fft(originals[i, :]))
                recon_fft = np.abs(np.fft.fft(reconstructions[i, :]))
                
                # Normalize
                orig_fft = orig_fft / (np.sum(orig_fft) + 1e-8)
                recon_fft = recon_fft / (np.sum(recon_fft) + 1e-8)
                
                # Compute similarity
                similarity = 1 - wasserstein_distance(orig_fft, recon_fft)
                spectral_similarities.append(similarity)
        
        metrics['spectral_similarity'] = float(np.mean(spectral_similarities))
        
        return metrics
    
    def _compute_generation_metrics(self, 
                                  generated: np.ndarray, 
                                  real: np.ndarray) -> Dict[str, float]:
        """Compute generation quality metrics."""
        metrics = {}
        
        # Statistical moments comparison
        for moment, func in [('mean', np.mean), ('std', np.std), ('skew', stats.skew), ('kurtosis', stats.kurtosis)]:
            real_moment = func(real.flatten())
            gen_moment = func(generated.flatten())
            metrics[f'{moment}_real'] = float(real_moment)
            metrics[f'{moment}_generated'] = float(gen_moment)
            metrics[f'{moment}_difference'] = float(abs(real_moment - gen_moment))
        
        # Wasserstein distance
        try:
            wd = wasserstein_distance(real.flatten(), generated.flatten())
            metrics['wasserstein_distance'] = float(wd)
        except:
            metrics['wasserstein_distance'] = float('inf')
        
        # Peak detection comparison
        real_peaks = self._detect_abr_peaks(real)
        gen_peaks = self._detect_abr_peaks(generated)
        
        # Count valid peaks (confidence > 0)
        real_peak_count = np.mean(np.sum(real_peaks['confidence'] > 0, axis=1))
        gen_peak_count = np.mean(np.sum(gen_peaks['confidence'] > 0, axis=1))
        
        metrics['real_peak_count'] = float(real_peak_count)
        metrics['generated_peak_count'] = float(gen_peak_count)
        
        # Frequency domain analysis
        real_power = np.mean([np.sum(np.abs(np.fft.fft(sample.flatten()))**2) for sample in real])
        gen_power = np.mean([np.sum(np.abs(np.fft.fft(sample.flatten()))**2) for sample in generated])
        
        metrics['real_power'] = float(real_power)
        metrics['generated_power'] = float(gen_power)
        metrics['power_ratio'] = float(gen_power / (real_power + 1e-8))
        
        return metrics
    
    def _detect_abr_peaks(self, waveforms: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect peaks in ABR waveforms and return in the expected format."""
        num_samples = len(waveforms)
        latencies = np.zeros((num_samples, 3))  # I, III, V peaks
        amplitudes = np.zeros((num_samples, 3))
        confidence = np.zeros((num_samples, 3))
        
        # Typical ABR peak timing windows (in samples, assuming 200 points for ~10ms)
        peak_windows = {
            'I': (10, 40),    # ~0.5-2ms
            'III': (60, 100), # ~3-5ms  
            'V': (120, 180)   # ~6-9ms
        }
        
        for i, waveform in enumerate(waveforms):
            # Use first channel for peak detection
            signal = waveform if waveform.ndim == 1 else waveform
            
            for peak_idx, (peak_name, (start, end)) in enumerate(peak_windows.items()):
                # Find peaks in the specified window
                if end <= len(signal):
                    window_signal = signal[start:end]
                    peaks, properties = find_peaks(
                        window_signal, 
                        height=np.std(window_signal) * 0.5,
                        distance=5
                    )
                    
                    if len(peaks) > 0:
                        # Take the highest peak
                        peak_heights = properties['peak_heights']
                        max_peak_idx = np.argmax(peak_heights)
                        peak_position = peaks[max_peak_idx]
                        
                        # Convert to absolute position
                        absolute_position = start + peak_position
                        latencies[i, peak_idx] = absolute_position
                        amplitudes[i, peak_idx] = peak_heights[max_peak_idx]
                        confidence[i, peak_idx] = peak_heights[max_peak_idx] / np.max(np.abs(signal))
                    else:
                        # No peak found
                        latencies[i, peak_idx] = -1
                        amplitudes[i, peak_idx] = 0
                        confidence[i, peak_idx] = 0
        
        return {
            'latencies': latencies,
            'amplitudes': amplitudes,
            'confidence': confidence
        }
    
    def _analyze_latent_distributions(self, 
                                    latents: np.ndarray, 
                                    conditions: np.ndarray,
                                    static_params: np.ndarray) -> Dict[str, Any]:
        """Analyze latent space distributions."""
        analysis = {}
        
        # Basic statistics
        analysis['latent_mean'] = np.mean(latents, axis=0).tolist()
        analysis['latent_std'] = np.std(latents, axis=0).tolist()
        
        # Check for posterior collapse
        latent_vars = np.var(latents, axis=0)
        analysis['active_dimensions'] = int(np.sum(latent_vars > 0.01))
        analysis['total_dimensions'] = int(latents.shape[1])
        analysis['posterior_collapse_ratio'] = float(1 - analysis['active_dimensions'] / analysis['total_dimensions'])
        
        # Condition-latent correlation
        if static_params.shape[1] > 0:
            correlations = []
            for i in range(min(latents.shape[1], 10)):  # Check first 10 latent dims
                for j in range(static_params.shape[1]):
                    corr = np.corrcoef(latents[:, i], static_params[:, j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            analysis['max_condition_correlation'] = float(np.max(correlations)) if correlations else 0.0
            analysis['mean_condition_correlation'] = float(np.mean(correlations)) if correlations else 0.0
        
        return analysis
    
    def _analyze_interpolation_smoothness(self, interpolation_results: List[Dict]) -> Dict[str, float]:
        """Analyze smoothness of interpolations."""
        smoothness_scores = []
        
        for result in interpolation_results:
            interpolated = result['interpolated']
            
            # Compute smoothness as inverse of total variation
            for ch in range(interpolated.shape[2]):
                for step in range(interpolated.shape[0] - 1):
                    diff = np.mean(np.abs(interpolated[step+1, :, ch] - interpolated[step, :, ch]))
                    smoothness_scores.append(diff)
        
        return {
            'mean_smoothness': float(np.mean(smoothness_scores)),
            'std_smoothness': float(np.std(smoothness_scores))
        }
    
    def _plot_reconstruction_examples(self, 
                                    originals: np.ndarray, 
                                    reconstructions: np.ndarray,
                                    static_params: np.ndarray):
        """Plot reconstruction examples with static parameter information."""
        fig, axes = plt.subplots(4, 2, figsize=(16, 14))
        
        for i in range(4):
            if originals.ndim == 3:  # Multi-channel data
                # Use first channel for plotting
                orig_signal = originals[i, :, 0]
                recon_signal = reconstructions[i, :, 0]
            else:  # Single channel data
                orig_signal = originals[i, :]
                recon_signal = reconstructions[i, :]
            
            # Extract static parameters for this sample
            if static_params.shape[1] >= 4:
                age = static_params[i, 0]
                intensity = static_params[i, 1] 
                stimulus_rate = static_params[i, 2]
                hear_loss = int(static_params[i, 3])
                
                # Create parameter string
                param_str = f'Age: {age:.1f}, Int: {intensity:.1f}dB, Rate: {stimulus_rate:.1f}Hz, HL: {hear_loss}'
            else:
                param_str = 'Parameters: N/A'
            
            # Original vs Reconstruction
            axes[i, 0].plot(orig_signal, label='Original', alpha=0.8, linewidth=1.5)
            axes[i, 0].plot(recon_signal, label='Reconstruction', alpha=0.8, linewidth=1.5)
            axes[i, 0].set_title(f'Sample {i+1} - Reconstruction\n{param_str}', fontsize=10)
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_xlabel('Time (samples)')
            axes[i, 0].set_ylabel('Amplitude')
            
            # Difference
            diff = orig_signal - recon_signal
            mse = np.mean(diff**2)
            mae = np.mean(np.abs(diff))
            
            axes[i, 1].plot(diff, color='red', alpha=0.8, linewidth=1.5)
            axes[i, 1].set_title(f'Sample {i+1} - Reconstruction Error\nMSE: {mse:.4f}, MAE: {mae:.4f}', fontsize=10)
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_xlabel('Time (samples)')
            axes[i, 1].set_ylabel('Error')
            axes[i, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'reconstruction_examples.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_reconstruction_distribution(self, originals: np.ndarray, reconstructions: np.ndarray):
        """Plot distribution comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Amplitude distributions
        axes[0, 0].hist(originals.flatten(), bins=50, alpha=0.7, label='Original', density=True)
        axes[0, 0].hist(reconstructions.flatten(), bins=50, alpha=0.7, label='Reconstruction', density=True)
        axes[0, 0].set_title('Amplitude Distribution')
        axes[0, 0].set_xlabel('Amplitude')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(originals.flatten(), dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot: Original vs Normal')
        
        # Correlation plot
        sample_size = min(10000, len(originals.flatten()))
        indices = np.random.choice(len(originals.flatten()), sample_size, replace=False)
        
        axes[1, 0].scatter(originals.flatten()[indices], 
                          reconstructions.flatten()[indices], 
                          alpha=0.5, s=1)
        axes[1, 0].plot([originals.min(), originals.max()], 
                       [originals.min(), originals.max()], 'r--')
        axes[1, 0].set_xlabel('Original')
        axes[1, 0].set_ylabel('Reconstruction')
        axes[1, 0].set_title('Original vs Reconstruction')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error distribution
        errors = (originals - reconstructions).flatten()
        axes[1, 1].hist(errors, bins=50, alpha=0.7, density=True)
        axes[1, 1].set_title('Reconstruction Error Distribution')
        axes[1, 1].set_xlabel('Error')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'reconstruction_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_generation_examples(self, 
                                generated: np.ndarray, 
                                real: np.ndarray,
                                conditions: np.ndarray):
        """Plot generation examples with condition information."""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        for i in range(4):
            # Extract condition parameters for this sample
            if conditions.shape[1] >= 4:
                age = conditions[i, 0]
                intensity = conditions[i, 1]
                stimulus_rate = conditions[i, 2] 
                hear_loss = int(conditions[i, 3])
                
                condition_str = f'Age: {age:.1f}, Int: {intensity:.1f}dB, Rate: {stimulus_rate:.1f}Hz, HL: {hear_loss}'
            else:
                condition_str = 'Conditions: N/A'
            
            # Real sample
            axes[0, i].plot(real[i, :, 0], linewidth=1.5, color='blue')
            axes[0, i].set_title(f'Real Sample {i+1}\n{condition_str}', fontsize=9)
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].set_xlabel('Time (samples)')
            axes[0, i].set_ylabel('Amplitude')
            
            # Generated samples
            for j in range(2):
                gen_idx = i * 2 + j
                if gen_idx < len(generated):
                    axes[j+1, i].plot(generated[gen_idx, :, 0], linewidth=1.5, color='red', alpha=0.8)
                    axes[j+1, i].set_title(f'Generated {gen_idx+1}\n{condition_str}', fontsize=9)
                    axes[j+1, i].grid(True, alpha=0.3)
                    axes[j+1, i].set_xlabel('Time (samples)')
                    axes[j+1, i].set_ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'generation_examples.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_generation_statistics(self, generated: np.ndarray, real: np.ndarray):
        """Plot generation statistics comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Amplitude distributions
        axes[0, 0].hist(real.flatten(), bins=50, alpha=0.7, label='Real', density=True)
        axes[0, 0].hist(generated.flatten(), bins=50, alpha=0.7, label='Generated', density=True)
        axes[0, 0].set_title('Amplitude Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Power spectral density
        real_psd = np.mean([np.abs(np.fft.fft(sample[:, 0]))**2 for sample in real[:100]], axis=0)
        gen_psd = np.mean([np.abs(np.fft.fft(sample[:, 0]))**2 for sample in generated[:100]], axis=0)
        
        freqs = np.fft.fftfreq(len(real_psd))
        axes[0, 1].plot(freqs[:len(freqs)//2], real_psd[:len(freqs)//2], label='Real')
        axes[0, 1].plot(freqs[:len(freqs)//2], gen_psd[:len(freqs)//2], label='Generated')
        axes[0, 1].set_title('Power Spectral Density')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Statistical moments
        moments = ['mean', 'std', 'skew', 'kurtosis']
        real_moments = [np.mean(real.flatten()), np.std(real.flatten()), 
                       stats.skew(real.flatten()), stats.kurtosis(real.flatten())]
        gen_moments = [np.mean(generated.flatten()), np.std(generated.flatten()),
                      stats.skew(generated.flatten()), stats.kurtosis(generated.flatten())]
        
        x = np.arange(len(moments))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, real_moments, width, label='Real', alpha=0.8)
        axes[1, 0].bar(x + width/2, gen_moments, width, label='Generated', alpha=0.8)
        axes[1, 0].set_title('Statistical Moments Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(moments)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Autocorrelation
        real_autocorr = np.mean([np.correlate(sample[:, 0], sample[:, 0], mode='full') 
                                for sample in real[:50]], axis=0)
        gen_autocorr = np.mean([np.correlate(sample[:, 0], sample[:, 0], mode='full') 
                               for sample in generated[:50]], axis=0)
        
        center = len(real_autocorr) // 2
        lags = np.arange(-center, center + 1)
        
        axes[1, 1].plot(lags[center-50:center+51], real_autocorr[center-50:center+51], label='Real')
        axes[1, 1].plot(lags[center-50:center+51], gen_autocorr[center-50:center+51], label='Generated')
        axes[1, 1].set_title('Autocorrelation Comparison')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'generation_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_latent_space(self, latents: np.ndarray, conditions: np.ndarray, static_params: np.ndarray):
        """Plot latent space analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Latent distribution
        axes[0, 0].hist(latents.flatten(), bins=50, alpha=0.7, density=True)
        axes[0, 0].set_title('Latent Space Distribution')
        axes[0, 0].set_xlabel('Latent Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Latent dimensions variance
        latent_vars = np.var(latents, axis=0)
        axes[0, 1].bar(range(len(latent_vars)), latent_vars)
        axes[0, 1].set_title('Latent Dimensions Variance')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2D projection (first two dimensions) colored by age
        if latents.shape[1] >= 2:
            if static_params.shape[1] > 0:
                scatter = axes[1, 0].scatter(latents[:, 0], latents[:, 1], 
                                           c=static_params[:, 0], 
                                           alpha=0.6, s=15, cmap='viridis')
                axes[1, 0].set_title('Latent Space 2D Projection\n(Colored by Age)')
                cbar = plt.colorbar(scatter, ax=axes[1, 0])
                cbar.set_label('Age')
            else:
                axes[1, 0].scatter(latents[:, 0], latents[:, 1], alpha=0.6, s=15, color='blue')
                axes[1, 0].set_title('Latent Space 2D Projection')
            axes[1, 0].set_xlabel('Latent Dim 0')
            axes[1, 0].set_ylabel('Latent Dim 1')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Condition space distribution
        if conditions.shape[1] >= 2:
            axes[1, 1].scatter(conditions[:, 0], conditions[:, 1], alpha=0.6, s=10)
            axes[1, 1].set_title('Condition Space 2D Projection')
            axes[1, 1].set_xlabel('Condition Dim 0')
            axes[1, 1].set_ylabel('Condition Dim 1')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'latent_space_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_latent_interpolation(self):
        """Plot latent space interpolation examples."""
        # This would require specific test conditions - placeholder for now
        pass
    
    def _plot_interpolation_examples(self, interpolation_results: List[Dict]):
        """Plot condition interpolation examples."""
        fig, axes = plt.subplots(len(interpolation_results), 1, figsize=(12, 3*len(interpolation_results)))
        
        if len(interpolation_results) == 1:
            axes = [axes]
        
        for i, result in enumerate(interpolation_results):
            interpolated = result['interpolated']
            
            # Plot interpolation sequence
            for step in range(interpolated.shape[0]):
                alpha = 0.3 + 0.7 * (step / (interpolated.shape[0] - 1))
                axes[i].plot(interpolated[step, :, 0], alpha=alpha, 
                           color=plt.cm.viridis(step / (interpolated.shape[0] - 1)))
            
            axes[i].set_title(f'Interpolation Example {i+1}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'interpolation_examples.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_metrics(self, metrics: Dict, filename: str):
        """Save metrics to JSON file."""
        import json
        
        with open(self.metrics_dir / filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {self.metrics_dir / filename}")
    
    def generate_evaluation_report(self, 
                                 test_loader: torch.utils.data.DataLoader,
                                 test_dataset: ABRMaskedDataset) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            test_loader: Test data loader
            test_dataset: Test dataset
            
        Returns:
            Complete evaluation results
        """
        logger.info("Generating comprehensive evaluation report...")
        
        # Run all evaluations
        reconstruction_metrics = self.evaluate_reconstruction(test_loader)
        generation_metrics = self.evaluate_generation(test_dataset)
        latent_analysis = self.analyze_latent_space(test_loader)
        interpolation_analysis = self.evaluate_condition_interpolation(test_dataset)
        
        # Combine results
        report = {
            'reconstruction': reconstruction_metrics,
            'generation': generation_metrics,
            'latent_space': latent_analysis,
            'interpolation': interpolation_analysis,
            'summary': {
                'reconstruction_quality': 'Good' if reconstruction_metrics['psnr'] > 20 else 'Poor',
                'generation_quality': 'Good' if generation_metrics['wasserstein_distance'] < 1.0 else 'Poor',
                'latent_utilization': f"{latent_analysis['active_dimensions']}/{latent_analysis['total_dimensions']} dimensions active"
            }
        }
        
        # Save complete report
        self._save_metrics(report, "evaluation_report.json")
        
        logger.info("Evaluation report completed and saved")
        return report
    
    def _create_generation_summary(self, generated_results: Dict, real_sample: Dict, 
                                 condition_idx: int, samples_per_condition: int, extracted_peaks: Dict) -> List[Dict]:
        """Create detailed summary for each generated sample with comprehensive static parameter comparison."""
        summaries = []
        
        for i in range(samples_per_condition):
            summary = {
                'condition_idx': condition_idx,
                'sample_idx': i,
                
                # Generated static parameters
                'age': float(generated_results['generated_static_params']['age'][i]),
                'intensity': float(generated_results['generated_static_params']['intensity'][i]),
                'stimulus_rate': float(generated_results['generated_static_params']['stimulus_rate'][i]),
                'hear_loss': int(generated_results['generated_static_params']['hear_loss'][i]),
                
                # Real static parameters for comparison
                'real_age': float(real_sample['static_params']['age']),
                'real_intensity': float(real_sample['static_params']['intensity']),
                'real_stimulus_rate': float(real_sample['static_params']['stimulus_rate']),
                'real_hear_loss': int(real_sample['static_params']['hear_loss']),
                
                # Static parameter errors
                'age_error': float(generated_results['generated_static_params']['age'][i]) - float(real_sample['static_params']['age']),
                'intensity_error': float(generated_results['generated_static_params']['intensity'][i]) - float(real_sample['static_params']['intensity']),
                'stimulus_rate_error': float(generated_results['generated_static_params']['stimulus_rate'][i]) - float(real_sample['static_params']['stimulus_rate']),
                'hear_loss_error': int(generated_results['generated_static_params']['hear_loss'][i]) - int(real_sample['static_params']['hear_loss']),
                
                # Generated clinical features (from model decoder)
                'gen_latency_I': float(generated_results.get('generated_latencies', torch.zeros(samples_per_condition, 3))[i, 0]),
                'gen_latency_III': float(generated_results.get('generated_latencies', torch.zeros(samples_per_condition, 3))[i, 1]),
                'gen_latency_V': float(generated_results.get('generated_latencies', torch.zeros(samples_per_condition, 3))[i, 2]),
                'gen_amplitude_I': float(generated_results.get('generated_amplitudes', torch.zeros(samples_per_condition, 3))[i, 0]),
                'gen_amplitude_III': float(generated_results.get('generated_amplitudes', torch.zeros(samples_per_condition, 3))[i, 1]),
                'gen_amplitude_V': float(generated_results.get('generated_amplitudes', torch.zeros(samples_per_condition, 3))[i, 2]),
                
                # Extracted peaks (from waveform analysis)
                'ext_latency_I': float(extracted_peaks['latencies'][i, 0]),
                'ext_latency_III': float(extracted_peaks['latencies'][i, 1]),
                'ext_latency_V': float(extracted_peaks['latencies'][i, 2]),
                'ext_amplitude_I': float(extracted_peaks['amplitudes'][i, 0]),
                'ext_amplitude_III': float(extracted_peaks['amplitudes'][i, 1]),
                'ext_amplitude_V': float(extracted_peaks['amplitudes'][i, 2]),
                
                # Peak detection confidence
                'confidence_I': float(extracted_peaks['confidence'][i, 0]),
                'confidence_III': float(extracted_peaks['confidence'][i, 1]),
                'confidence_V': float(extracted_peaks['confidence'][i, 2]),
                
                # Real clinical values for comparison
                'real_latency_I': float(real_sample['latency_data']['I Latancy']),
                'real_latency_III': float(real_sample['latency_data']['III Latancy']),
                'real_latency_V': float(real_sample['latency_data']['V Latancy']),
                'real_amplitude_I': float(real_sample['amplitude_data']['I Amplitude']),
                'real_amplitude_III': float(real_sample['amplitude_data']['III Amplitude']),
                'real_amplitude_V': float(real_sample['amplitude_data']['V Amplitude'])
            }
            summaries.append(summary)
        
        return summaries
    
    def _save_generation_table(self, generation_summary: List[Dict]):
        """Save detailed generation results as CSV and formatted table with comprehensive static parameter analysis."""
        df = pd.DataFrame(generation_summary)
        
        # Save as CSV
        csv_path = self.metrics_dir / "generation_detailed_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Create formatted table for first 30 samples (increased from 20)
        display_df = df.head(30).round(3)
        
        # Create a more readable summary table with enhanced static parameter information
        summary_columns = [
            'condition_idx', 'sample_idx', 
            'age', 'real_age', 'age_error',
            'intensity', 'real_intensity', 'intensity_error',
            'stimulus_rate', 'real_stimulus_rate', 'stimulus_rate_error',
            'hear_loss', 'real_hear_loss', 'hear_loss_error',
            'gen_latency_I', 'gen_latency_III', 'gen_latency_V',
            'ext_latency_I', 'ext_latency_III', 'ext_latency_V',
            'real_latency_I', 'real_latency_III', 'real_latency_V',
            'confidence_I', 'confidence_III', 'confidence_V'
        ]
        
        summary_table = display_df[summary_columns]
        
        # Save formatted table
        table_path = self.metrics_dir / "generation_summary_table.txt"
        with open(table_path, 'w') as f:
            f.write("=" * 180 + "\n")
            f.write("COMPREHENSIVE GENERATION EVALUATION RESULTS WITH STATIC PARAMETERS\n")
            f.write("=" * 180 + "\n\n")
            f.write("Legend:\n")
            f.write("- age, intensity, stimulus_rate, hear_loss: Generated static parameters\n")
            f.write("- real_age, real_intensity, real_stimulus_rate, real_hear_loss: Original conditioning static parameters\n")
            f.write("- age_error, intensity_error, stimulus_rate_error, hear_loss_error: Generated - Real parameter differences\n")
            f.write("- gen_latency_*: Latencies generated by model decoder\n")
            f.write("- ext_latency_*: Latencies extracted from generated waveform\n")
            f.write("- real_latency_*: Original latencies from conditioning data\n")
            f.write("- confidence_*: Peak detection confidence (0-1)\n\n")
            f.write(summary_table.to_string(index=False))
            f.write("\n\n")
            
            # Enhanced static parameter analysis with comparison
            f.write("STATIC PARAMETER ANALYSIS (Generated vs Real):\n")
            f.write("=" * 60 + "\n")
            
            # Age statistics
            f.write(f"Age Distribution:\n")
            f.write(f"  Generated - Mean: {df['age'].mean():.3f}, Std: {df['age'].std():.3f}, Range: [{df['age'].min():.3f}, {df['age'].max():.3f}]\n")
            f.write(f"  Real      - Mean: {df['real_age'].mean():.3f}, Std: {df['real_age'].std():.3f}, Range: [{df['real_age'].min():.3f}, {df['real_age'].max():.3f}]\n")
            f.write(f"  Error     - Mean: {df['age_error'].mean():.3f}, Std: {df['age_error'].std():.3f}, MAE: {np.abs(df['age_error']).mean():.3f}\n\n")
            
            # Intensity statistics  
            f.write(f"Intensity Distribution:\n")
            f.write(f"  Generated - Mean: {df['intensity'].mean():.3f}, Std: {df['intensity'].std():.3f}, Range: [{df['intensity'].min():.3f}, {df['intensity'].max():.3f}]\n")
            f.write(f"  Real      - Mean: {df['real_intensity'].mean():.3f}, Std: {df['real_intensity'].std():.3f}, Range: [{df['real_intensity'].min():.3f}, {df['real_intensity'].max():.3f}]\n")
            f.write(f"  Error     - Mean: {df['intensity_error'].mean():.3f}, Std: {df['intensity_error'].std():.3f}, MAE: {np.abs(df['intensity_error']).mean():.3f}\n\n")
            
            # Stimulus rate statistics
            f.write(f"Stimulus Rate Distribution:\n")
            f.write(f"  Generated - Mean: {df['stimulus_rate'].mean():.3f}, Std: {df['stimulus_rate'].std():.3f}, Range: [{df['stimulus_rate'].min():.3f}, {df['stimulus_rate'].max():.3f}]\n")
            f.write(f"  Real      - Mean: {df['real_stimulus_rate'].mean():.3f}, Std: {df['real_stimulus_rate'].std():.3f}, Range: [{df['real_stimulus_rate'].min():.3f}, {df['real_stimulus_rate'].max():.3f}]\n")
            f.write(f"  Error     - Mean: {df['stimulus_rate_error'].mean():.3f}, Std: {df['stimulus_rate_error'].std():.3f}, MAE: {np.abs(df['stimulus_rate_error']).mean():.3f}\n\n")
            
            # Hearing loss distribution
            f.write(f"Hearing Loss Distribution:\n")
            f.write("  Generated:\n")
            hearing_loss_counts = df['hear_loss'].value_counts().sort_index()
            for level, count in hearing_loss_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"    Level {level}: {count} samples ({percentage:.1f}%)\n")
            
            f.write("  Real:\n")
            real_hearing_loss_counts = df['real_hear_loss'].value_counts().sort_index()
            for level, count in real_hearing_loss_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"    Level {level}: {count} samples ({percentage:.1f}%)\n")
            
            # Hearing loss accuracy
            correct_hear_loss = (df['hear_loss'] == df['real_hear_loss']).sum()
            hear_loss_accuracy = correct_hear_loss / len(df) * 100
            f.write(f"  Accuracy: {correct_hear_loss}/{len(df)} ({hear_loss_accuracy:.1f}%)\n")
            
            f.write("\n")
            
            # Add generation statistics
            f.write("GENERATION STATISTICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total samples generated: {len(df)}\n")
            f.write(f"Unique conditions: {df['condition_idx'].nunique()}\n")
            f.write(f"Average peak detection confidence: {df[['confidence_I', 'confidence_III', 'confidence_V']].mean().mean():.3f}\n")
            
            # Latency comparison statistics
            latency_errors = {
                'I': np.abs(df['ext_latency_I'] - df['real_latency_I']).mean(),
                'III': np.abs(df['ext_latency_III'] - df['real_latency_III']).mean(),
                'V': np.abs(df['ext_latency_V'] - df['real_latency_V']).mean()
            }
            
            f.write(f"\nMean Absolute Latency Errors (extracted vs real):\n")
            for peak, error in latency_errors.items():
                f.write(f"  Peak {peak}: {error:.3f} samples\n")
            
            # Add detailed static parameter comparison table
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED STATIC PARAMETER COMPARISON (First 15 Samples)\n")
            f.write("=" * 80 + "\n")
            
            # Create a focused static parameter table with comparison
            static_param_df = display_df.head(15)[['condition_idx', 'sample_idx', 
                                                  'age', 'real_age', 'age_error',
                                                  'intensity', 'real_intensity', 'intensity_error',
                                                  'stimulus_rate', 'real_stimulus_rate', 'stimulus_rate_error',
                                                  'hear_loss', 'real_hear_loss', 'hear_loss_error']].copy()
            
            # Add some derived metrics for better understanding
            static_param_df['age_match'] = (np.abs(static_param_df['age_error']) < 0.1).astype(str)
            static_param_df['intensity_match'] = (np.abs(static_param_df['intensity_error']) < 5.0).astype(str)
            static_param_df['stimulus_rate_match'] = (np.abs(static_param_df['stimulus_rate_error']) < 2.0).astype(str)
            static_param_df['hear_loss_match'] = (static_param_df['hear_loss_error'] == 0).astype(str)
            
            f.write(static_param_df.to_string(index=False))
            f.write("\n\n")
            
            # Summary statistics for all static parameters
            f.write("COMPLETE DATASET STATIC PARAMETER SUMMARY:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Samples: {len(df)}\n\n")
            
            # Age analysis
            f.write("Age Analysis:\n")
            f.write(f"  Range: {df['age'].min():.4f} - {df['age'].max():.4f} years\n")
            f.write(f"  Mean ± Std: {df['age'].mean():.4f} ± {df['age'].std():.4f}\n")
            f.write(f"  Median: {df['age'].median():.4f}\n")
            f.write(f"  Unique values: {df['age'].nunique()}\n\n")
            
            # Intensity analysis
            f.write("Intensity Analysis:\n")
            f.write(f"  Range: {df['intensity'].min():.2f} - {df['intensity'].max():.2f} dB\n")
            f.write(f"  Mean ± Std: {df['intensity'].mean():.2f} ± {df['intensity'].std():.2f}\n")
            f.write(f"  Median: {df['intensity'].median():.2f}\n")
            f.write(f"  Unique values: {df['intensity'].nunique()}\n\n")
            
            # Stimulus rate analysis
            f.write("Stimulus Rate Analysis:\n")
            f.write(f"  Range: {df['stimulus_rate'].min():.2f} - {df['stimulus_rate'].max():.2f} Hz\n")
            f.write(f"  Mean ± Std: {df['stimulus_rate'].mean():.2f} ± {df['stimulus_rate'].std():.2f}\n")
            f.write(f"  Median: {df['stimulus_rate'].median():.2f}\n")
            f.write(f"  Unique values: {df['stimulus_rate'].nunique()}\n\n")
            
            # Hearing loss analysis
            f.write("Hearing Loss Analysis:\n")
            for level in sorted(df['hear_loss'].unique()):
                count = (df['hear_loss'] == level).sum()
                percentage = (count / len(df)) * 100
                f.write(f"  Level {level}: {count} samples ({percentage:.1f}%)\n")
        
        # Also create a separate CSV with comprehensive static parameters for easy analysis
        static_params_csv = self.metrics_dir / "static_parameters_analysis.csv"
        static_only_df = df[['condition_idx', 'sample_idx', 
                           'age', 'real_age', 'age_error',
                           'intensity', 'real_intensity', 'intensity_error',
                           'stimulus_rate', 'real_stimulus_rate', 'stimulus_rate_error',
                           'hear_loss', 'real_hear_loss', 'hear_loss_error']].copy()
        
        # Add accuracy flags
        static_only_df['age_accurate'] = np.abs(static_only_df['age_error']) < 0.1
        static_only_df['intensity_accurate'] = np.abs(static_only_df['intensity_error']) < 5.0
        static_only_df['stimulus_rate_accurate'] = np.abs(static_only_df['stimulus_rate_error']) < 2.0
        static_only_df['hear_loss_accurate'] = static_only_df['hear_loss_error'] == 0
        
        static_only_df.to_csv(static_params_csv, index=False)
        
        logger.info(f"Detailed generation results saved to {csv_path}")
        logger.info(f"Enhanced summary table saved to {table_path}")
        logger.info(f"Static parameters analysis saved to {static_params_csv}")
    
    def _compute_comprehensive_generation_metrics(self, generated_data: List[Dict], 
                                                real_data: List[Dict]) -> Dict[str, float]:
        """Compute comprehensive metrics including clinical parameter accuracy."""
        metrics = {}
        
        # Collect all waveforms
        all_generated_waveforms = np.concatenate([data['waveforms'] for data in generated_data], axis=0)
        all_real_waveforms = np.array([data['waveform'] for data in real_data])
        
        # Basic waveform metrics
        basic_metrics = self._compute_generation_metrics(all_generated_waveforms, all_real_waveforms)
        metrics.update(basic_metrics)
        
        # Clinical parameter metrics
        all_gen_latencies = np.concatenate([data['extracted_latencies'] for data in generated_data], axis=0)
        all_real_latencies = np.array([data['real_latencies'] for data in real_data])
        
        all_gen_amplitudes = np.concatenate([data['extracted_amplitudes'] for data in generated_data], axis=0)
        all_real_amplitudes = np.array([data['real_amplitudes'] for data in real_data])
        
        # Expand real data to match generated data (multiple samples per condition)
        samples_per_condition = len(all_gen_latencies) // len(all_real_latencies)
        expanded_real_latencies = np.repeat(all_real_latencies, samples_per_condition, axis=0)
        expanded_real_amplitudes = np.repeat(all_real_amplitudes, samples_per_condition, axis=0)
        
        # Latency accuracy (only for detected peaks)
        valid_latencies = (all_gen_latencies > 0) & (expanded_real_latencies > 0)
        if valid_latencies.any():
            latency_mae = np.mean(np.abs(all_gen_latencies[valid_latencies] - expanded_real_latencies[valid_latencies]))
            metrics['latency_mae'] = float(latency_mae)
            
            # Per-peak latency accuracy
            for i, peak_name in enumerate(['I', 'III', 'V']):
                valid_peak = valid_latencies[:, i]
                if valid_peak.any():
                    peak_mae = np.mean(np.abs(all_gen_latencies[valid_peak, i] - expanded_real_latencies[valid_peak, i]))
                    metrics[f'latency_mae_peak_{peak_name}'] = float(peak_mae)
        
        # Amplitude accuracy
        valid_amplitudes = (all_gen_amplitudes != 0) & (expanded_real_amplitudes != 0)
        if valid_amplitudes.any():
            amplitude_mae = np.mean(np.abs(all_gen_amplitudes[valid_amplitudes] - expanded_real_amplitudes[valid_amplitudes]))
            metrics['amplitude_mae'] = float(amplitude_mae)
        
        # Peak detection success rate
        detection_rates = []
        for i in range(3):  # I, III, V peaks
            detected = np.sum(all_gen_latencies[:, i] > 0)
            total = len(all_gen_latencies)
            detection_rates.append(detected / total)
        
        metrics['peak_detection_rate_I'] = detection_rates[0]
        metrics['peak_detection_rate_III'] = detection_rates[1]
        metrics['peak_detection_rate_V'] = detection_rates[2]
        metrics['overall_peak_detection_rate'] = np.mean(detection_rates)
        
        # Confidence statistics
        all_confidence = np.concatenate([data['peak_confidence'] for data in generated_data], axis=0)
        metrics['mean_peak_confidence'] = float(np.mean(all_confidence))
        metrics['min_peak_confidence'] = float(np.min(all_confidence))
        metrics['max_peak_confidence'] = float(np.max(all_confidence))
        
        return metrics
    
    def _plot_comprehensive_generation_examples(self, generated_data: List[Dict], real_data: List[Dict]):
        """Plot comprehensive generation examples with clinical parameters and static parameter information."""
        num_examples = min(4, len(generated_data))
        fig, axes = plt.subplots(num_examples, 3, figsize=(20, 5*num_examples))
        
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_examples):
            gen_data = generated_data[i]
            real_data_item = real_data[i]
            
            # Extract static parameters
            static_params = real_data_item.get('static_params', np.array([0, 0, 0, 0]))
            if len(static_params) >= 4:
                age = static_params[0]
                intensity = static_params[1]
                stimulus_rate = static_params[2]
                hear_loss = int(static_params[3])
                
                param_str = f'Age: {age:.1f}, Intensity: {intensity:.1f}dB, Rate: {stimulus_rate:.1f}Hz, HL: {hear_loss}'
            else:
                param_str = 'Parameters: N/A'
            
            # Plot first generated waveform vs real
            axes[i, 0].plot(gen_data['waveforms'][0], label='Generated', alpha=0.8, color='red', linewidth=1.5)
            axes[i, 0].plot(real_data_item['waveform'], label='Real', alpha=0.8, color='blue', linewidth=1.5)
            axes[i, 0].set_title(f'Condition {i+1}: Waveform Comparison\n{param_str}', fontsize=10)
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].set_xlabel('Time (samples)')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot latency comparison
            peaks = ['I', 'III', 'V']
            x_pos = np.arange(len(peaks))
            
            gen_latencies = gen_data['extracted_latencies'][0]
            real_latencies = real_data_item['real_latencies']
            
            width = 0.35
            axes[i, 1].bar(x_pos - width/2, gen_latencies, width, label='Generated', alpha=0.8, color='red')
            axes[i, 1].bar(x_pos + width/2, real_latencies, width, label='Real', alpha=0.8, color='blue')
            axes[i, 1].set_title(f'Condition {i+1}: Latency Comparison\n{param_str}', fontsize=10)
            axes[i, 1].set_ylabel('Latency (samples)')
            axes[i, 1].set_xlabel('Peak')
            axes[i, 1].set_xticks(x_pos)
            axes[i, 1].set_xticklabels(peaks)
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
            
            # Plot amplitude comparison
            gen_amplitudes = gen_data['extracted_amplitudes'][0]
            real_amplitudes = real_data_item['real_amplitudes']
            
            axes[i, 2].bar(x_pos - width/2, gen_amplitudes, width, label='Generated', alpha=0.8, color='red')
            axes[i, 2].bar(x_pos + width/2, real_amplitudes, width, label='Real', alpha=0.8, color='blue')
            axes[i, 2].set_title(f'Condition {i+1}: Amplitude Comparison\n{param_str}', fontsize=10)
            axes[i, 2].set_ylabel('Amplitude')
            axes[i, 2].set_xlabel('Peak')
            axes[i, 2].set_xticks(x_pos)
            axes[i, 2].set_xticklabels(peaks)
            axes[i, 2].legend()
            axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'comprehensive_generation_examples.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Comprehensive generation examples plot saved")
    
    def _create_static_parameter_visualization_table(self, test_loader: torch.utils.data.DataLoader, num_samples: int = 20):
        """Create a detailed table showing static parameters for reconstruction examples."""
        self.model.eval()
        
        static_params_list = []
        sample_indices = []
        
        with torch.no_grad():
            sample_count = 0
            for batch_idx, batch in enumerate(test_loader):
                if sample_count >= num_samples:
                    break
                    
                time_series = batch['time_series'].to(self.device)
                static_params = batch['static_params']
                
                batch_size = time_series.shape[0]
                for i in range(min(batch_size, num_samples - sample_count)):
                    # Extract static parameters from dictionary
                    params = [
                        static_params['age'][i].item(),
                        static_params['intensity'][i].item(),
                        static_params['stimulus_rate'][i].item(),
                        static_params['hear_loss'][i].item()
                    ]
                    static_params_list.append(params)
                    sample_indices.append(sample_count + i + 1)
                    
                sample_count += batch_size
        
        # Create DataFrame
        df = pd.DataFrame(static_params_list, columns=['age', 'intensity', 'stimulus_rate', 'hear_loss'])
        df.insert(0, 'sample_id', sample_indices)
        
        # Add derived information
        df['age_category'] = pd.cut(df['age'], 
                                  bins=[-float('inf'), 0.1, 0.3, 0.6, float('inf')], 
                                  labels=['Infant', 'Child', 'Adult', 'Senior'])
        df['intensity_level'] = pd.cut(df['intensity'], 
                                     bins=[-float('inf'), -1.0, 0.0, 1.0, float('inf')], 
                                     labels=['Low', 'Medium', 'High', 'Very High'])
        df['hear_loss_desc'] = df['hear_loss'].map({0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'})
        
        # Save detailed table
        table_path = self.metrics_dir / "reconstruction_static_parameters_table.txt"
        with open(table_path, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("STATIC PARAMETERS FOR RECONSTRUCTION EXAMPLES\n")
            f.write("=" * 120 + "\n\n")
            f.write("This table shows the static parameters for each sample used in reconstruction analysis.\n")
            f.write("These parameters condition the model's generation process.\n\n")
            
            # Main table
            f.write("DETAILED STATIC PARAMETERS:\n")
            f.write("-" * 80 + "\n")
            display_df = df[['sample_id', 'age', 'intensity', 'stimulus_rate', 'hear_loss', 
                           'age_category', 'intensity_level', 'hear_loss_desc']].round(3)
            f.write(display_df.to_string(index=False))
            f.write("\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Age range: {df['age'].min():.3f} - {df['age'].max():.3f}\n")
            f.write(f"Intensity range: {df['intensity'].min():.3f} - {df['intensity'].max():.3f} dB\n")
            f.write(f"Stimulus rate range: {df['stimulus_rate'].min():.3f} - {df['stimulus_rate'].max():.3f} Hz\n")
            
            f.write(f"\nHearing loss distribution:\n")
            for level, desc in {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}.items():
                count = (df['hear_loss'] == level).sum()
                percentage = (count / len(df)) * 100
                f.write(f"  {desc} (Level {level}): {count} samples ({percentage:.1f}%)\n")
        
        # Save CSV
        csv_path = self.metrics_dir / "reconstruction_static_parameters.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Static parameters table saved to {table_path}")
        logger.info(f"Static parameters CSV saved to {csv_path}")
        
        return df
    
    def _plot_clinical_parameter_comparison(self, generated_data: List[Dict], real_data: List[Dict]):
        """Plot clinical parameter comparison across all generated samples."""
        # Collect all data
        all_gen_latencies = np.concatenate([data['extracted_latencies'] for data in generated_data], axis=0)
        all_real_latencies = np.array([data['real_latencies'] for data in real_data])
        
        all_gen_amplitudes = np.concatenate([data['extracted_amplitudes'] for data in generated_data], axis=0)
        all_real_amplitudes = np.array([data['real_amplitudes'] for data in real_data])
        
        all_confidence = np.concatenate([data['peak_confidence'] for data in generated_data], axis=0)
        
        # Expand real data to match generated data
        samples_per_condition = len(all_gen_latencies) // len(all_real_latencies)
        expanded_real_latencies = np.repeat(all_real_latencies, samples_per_condition, axis=0)
        expanded_real_amplitudes = np.repeat(all_real_amplitudes, samples_per_condition, axis=0)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        peaks = ['I', 'III', 'V']
        
        # Latency scatter plots
        for i, peak in enumerate(peaks):
            valid_mask = (all_gen_latencies[:, i] > 0) & (expanded_real_latencies[:, i] > 0)
            if valid_mask.any():
                axes[0, i].scatter(expanded_real_latencies[valid_mask, i], all_gen_latencies[valid_mask, i], 
                                 alpha=0.6, c=all_confidence[valid_mask, i], cmap='viridis')
                
                # Perfect correlation line
                min_val = min(expanded_real_latencies[valid_mask, i].min(), all_gen_latencies[valid_mask, i].min())
                max_val = max(expanded_real_latencies[valid_mask, i].max(), all_gen_latencies[valid_mask, i].max())
                axes[0, i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect correlation')
                
                axes[0, i].set_xlabel(f'Real Peak {peak} Latency')
                axes[0, i].set_ylabel(f'Generated Peak {peak} Latency')
                axes[0, i].set_title(f'Peak {peak} Latency Correlation')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr = np.corrcoef(expanded_real_latencies[valid_mask, i], all_gen_latencies[valid_mask, i])[0, 1]
                axes[0, i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0, i].transAxes, 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Amplitude scatter plots
        for i, peak in enumerate(peaks):
            valid_mask = (all_gen_amplitudes[:, i] != 0) & (expanded_real_amplitudes[:, i] != 0)
            if valid_mask.any():
                axes[1, i].scatter(expanded_real_amplitudes[valid_mask, i], all_gen_amplitudes[valid_mask, i], 
                                 alpha=0.6, c=all_confidence[valid_mask, i], cmap='viridis')
                
                # Perfect correlation line
                min_val = min(expanded_real_amplitudes[valid_mask, i].min(), all_gen_amplitudes[valid_mask, i].min())
                max_val = max(expanded_real_amplitudes[valid_mask, i].max(), all_gen_amplitudes[valid_mask, i].max())
                axes[1, i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect correlation')
                
                axes[1, i].set_xlabel(f'Real Peak {peak} Amplitude')
                axes[1, i].set_ylabel(f'Generated Peak {peak} Amplitude')
                axes[1, i].set_title(f'Peak {peak} Amplitude Correlation')
                axes[1, i].legend()
                axes[1, i].grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr = np.corrcoef(expanded_real_amplitudes[valid_mask, i], all_gen_amplitudes[valid_mask, i])[0, 1]
                axes[1, i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1, i].transAxes, 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'clinical_parameter_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Peak detection rates
        detection_rates = []
        for i in range(3):
            detected = np.sum(all_gen_latencies[:, i] > 0)
            total = len(all_gen_latencies)
            detection_rates.append(detected / total)
        
        axes[0].bar(peaks, detection_rates, alpha=0.8, color=['red', 'green', 'blue'])
        axes[0].set_title('Peak Detection Success Rate')
        axes[0].set_ylabel('Detection Rate')
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, rate in enumerate(detection_rates):
            axes[0].text(i, rate + 0.02, f'{rate:.1%}', ha='center', va='bottom')
        
        # Confidence distribution
        axes[1].hist(all_confidence.flatten(), bins=30, alpha=0.8, color='purple', edgecolor='black')
        axes[1].set_title('Peak Detection Confidence Distribution')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        # Latency error distribution
        valid_latencies = (all_gen_latencies > 0) & (expanded_real_latencies > 0)
        if valid_latencies.any():
            latency_errors = np.abs(all_gen_latencies[valid_latencies] - expanded_real_latencies[valid_latencies])
            axes[2].hist(latency_errors, bins=30, alpha=0.8, color='orange', edgecolor='black')
            axes[2].set_title('Latency Prediction Error Distribution')
            axes[2].set_xlabel('Absolute Error (samples)')
            axes[2].set_ylabel('Frequency')
            axes[2].grid(True, alpha=0.3)
            
            # Add mean error line
            mean_error = np.mean(latency_errors)
            axes[2].axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean Error: {mean_error:.2f}')
            axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'clinical_parameter_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Clinical parameter comparison plots saved")

    def _compute_pure_generation_metrics(self, generated_data: List[Dict], real_data: List[Dict]) -> Dict[str, float]:
        """Compute metrics for pure generation evaluation."""
        metrics = {}
        
        # Collect generated static parameters
        gen_ages = [data['generated_static_params']['age'] for data in generated_data]
        gen_intensities = [data['generated_static_params']['intensity'] for data in generated_data]
        gen_stimulus_rates = [data['generated_static_params']['stimulus_rate'] for data in generated_data]
        gen_hear_loss = [data['generated_static_params']['hear_loss'] for data in generated_data]
        
        # Collect real static parameters for comparison
        real_ages = [data['static_params'][0] for data in real_data]
        real_intensities = [data['static_params'][1] for data in real_data]
        real_stimulus_rates = [data['static_params'][2] for data in real_data]
        real_hear_loss = [data['static_params'][3] for data in real_data]
        
        # Static parameter distribution metrics
        metrics['generated_age_mean'] = float(np.mean(gen_ages))
        metrics['generated_age_std'] = float(np.std(gen_ages))
        metrics['real_age_mean'] = float(np.mean(real_ages))
        metrics['real_age_std'] = float(np.std(real_ages))
        
        metrics['generated_intensity_mean'] = float(np.mean(gen_intensities))
        metrics['generated_intensity_std'] = float(np.std(gen_intensities))
        metrics['real_intensity_mean'] = float(np.mean(real_intensities))
        metrics['real_intensity_std'] = float(np.std(real_intensities))
        
        metrics['generated_stimulus_rate_mean'] = float(np.mean(gen_stimulus_rates))
        metrics['generated_stimulus_rate_std'] = float(np.std(gen_stimulus_rates))
        metrics['real_stimulus_rate_mean'] = float(np.mean(real_stimulus_rates))
        metrics['real_stimulus_rate_std'] = float(np.std(real_stimulus_rates))
        
        # Hearing loss distribution
        gen_hear_loss_counts = np.bincount(np.array(gen_hear_loss, dtype=int), minlength=5)
        real_hear_loss_counts = np.bincount(np.array(real_hear_loss, dtype=int), minlength=5)
        
        for i in range(5):
            metrics[f'generated_hear_loss_{i}_ratio'] = float(gen_hear_loss_counts[i] / len(gen_hear_loss))
            metrics[f'real_hear_loss_{i}_ratio'] = float(real_hear_loss_counts[i] / len(real_hear_loss))
        
        # Waveform quality metrics
        gen_waveforms = np.array([data['waveform'] for data in generated_data])
        real_waveforms = np.array([data['waveform'] for data in real_data])
        
        # Signal statistics
        metrics['generated_signal_mean'] = float(np.mean(gen_waveforms))
        metrics['generated_signal_std'] = float(np.std(gen_waveforms))
        metrics['real_signal_mean'] = float(np.mean(real_waveforms))
        metrics['real_signal_std'] = float(np.std(real_waveforms))
        
        # Peak detection metrics
        all_gen_latencies = np.array([data['extracted_latencies'] for data in generated_data])
        all_gen_amplitudes = np.array([data['extracted_amplitudes'] for data in generated_data])
        all_confidence = np.array([data['peak_confidence'] for data in generated_data])
        
        # Peak detection success rates
        detection_rates = []
        for i in range(3):  # I, III, V peaks
            detected = np.sum(all_gen_latencies[:, i] > 0)
            total = len(all_gen_latencies)
            detection_rates.append(detected / total)
        
        metrics['peak_detection_rate_I'] = detection_rates[0]
        metrics['peak_detection_rate_III'] = detection_rates[1]
        metrics['peak_detection_rate_V'] = detection_rates[2]
        metrics['overall_peak_detection_rate'] = np.mean(detection_rates)
        
        # Confidence statistics
        metrics['mean_peak_confidence'] = float(np.mean(all_confidence))
        metrics['std_peak_confidence'] = float(np.std(all_confidence))
        metrics['min_peak_confidence'] = float(np.min(all_confidence))
        metrics['max_peak_confidence'] = float(np.max(all_confidence))
        
        # Clinical parameter ranges (check if generated values are realistic)
        metrics['age_range_coverage'] = float(np.ptp(gen_ages))  # Peak-to-peak range
        metrics['intensity_range_coverage'] = float(np.ptp(gen_intensities))
        metrics['stimulus_rate_range_coverage'] = float(np.ptp(gen_stimulus_rates))
        
        # Latency and amplitude statistics for detected peaks
        valid_latencies = all_gen_latencies[all_gen_latencies > 0]
        valid_amplitudes = all_gen_amplitudes[all_gen_amplitudes != 0]
        
        if len(valid_latencies) > 0:
            metrics['generated_latency_mean'] = float(np.mean(valid_latencies))
            metrics['generated_latency_std'] = float(np.std(valid_latencies))
            metrics['generated_latency_min'] = float(np.min(valid_latencies))
            metrics['generated_latency_max'] = float(np.max(valid_latencies))
        
        if len(valid_amplitudes) > 0:
            metrics['generated_amplitude_mean'] = float(np.mean(valid_amplitudes))
            metrics['generated_amplitude_std'] = float(np.std(valid_amplitudes))
            metrics['generated_amplitude_min'] = float(np.min(valid_amplitudes))
            metrics['generated_amplitude_max'] = float(np.max(valid_amplitudes))
        
        return metrics
    
    def _save_pure_generation_table(self, generated_data: List[Dict]):
        """Save detailed pure generation results to CSV."""
        results = []
        
        for i, data in enumerate(generated_data):
            result = {
                'sample_id': i,
                'generated_age': data['generated_static_params']['age'],
                'generated_intensity': data['generated_static_params']['intensity'],
                'generated_stimulus_rate': data['generated_static_params']['stimulus_rate'],
                'generated_hear_loss': data['generated_static_params']['hear_loss'],
                'extracted_latency_I': data['extracted_latencies'][0],
                'extracted_latency_III': data['extracted_latencies'][1],
                'extracted_latency_V': data['extracted_latencies'][2],
                'extracted_amplitude_I': data['extracted_amplitudes'][0],
                'extracted_amplitude_III': data['extracted_amplitudes'][1],
                'extracted_amplitude_V': data['extracted_amplitudes'][2],
                'confidence_I': data['peak_confidence'][0],
                'confidence_III': data['peak_confidence'][1],
                'confidence_V': data['peak_confidence'][2],
                'mean_confidence': np.mean(data['peak_confidence'])
            }
            
            # Add generated clinical features if available
            if 'generated_latencies' in data:
                result.update({
                    'neural_latency_I': data['generated_latencies'][0],
                    'neural_latency_III': data['generated_latencies'][1],
                    'neural_latency_V': data['generated_latencies'][2],
                    'neural_amplitude_I': data['generated_amplitudes'][0],
                    'neural_amplitude_III': data['generated_amplitudes'][1],
                    'neural_amplitude_V': data['generated_amplitudes'][2]
                })
            
            results.append(result)
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / 'pure_generation_detailed_results.csv', index=False)
        
        logger.info(f"Pure generation detailed results saved to {self.output_dir / 'pure_generation_detailed_results.csv'}")
    
    def _plot_pure_generation_examples(self, generated_data: List[Dict], real_data: List[Dict]):
        """Plot pure generation examples."""
        num_examples = min(6, len(generated_data))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(num_examples):
            gen_data = generated_data[i]
            
            # Plot generated waveform
            axes[i].plot(gen_data['waveform'], color='red', alpha=0.8, linewidth=1.5)
            
            # Mark detected peaks
            latencies = gen_data['extracted_latencies']
            amplitudes = gen_data['extracted_amplitudes']
            confidence = gen_data['peak_confidence']
            
            peak_names = ['I', 'III', 'V']
            colors = ['blue', 'green', 'orange']
            
            for j, (lat, amp, conf, name, color) in enumerate(zip(latencies, amplitudes, confidence, peak_names, colors)):
                if lat > 0:  # Peak detected
                    axes[i].scatter(lat, gen_data['waveform'][int(lat)], 
                                  color=color, s=100, alpha=0.8, 
                                  label=f'Peak {name} (conf: {conf:.2f})')
            
            # Add title with generated static parameters
            static_params = gen_data['generated_static_params']
            title = (f'Generated Sample {i+1}\n'
                    f'Age: {static_params["age"]:.1f}, Intensity: {static_params["intensity"]:.1f} dB\n'
                    f'Rate: {static_params["stimulus_rate"]:.1f} Hz, HL: {static_params["hear_loss"]:.0f}')
            axes[i].set_title(title, fontsize=10)
            axes[i].set_xlabel('Time (samples)')
            axes[i].set_ylabel('Amplitude (μV)')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'pure_generation_examples.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Pure generation examples plot saved")
    
    def _plot_generated_static_parameters(self, generated_data: List[Dict], real_data: List[Dict]):
        """Plot comparison of generated vs real static parameters."""
        # Collect data
        gen_ages = [data['generated_static_params']['age'] for data in generated_data]
        gen_intensities = [data['generated_static_params']['intensity'] for data in generated_data]
        gen_stimulus_rates = [data['generated_static_params']['stimulus_rate'] for data in generated_data]
        gen_hear_loss = [data['generated_static_params']['hear_loss'] for data in generated_data]
        
        real_ages = [data['static_params'][0] for data in real_data]
        real_intensities = [data['static_params'][1] for data in real_data]
        real_stimulus_rates = [data['static_params'][2] for data in real_data]
        real_hear_loss = [data['static_params'][3] for data in real_data]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Age distribution
        axes[0, 0].hist(real_ages, bins=30, alpha=0.7, label='Real', color='blue', density=True)
        axes[0, 0].hist(gen_ages, bins=30, alpha=0.7, label='Generated', color='red', density=True)
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age (years)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Intensity distribution
        axes[0, 1].hist(real_intensities, bins=30, alpha=0.7, label='Real', color='blue', density=True)
        axes[0, 1].hist(gen_intensities, bins=30, alpha=0.7, label='Generated', color='red', density=True)
        axes[0, 1].set_title('Intensity Distribution')
        axes[0, 1].set_xlabel('Intensity (dB)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Stimulus rate distribution
        axes[1, 0].hist(real_stimulus_rates, bins=30, alpha=0.7, label='Real', color='blue', density=True)
        axes[1, 0].hist(gen_stimulus_rates, bins=30, alpha=0.7, label='Generated', color='red', density=True)
        axes[1, 0].set_title('Stimulus Rate Distribution')
        axes[1, 0].set_xlabel('Stimulus Rate (Hz)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Hearing loss distribution
        hear_loss_labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Profound']
        real_hl_counts = np.bincount(np.array(real_hear_loss, dtype=int), minlength=5)
        gen_hl_counts = np.bincount(np.array(gen_hear_loss, dtype=int), minlength=5)
        
        x_pos = np.arange(len(hear_loss_labels))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, real_hl_counts/len(real_hear_loss), width, 
                      label='Real', alpha=0.7, color='blue')
        axes[1, 1].bar(x_pos + width/2, gen_hl_counts/len(gen_hear_loss), width, 
                      label='Generated', alpha=0.7, color='red')
        axes[1, 1].set_title('Hearing Loss Distribution')
        axes[1, 1].set_xlabel('Hearing Loss Category')
        axes[1, 1].set_ylabel('Proportion')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(hear_loss_labels, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'generated_static_parameters.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create correlation matrix plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Generated parameters correlation
        gen_params = np.column_stack([gen_ages, gen_intensities, gen_stimulus_rates, gen_hear_loss])
        gen_corr = np.corrcoef(gen_params.T)
        
        im1 = axes[0].imshow(gen_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0].set_title('Generated Parameters Correlation')
        axes[0].set_xticks(range(4))
        axes[0].set_yticks(range(4))
        axes[0].set_xticklabels(['Age', 'Intensity', 'Stim Rate', 'Hear Loss'])
        axes[0].set_yticklabels(['Age', 'Intensity', 'Stim Rate', 'Hear Loss'])
        
        # Add correlation values
        for i in range(4):
            for j in range(4):
                axes[0].text(j, i, f'{gen_corr[i, j]:.2f}', ha='center', va='center')
        
        # Real parameters correlation
        real_params = np.column_stack([real_ages, real_intensities, real_stimulus_rates, real_hear_loss])
        real_corr = np.corrcoef(real_params.T)
        
        im2 = axes[1].imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1].set_title('Real Parameters Correlation')
        axes[1].set_xticks(range(4))
        axes[1].set_yticks(range(4))
        axes[1].set_xticklabels(['Age', 'Intensity', 'Stim Rate', 'Hear Loss'])
        axes[1].set_yticklabels(['Age', 'Intensity', 'Stim Rate', 'Hear Loss'])
        
        # Add correlation values
        for i in range(4):
            for j in range(4):
                axes[1].text(j, i, f'{real_corr[i, j]:.2f}', ha='center', va='center')
        
        # Add colorbar
        fig.colorbar(im1, ax=axes, shrink=0.6, label='Correlation Coefficient')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'static_parameters_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated static parameters plots saved") 