import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
from typing import Dict, Tuple, Optional, List
import math
import logging

logger = logging.getLogger(__name__)

class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer for dynamic conditioning."""
    
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, ..., feature_dim] - features to modulate
            c: [batch_size, cond_dim] - conditioning information
        Returns:
            modulated features
        """
        gamma = self.gamma(c)
        beta = self.beta(c)
        
        # Handle different input shapes
        if x.dim() == 3:  # [batch_size, seq_len, feature_dim]
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        elif x.dim() == 4:  # [batch_size, channels, height, width]
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            
        return gamma * x + beta

class ConditionalPriorNetwork(nn.Module):
    """Network to learn conditional prior p(z|c) instead of standard normal."""
    
    def __init__(self, condition_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.prior_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate conditional prior parameters.
        
        Args:
            condition: [batch_size, condition_dim]
        Returns:
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]
        """
        h = self.prior_net(condition)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        # Clamp for stability
        logvar = torch.clamp(logvar, min=-10, max=2)
        
        return mu, logvar

class PeakDetectionHead(nn.Module):
    """Multi-task head for explicit peak detection and characterization."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_peaks: int = 3):
        super().__init__()
        
        self.num_peaks = num_peaks  # I, III, V peaks
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Peak presence prediction (binary classification for each peak)
        self.presence_head = nn.Linear(hidden_dim, num_peaks)
        
        # Peak latency prediction (continuous values)
        self.latency_head = nn.Linear(hidden_dim, num_peaks)
        
        # Peak amplitude prediction (continuous values)
        self.amplitude_head = nn.Linear(hidden_dim, num_peaks)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, input_dim] - input features
        Returns:
            Dictionary with peak predictions
        """
        shared_features = self.shared_net(x)
        
        # Peak presence (logits for BCE loss)
        presence_logits = self.presence_head(shared_features)
        
        # Peak characteristics (continuous values)
        latencies = self.latency_head(shared_features)
        amplitudes = self.amplitude_head(shared_features)
        
        return {
            'presence_logits': presence_logits,  # [batch_size, num_peaks]
            'latencies': latencies,              # [batch_size, num_peaks]
            'amplitudes': amplitudes,            # [batch_size, num_peaks]
            'presence_probs': torch.sigmoid(presence_logits)
        }

class TemporalDecoder(nn.Module):
    """Temporal decoder using Transformer architecture for structured generation."""
    
    def __init__(self,
                 latent_dim: int,
                 condition_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 sequence_length: int = 200,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # Project latent + condition to initial hidden state
        self.input_projection = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(sequence_length, hidden_dim))
        
        # FiLM layers for dynamic conditioning
        self.film_layers = nn.ModuleList([
            FiLM(condition_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Create causal mask for autoregressive generation
        self.register_buffer('causal_mask', self._generate_causal_mask(sequence_length))
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, latent_dim]
            condition: [batch_size, condition_dim]
        Returns:
            generated_sequence: [batch_size, sequence_length]
        """
        batch_size = z.size(0)
        
        # Combine latent and condition
        combined = torch.cat([z, condition], dim=-1)
        
        # Project to hidden dimension
        hidden = self.input_projection(combined)  # [batch_size, hidden_dim]
        
        # Expand to sequence length and add positional embeddings
        hidden = hidden.unsqueeze(1).expand(-1, self.sequence_length, -1)
        hidden = hidden + self.pos_embedding.unsqueeze(0)
        
        # Create memory (learnable context)
        memory = hidden.clone()
        
        # Apply FiLM conditioning at each layer
        for i, film_layer in enumerate(self.film_layers):
            if i == 0:
                conditioned_hidden = film_layer(hidden, condition)
            else:
                conditioned_hidden = film_layer(conditioned_hidden, condition)
        
        # Transformer decoding with causal mask
        decoded = self.transformer(
            conditioned_hidden,
            memory,
            tgt_mask=self.causal_mask[:self.sequence_length, :self.sequence_length]
        )
        
        # Project to output
        output = self.output_projection(decoded)  # [batch_size, sequence_length, 1]
        output = output.squeeze(-1)  # [batch_size, sequence_length]
        
        return output

class HierarchicalEncoder(nn.Module):
    """Hierarchical encoder with two-stage latent variables."""
    
    def __init__(self,
                 input_dim: int = 1,
                 condition_dim: int = 128,
                 hidden_dim: int = 256,
                 waveform_latent_dim: int = 64,
                 peak_latent_dim: int = 32,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 sequence_length: int = 200):
        super().__init__()
        
        self.waveform_latent_dim = waveform_latent_dim
        self.peak_latent_dim = peak_latent_dim
        
        # Convolutional feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=7, padding=3),
            nn.GroupNorm(4, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, padding=2, stride=2),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.conv_seq_len = sequence_length // 4
        
        # FiLM conditioning for encoder
        self.film_layers = nn.ModuleList([
            FiLM(condition_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Stage 1: Waveform latent variables
        self.waveform_mu = nn.Linear(hidden_dim, waveform_latent_dim)
        self.waveform_logvar = nn.Linear(hidden_dim, waveform_latent_dim)
        
        # Stage 2: Peak latent variables (conditioned on waveform latent)
        self.peak_encoder = nn.Sequential(
            nn.Linear(hidden_dim + waveform_latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.peak_mu = nn.Linear(hidden_dim // 2, peak_latent_dim)
        self.peak_logvar = nn.Linear(hidden_dim // 2, peak_latent_dim)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, sequence_length]
            condition: [batch_size, condition_dim]
        Returns:
            Dictionary with hierarchical latent variables
        """
        batch_size = x.size(0)
        
        # Add channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Convolutional feature extraction
        x = self.conv_layers(x)  # [batch_size, hidden_dim, conv_seq_len]
        x = x.transpose(1, 2)    # [batch_size, conv_seq_len, hidden_dim]
        
        # Apply FiLM conditioning
        for film_layer in self.film_layers:
            x = film_layer(x, condition)
        
        # Transformer encoding
        x = self.transformer(x)  # [batch_size, conv_seq_len, hidden_dim]
        
        # Global pooling
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, conv_seq_len]
        pooled = self.global_pool(x).squeeze(-1)  # [batch_size, hidden_dim]
        
        # Stage 1: Waveform latent variables
        waveform_mu = self.waveform_mu(pooled)
        waveform_logvar = self.waveform_logvar(pooled)
        waveform_z = self._reparameterize(waveform_mu, waveform_logvar)
        
        # Stage 2: Peak latent variables (conditioned on waveform latent)
        peak_input = torch.cat([pooled, waveform_z], dim=-1)
        peak_features = self.peak_encoder(peak_input)
        
        peak_mu = self.peak_mu(peak_features)
        peak_logvar = self.peak_logvar(peak_features)
        peak_z = self._reparameterize(peak_mu, peak_logvar)
        
        return {
            'waveform_mu': waveform_mu,
            'waveform_logvar': waveform_logvar,
            'waveform_z': waveform_z,
            'peak_mu': peak_mu,
            'peak_logvar': peak_logvar,
            'peak_z': peak_z,
            'combined_z': torch.cat([waveform_z, peak_z], dim=-1)
        }
    
    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class ImprovedStaticParameterDecoder(nn.Module):
    """Improved static parameter decoder with better scaling and normalization."""
    
    def __init__(self, latent_dim: int = 48, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Age decoder (20-100 years)
        self.age_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Intensity decoder (20-100 dB)
        self.intensity_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Stimulus rate decoder (10-100 Hz)
        self.stimulus_rate_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Hearing loss decoder (categorical 0-4)
        self.hearing_loss_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 5),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate static parameters with proper scaling."""
        # Shared encoding
        shared = self.shared_encoder(z)
        
        # Generate parameters with realistic ranges
        age_raw = self.age_decoder(shared).squeeze(-1)
        intensity_raw = self.intensity_decoder(shared).squeeze(-1)
        stimulus_rate_raw = self.stimulus_rate_decoder(shared).squeeze(-1)
        hearing_loss_probs = self.hearing_loss_decoder(shared)
        
        # Scale to realistic ranges
        age = age_raw * 80.0 + 20.0          # 20-100 years
        intensity = intensity_raw * 80.0 + 20.0   # 20-100 dB
        stimulus_rate = stimulus_rate_raw * 90.0 + 10.0  # 10-100 Hz
        
        # Convert to categorical
        hearing_loss_categorical = torch.argmax(hearing_loss_probs, dim=-1).float()
        
        return {
            'age': age,
            'intensity': intensity,
            'stimulus_rate': stimulus_rate,
            'hear_loss': hearing_loss_categorical,
            'hearing_loss_probs': hearing_loss_probs
        }


class AdvancedConditionalVAE(nn.Module):
    """Advanced Conditional VAE with hierarchical latents, conditional priors, and multi-task learning."""
    
    def __init__(self,
                 static_dim: int = 4,
                 masked_features_dim: int = 64,
                 waveform_latent_dim: int = 64,
                 peak_latent_dim: int = 32,
                 condition_dim: int = 128,
                 hidden_dim: int = 256,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 sequence_length: int = 200,
                 beta_waveform: float = 1.0,
                 beta_peak: float = 0.5,
                 use_conditional_prior: bool = True,
                 noise_augmentation: float = 0.1):
        super().__init__()
        
        self.waveform_latent_dim = waveform_latent_dim
        self.peak_latent_dim = peak_latent_dim
        self.total_latent_dim = waveform_latent_dim + peak_latent_dim
        self.condition_dim = condition_dim
        self.beta_waveform = beta_waveform
        self.beta_peak = beta_peak
        self.use_conditional_prior = use_conditional_prior
        self.noise_augmentation = noise_augmentation
        
        # Import components from original model
        from .cvae_model import MaskedFeatureEncoder, ConditionEncoder
        
        # Masked feature encoder
        self.masked_feature_encoder = MaskedFeatureEncoder(
            num_features=6,
            hidden_dim=hidden_dim // 2,
            output_dim=masked_features_dim,
            dropout=dropout
        )
        
        # Condition encoder
        self.condition_encoder = ConditionEncoder(
            static_dim=static_dim,
            masked_features_dim=masked_features_dim,
            hidden_dim=hidden_dim,
            output_dim=condition_dim,
            dropout=dropout
        )
        
        # Hierarchical encoder
        self.hierarchical_encoder = HierarchicalEncoder(
            input_dim=1,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            waveform_latent_dim=waveform_latent_dim,
            peak_latent_dim=peak_latent_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            sequence_length=sequence_length
        )
        
        # Conditional prior networks
        if use_conditional_prior:
            self.waveform_prior = ConditionalPriorNetwork(
                condition_dim, waveform_latent_dim, hidden_dim // 2
            )
            self.peak_prior = ConditionalPriorNetwork(
                condition_dim + waveform_latent_dim, peak_latent_dim, hidden_dim // 2
            )
        
        # Temporal decoder
        self.temporal_decoder = TemporalDecoder(
            latent_dim=self.total_latent_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            sequence_length=sequence_length,
            dropout=dropout
        )
        
        # Peak detection head
        self.peak_head = PeakDetectionHead(
            input_dim=self.total_latent_dim + condition_dim,
            hidden_dim=hidden_dim // 2,
            num_peaks=3
        )
        
        # Improved static parameter decoder
        self.static_param_decoder = ImprovedStaticParameterDecoder(
            latent_dim=self.total_latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1)
    
    def _apply_noise_augmentation(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply noise augmentation during training."""
        if training and self.noise_augmentation > 0:
            noise = torch.randn_like(x) * self.noise_augmentation
            return x + noise
        return x
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the advanced CVAE."""
        
        # Extract data
        time_series = batch['time_series']
        static_params = batch['static_params']
        latency_data = batch['latency_data']
        latency_masks = batch['latency_masks']
        amplitude_data = batch['amplitude_data']
        amplitude_masks = batch['amplitude_masks']
        
        # Apply noise augmentation
        time_series = self._apply_noise_augmentation(time_series, self.training)
        
        # Prepare conditioning
        static_tensor = torch.stack([
            static_params['age'],
            static_params['intensity'], 
            static_params['stimulus_rate'],
            static_params['hear_loss'].float()
        ], dim=-1)
        
        features = [
            latency_data['I Latancy'],
            latency_data['III Latancy'],
            latency_data['V Latancy'],
            amplitude_data['I Amplitude'],
            amplitude_data['III Amplitude'],
            amplitude_data['V Amplitude']
        ]
        
        masks = [
            latency_masks['I Latancy'],
            latency_masks['III Latancy'],
            latency_masks['V Latancy'],
            amplitude_masks['I Amplitude'],
            amplitude_masks['III Amplitude'],
            amplitude_masks['V Amplitude']
        ]
        
        # Encode conditioning information
        masked_features_encoded = self.masked_feature_encoder(features, masks)
        condition = self.condition_encoder(static_tensor, masked_features_encoded)
        
        # Hierarchical encoding
        encoding_results = self.hierarchical_encoder(time_series, condition)
        
        # Temporal decoding
        abr_reconstruction = self.temporal_decoder(encoding_results['combined_z'], condition)
        
        # Peak prediction
        peak_input = torch.cat([encoding_results['combined_z'], condition], dim=-1)
        peak_predictions = self.peak_head(peak_input)
        
        # Static parameter generation
        generated_static_params = self.static_param_decoder(encoding_results['combined_z'])
        
        # Conditional priors (if enabled)
        prior_results = {}
        if self.use_conditional_prior:
            waveform_prior_mu, waveform_prior_logvar = self.waveform_prior(condition)
            peak_prior_input = torch.cat([condition, encoding_results['waveform_z']], dim=-1)
            peak_prior_mu, peak_prior_logvar = self.peak_prior(peak_prior_input)
            
            prior_results = {
                'waveform_prior_mu': waveform_prior_mu,
                'waveform_prior_logvar': waveform_prior_logvar,
                'peak_prior_mu': peak_prior_mu,
                'peak_prior_logvar': peak_prior_logvar
            }
        
        return {
            'abr_reconstruction': abr_reconstruction,
            'peak_predictions': peak_predictions,
            'generated_static_params': generated_static_params,
            'condition': condition,
            'masked_features_encoded': masked_features_encoded,
            **encoding_results,
            **prior_results
        }
    
    def generate(self, batch: Dict[str, torch.Tensor] = None, num_samples: int = 1, 
                 use_conditioning: bool = False, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """Generate new ABR samples with improved sampling strategy."""
        
        self.eval()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            if use_conditioning and batch is not None:
                # Conditional generation
                static_params = batch['static_params']
                latency_data = batch['latency_data']
                latency_masks = batch['latency_masks']
                amplitude_data = batch['amplitude_data']
                amplitude_masks = batch['amplitude_masks']
                
                static_tensor = torch.stack([
                    static_params['age'],
                    static_params['intensity'], 
                    static_params['stimulus_rate'],
                    static_params['hear_loss'].float()
                ], dim=-1)
                
                features = [
                    latency_data['I Latancy'],
                    latency_data['III Latancy'],
                    latency_data['V Latancy'],
                    amplitude_data['I Amplitude'],
                    amplitude_data['III Amplitude'],
                    amplitude_data['V Amplitude']
                ]
                
                masks = [
                    latency_masks['I Latancy'],
                    latency_masks['III Latancy'],
                    latency_masks['V Latancy'],
                    amplitude_masks['I Amplitude'],
                    amplitude_masks['III Amplitude'],
                    amplitude_masks['V Amplitude']
                ]
                
                masked_features_encoded = self.masked_feature_encoder(features, masks)
                condition = self.condition_encoder(static_tensor, masked_features_encoded)
                
                batch_size = condition.size(0)
                condition_repeated = condition.repeat_interleave(num_samples, dim=0)
                total_samples = batch_size * num_samples
                
            else:
                # Unconditional generation - create dummy conditioning
                total_samples = num_samples
                
                # Generate random static parameters
                dummy_static = torch.stack([
                    torch.rand(total_samples, device=device) * 80 + 20,  # age: 20-100
                    torch.rand(total_samples, device=device) * 80 + 20,  # intensity: 20-100
                    torch.rand(total_samples, device=device) * 90 + 10,  # stimulus_rate: 10-100
                    torch.randint(0, 5, (total_samples,), device=device).float()  # hear_loss: 0-4
                ], dim=-1)
                
                dummy_features = [torch.zeros(total_samples, device=device) for _ in range(6)]
                dummy_masks = [torch.zeros(total_samples, device=device) for _ in range(6)]
                
                masked_features_encoded = self.masked_feature_encoder(dummy_features, dummy_masks)
                condition_repeated = self.condition_encoder(dummy_static, masked_features_encoded)
            
            # Sample from conditional or standard prior
            if self.use_conditional_prior:
                # Sample from conditional priors
                waveform_prior_mu, waveform_prior_logvar = self.waveform_prior(condition_repeated)
                waveform_z = self._sample_from_prior(waveform_prior_mu, waveform_prior_logvar, temperature)
                
                peak_prior_input = torch.cat([condition_repeated, waveform_z], dim=-1)
                peak_prior_mu, peak_prior_logvar = self.peak_prior(peak_prior_input)
                peak_z = self._sample_from_prior(peak_prior_mu, peak_prior_logvar, temperature)
                
            else:
                # Sample from standard normal prior
                waveform_z = torch.randn(total_samples, self.waveform_latent_dim, device=device) * temperature
                peak_z = torch.randn(total_samples, self.peak_latent_dim, device=device) * temperature
            
            # Combine latent variables
            combined_z = torch.cat([waveform_z, peak_z], dim=-1)
            
            # Generate ABR waveforms
            generated_abr = self.temporal_decoder(combined_z, condition_repeated)
            
            # Generate peaks
            peak_input = torch.cat([combined_z, condition_repeated], dim=-1)
            peak_predictions = self.peak_head(peak_input)
            
            # Generate static parameters
            generated_static_params = self.static_param_decoder(combined_z)
            
            return {
                'abr_waveforms': generated_abr,
                'peak_predictions': peak_predictions,
                'generated_static_params': generated_static_params,
                'waveform_z': waveform_z,
                'peak_z': peak_z,
                'combined_z': combined_z,
                'condition_embeddings': condition_repeated
            }
    
    def _sample_from_prior(self, mu: torch.Tensor, logvar: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample from conditional prior with temperature scaling."""
        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], reduction: str = 'mean') -> Dict[str, torch.Tensor]:
        """Compute multi-task loss with hierarchical KL divergence."""
        
        outputs = self.forward(batch)
        
        time_series = batch['time_series']
        device = time_series.device
        
        # ABR reconstruction loss
        abr_recon_loss = F.mse_loss(
            outputs['abr_reconstruction'], 
            time_series, 
            reduction=reduction
        )
        
        # Hierarchical KL divergence losses
        waveform_kl = self._compute_kl_loss(
            outputs['waveform_mu'], 
            outputs['waveform_logvar'],
            outputs.get('waveform_prior_mu'),
            outputs.get('waveform_prior_logvar'),
            reduction
        )
        
        peak_kl = self._compute_kl_loss(
            outputs['peak_mu'], 
            outputs['peak_logvar'],
            outputs.get('peak_prior_mu'),
            outputs.get('peak_prior_logvar'),
            reduction
        )
        
        # Peak prediction losses
        peak_losses = self._compute_peak_losses(batch, outputs['peak_predictions'], reduction)
        
        # Static parameter losses
        static_losses = self._compute_static_param_losses(batch, outputs['generated_static_params'], reduction)
        
        # Total loss with improved weighting
        total_loss = (abr_recon_loss + 
                     self.beta_waveform * waveform_kl + 
                     self.beta_peak * peak_kl +
                     1.0 * peak_losses['total_peak_loss'] +
                     2.0 * static_losses['total_static_loss'])  # Increased static param weight
        
        # Combined KL loss for trainer compatibility
        combined_kl_loss = waveform_kl + peak_kl
        
        return {
            'total_loss': total_loss,
            'abr_recon_loss': abr_recon_loss,
            'waveform_kl_loss': waveform_kl,
            'peak_kl_loss': peak_kl,
            'kl_loss': combined_kl_loss,  # For trainer compatibility
            **peak_losses,
            **static_losses
        }
    
    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor,
                        prior_mu: Optional[torch.Tensor] = None,
                        prior_logvar: Optional[torch.Tensor] = None,
                        reduction: str = 'mean') -> torch.Tensor:
        """Compute KL divergence loss with optional conditional prior."""
        
        mu = torch.clamp(mu, min=-10, max=10)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        if prior_mu is not None and prior_logvar is not None:
            # KL divergence with conditional prior
            prior_mu = torch.clamp(prior_mu, min=-10, max=10)
            prior_logvar = torch.clamp(prior_logvar, min=-10, max=10)
            
            kl_loss = 0.5 * torch.sum(
                prior_logvar - logvar + 
                (logvar.exp() + (mu - prior_mu).pow(2)) / prior_logvar.exp() - 1,
                dim=-1
            )
        else:
            # Standard KL divergence with N(0,1) prior
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
        if torch.isnan(kl_loss).any():
            kl_loss = torch.zeros_like(kl_loss)
        
        if reduction == 'mean':
            return kl_loss.mean()
        elif reduction == 'sum':
            return kl_loss.sum()
        return kl_loss
    
    def _compute_peak_losses(self, batch: Dict[str, torch.Tensor], 
                           peak_predictions: Dict[str, torch.Tensor],
                           reduction: str = 'mean') -> Dict[str, torch.Tensor]:
        """Compute multi-task peak prediction losses."""
        
        # Extract ground truth peak data
        gt_latencies = torch.stack([
            batch['latency_data']['I Latancy'],
            batch['latency_data']['III Latancy'],
            batch['latency_data']['V Latancy']
        ], dim=1)  # [batch_size, 3]
        
        gt_amplitudes = torch.stack([
            batch['amplitude_data']['I Amplitude'],
            batch['amplitude_data']['III Amplitude'],
            batch['amplitude_data']['V Amplitude']
        ], dim=1)  # [batch_size, 3]
        
        gt_presence = torch.stack([
            batch['latency_masks']['I Latancy'],
            batch['latency_masks']['III Latancy'],
            batch['latency_masks']['V Latancy']
        ], dim=1)  # [batch_size, 3]
        
        # Peak presence loss (binary classification)
        presence_loss = F.binary_cross_entropy_with_logits(
            peak_predictions['presence_logits'],
            gt_presence,
            reduction=reduction
        )
        
        # Peak latency loss (only for present peaks)
        latency_loss = 0.0
        amplitude_loss = 0.0
        valid_peaks = gt_presence.sum()
        
        if valid_peaks > 0:
            # Mask for present peaks
            present_mask = gt_presence == 1
            
            if present_mask.sum() > 0:
                latency_loss = F.mse_loss(
                    peak_predictions['latencies'][present_mask],
                    gt_latencies[present_mask],
                    reduction=reduction
                )
                
                amplitude_loss = F.mse_loss(
                    peak_predictions['amplitudes'][present_mask],
                    gt_amplitudes[present_mask],
                    reduction=reduction
                )
        
        total_peak_loss = presence_loss + latency_loss + amplitude_loss
        
        return {
            'total_peak_loss': total_peak_loss,
            'peak_presence_loss': presence_loss,
            'peak_latency_loss': latency_loss,
            'peak_amplitude_loss': amplitude_loss
        }
    
    def _compute_static_param_losses(self, batch: Dict[str, torch.Tensor],
                                   generated_params: Dict[str, torch.Tensor],
                                   reduction: str = 'mean') -> Dict[str, torch.Tensor]:
        """Compute static parameter generation losses."""
        
        orig_params = batch['static_params']
        
        age_loss = F.mse_loss(generated_params['age'], orig_params['age'], reduction=reduction)
        intensity_loss = F.mse_loss(generated_params['intensity'], orig_params['intensity'], reduction=reduction)
        stimulus_rate_loss = F.mse_loss(generated_params['stimulus_rate'], orig_params['stimulus_rate'], reduction=reduction)
        
        hearing_loss_target = orig_params['hear_loss'].long()
        hearing_loss_loss = F.cross_entropy(
            generated_params['hearing_loss_probs'], 
            hearing_loss_target, 
            reduction=reduction
        )
        
        total_static_loss = (age_loss + intensity_loss + stimulus_rate_loss + hearing_loss_loss) / 4.0
        
        return {
            'total_static_loss': total_static_loss,
            'age_loss': age_loss,
            'intensity_loss': intensity_loss,
            'stimulus_rate_loss': stimulus_rate_loss,
            'hearing_loss_loss': hearing_loss_loss
        }


if __name__ == "__main__":
    # Test the advanced model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = AdvancedConditionalVAE(
        static_dim=4,
        masked_features_dim=64,
        waveform_latent_dim=64,
        peak_latent_dim=32,
        condition_dim=128,
        hidden_dim=256,
        sequence_length=200,
        use_conditional_prior=True,
        noise_augmentation=0.1
    ).to(device)
    
    # Create dummy batch
    batch_size = 4
    batch = {
        'time_series': torch.randn(batch_size, 200).to(device),
        'static_params': {
            'age': torch.randn(batch_size).to(device),
            'intensity': torch.randn(batch_size).to(device),
            'stimulus_rate': torch.randn(batch_size).to(device),
            'hear_loss': torch.randint(0, 5, (batch_size,)).to(device)
        },
        'latency_data': {
            'I Latancy': torch.randn(batch_size).to(device),
            'III Latancy': torch.randn(batch_size).to(device),
            'V Latancy': torch.randn(batch_size).to(device)
        },
        'latency_masks': {
            'I Latancy': torch.randint(0, 2, (batch_size,)).float().to(device),
            'III Latancy': torch.randint(0, 2, (batch_size,)).float().to(device),
            'V Latancy': torch.ones(batch_size).to(device)
        },
        'amplitude_data': {
            'I Amplitude': torch.randn(batch_size).to(device),
            'III Amplitude': torch.randn(batch_size).to(device),
            'V Amplitude': torch.randn(batch_size).to(device)
        },
        'amplitude_masks': {
            'I Amplitude': torch.randint(0, 2, (batch_size,)).float().to(device),
            'III Amplitude': torch.randint(0, 2, (batch_size,)).float().to(device),
            'V Amplitude': torch.ones(batch_size).to(device)
        }
    }
    
    print("Testing Advanced CVAE model...")
    
    # Forward pass
    outputs = model(batch)
    print(f"ABR reconstruction shape: {outputs['abr_reconstruction'].shape}")
    print(f"Waveform latent shape: {outputs['waveform_z'].shape}")
    print(f"Peak latent shape: {outputs['peak_z'].shape}")
    print(f"Peak predictions keys: {outputs['peak_predictions'].keys()}")
    
    # Compute loss
    loss_dict = model.compute_loss(batch)
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"ABR reconstruction loss: {loss_dict['abr_recon_loss'].item():.4f}")
    print(f"Waveform KL loss: {loss_dict['waveform_kl_loss'].item():.4f}")
    print(f"Peak KL loss: {loss_dict['peak_kl_loss'].item():.4f}")
    
    # Generate samples
    generated = model.generate(batch, num_samples=2, use_conditioning=True)
    print(f"Generated samples shape: {generated['abr_waveforms'].shape}")
    print(f"Generated peak predictions shape: {generated['peak_predictions']['presence_probs'].shape}")
    
    # Test unconditional generation
    unconditional = model.generate(num_samples=3, use_conditioning=False, temperature=0.8)
    print(f"Unconditional samples shape: {unconditional['abr_waveforms'].shape}")
    
    print("Advanced CVAE model test completed successfully!") 