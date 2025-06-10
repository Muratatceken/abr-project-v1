import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
from typing import Dict, Tuple, Optional, List
import math
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based architectures."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MaskedFeatureEncoder(nn.Module):
    """Encoder for masked latency and amplitude features."""
    
    def __init__(self, 
                 num_features: int = 6,  # 3 latency + 3 amplitude
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Feature-wise encoders
        self.feature_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2)
            ) for _ in range(num_features)
        ])
        
        # Mask embedding
        self.mask_embedding = nn.Embedding(2, hidden_dim // 2)  # 0: missing, 1: present
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(num_features * hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, features: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors [batch_size] for each feature
            masks: List of mask tensors [batch_size] for each feature
        Returns:
            encoded_features: [batch_size, output_dim]
        """
        batch_size = features[0].size(0)
        encoded_features = []
        
        for i, (feat, mask) in enumerate(zip(features, masks)):
            # Encode feature value
            feat_encoded = self.feature_encoders[i](feat.unsqueeze(-1))  # [batch_size, hidden_dim//2]
            
            # Encode mask
            mask_encoded = self.mask_embedding(mask.long())  # [batch_size, hidden_dim//2]
            
            # Combine feature and mask encoding
            # When mask=0 (missing), use only mask embedding
            # When mask=1 (present), use both feature and mask embedding
            combined = feat_encoded * mask.unsqueeze(-1) + mask_encoded
            encoded_features.append(combined)
        
        # Concatenate all features
        all_features = torch.cat(encoded_features, dim=-1)  # [batch_size, num_features * hidden_dim//2]
        
        # Final projection
        output = self.final_proj(all_features)  # [batch_size, output_dim]
        
        return output

class ConditionEncoder(nn.Module):
    """Enhanced encoder for conditioning information including masked features."""
    
    def __init__(self, 
                 static_dim: int = 4,  # age, intensity, stimulus_rate, hear_loss
                 masked_features_dim: int = 64,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        # Static parameters encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Combined encoder
        self.combined_encoder = nn.Sequential(
            nn.Linear(hidden_dim + masked_features_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    
    def forward(self, static_params: torch.Tensor, masked_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            static_params: [batch_size, static_dim]
            masked_features: [batch_size, masked_features_dim]
        Returns:
            condition_embedding: [batch_size, output_dim]
        """
        # Encode static parameters
        static_encoded = self.static_encoder(static_params)
        
        # Combine static and masked features
        combined = torch.cat([static_encoded, masked_features], dim=-1)
        
        # Final encoding
        return self.combined_encoder(combined)

class ABREncoder(nn.Module):
    """Encoder for ABR waveform data using 1D convolutions and transformers."""
    
    def __init__(self,
                 input_dim: int = 1,  # Single channel time series
                 hidden_dim: int = 256,
                 latent_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 sequence_length: int = 200):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 1D Convolutional layers for local feature extraction
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
        
        # Calculate sequence length after convolutions
        self.conv_seq_len = sequence_length // 4  # Due to stride=2 twice
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, self.conv_seq_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Global pooling and projection to latent space
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Latent space projection (mean and log variance)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, sequence_length] (single channel)
        Returns:
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]
        """
        batch_size = x.size(0)
        
        # Add channel dimension for conv1d: [batch_size, 1, sequence_length]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Convolutional feature extraction
        x = self.conv_layers(x)  # [batch_size, hidden_dim, conv_seq_len]
        
        # Transpose back for transformer: [batch_size, conv_seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer encoding
        x = self.transformer(x)  # [batch_size, conv_seq_len, hidden_dim]
        
        # Global pooling
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, conv_seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch_size, hidden_dim]
        
        # Project to latent space
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        
        return mu, logvar

class ABRDecoder(nn.Module):
    """Decoder for generating ABR waveforms from latent code and conditions."""
    
    def __init__(self,
                 latent_dim: int = 128,
                 condition_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 1,  # Single channel
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 sequence_length: int = 200):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        
        # Initial sequence length for upsampling
        self.init_seq_len = sequence_length // 4
        
        # Project latent + condition to initial hidden state
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim * self.init_seq_len),
            nn.LayerNorm(hidden_dim * self.init_seq_len),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, sequence_length)
        
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
        
        # Create a learnable memory for the transformer decoder
        self.memory = nn.Parameter(torch.randn(sequence_length, hidden_dim))
        
        # Upsampling layers using transposed convolutions
        self.upsample_layers = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim // 4, output_dim, kernel_size=7, padding=3),
            nn.Tanh()  # Output activation
        )
        
    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, latent_dim]
            condition: [batch_size, condition_dim]
        Returns:
            reconstruction: [batch_size, sequence_length]
        """
        batch_size = z.size(0)
        
        # Combine latent and condition
        combined = torch.cat([z, condition], dim=-1)
        
        # Project to initial sequence
        x = self.latent_proj(combined)  # [batch_size, hidden_dim * init_seq_len]
        x = x.view(batch_size, self.init_seq_len, self.hidden_dim)  # [batch_size, init_seq_len, hidden_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Expand memory for batch
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, sequence_length, hidden_dim]
        
        # Transformer decoding
        x = self.transformer(x, memory)  # [batch_size, init_seq_len, hidden_dim]
        
        # Transpose for conv operations: [batch_size, hidden_dim, init_seq_len]
        x = x.transpose(1, 2)
        
        # Upsample to target sequence length
        x = self.upsample_layers(x)  # [batch_size, output_dim, sequence_length]
        
        # Remove channel dimension if single channel
        if self.output_dim == 1:
            x = x.squeeze(1)  # [batch_size, sequence_length]
        
        return x

class MaskedFeatureDecoder(nn.Module):
    """Decoder for reconstructing masked latency and amplitude features."""
    
    def __init__(self,
                 latent_dim: int = 128,
                 condition_dim: int = 128,
                 num_features: int = 6,
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_features = num_features
        
        # Shared encoder for latent + condition
        self.shared_encoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Feature-specific decoders
        self.feature_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_features)
        ])
    
    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            z: [batch_size, latent_dim]
            condition: [batch_size, condition_dim]
        Returns:
            features: List of [batch_size] tensors for each feature
        """
        # Combine latent and condition
        combined = torch.cat([z, condition], dim=-1)
        
        # Shared encoding
        shared = self.shared_encoder(combined)
        
        # Decode each feature
        features = []
        for i in range(self.num_features):
            feat = self.feature_decoders[i](shared).squeeze(-1)
            features.append(feat)
        
        return features

class StaticParameterDecoder(nn.Module):
    """Decoder for generating static parameters that are causally consistent with ABR signals."""
    
    def __init__(self,
                 latent_dim: int = 128,
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        # Shared encoder for latent representation
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
        
        # Age decoder (continuous, 0-100 years)
        self.age_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0-1, will be scaled to 0-100
        )
        
        # Intensity decoder (continuous, typically 20-100 dB)
        self.intensity_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0-1, will be scaled to 20-100 dB
        )
        
        # Stimulus rate decoder (continuous, typically 10-100 Hz)
        self.stimulus_rate_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0-1, will be scaled to 10-100 Hz
        )
        
        # Hearing loss decoder (categorical/continuous, 0-4 scale)
        self.hearing_loss_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 5),  # 5 classes: normal, mild, moderate, severe, profound
            nn.Softmax(dim=-1)
        )
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate static parameters from latent representation.
        
        Args:
            z: [batch_size, latent_dim] - latent representation
            
        Returns:
            Dictionary with generated static parameters
        """
        # Shared encoding
        shared = self.shared_encoder(z)
        
        # Generate each parameter
        age_raw = self.age_decoder(shared).squeeze(-1)  # [batch_size]
        intensity_raw = self.intensity_decoder(shared).squeeze(-1)  # [batch_size]
        stimulus_rate_raw = self.stimulus_rate_decoder(shared).squeeze(-1)  # [batch_size]
        hearing_loss_probs = self.hearing_loss_decoder(shared)  # [batch_size, 5]
        
        # Scale parameters to realistic ranges
        age = age_raw * 100.0  # 0-100 years
        intensity = intensity_raw * 80.0 + 20.0  # 20-100 dB
        stimulus_rate = stimulus_rate_raw * 90.0 + 10.0  # 10-100 Hz
        
        # Convert hearing loss probabilities to categorical values (0-4)
        hearing_loss_categorical = torch.argmax(hearing_loss_probs, dim=-1).float()
        
        return {
            'age': age,
            'intensity': intensity,
            'stimulus_rate': stimulus_rate,
            'hear_loss': hearing_loss_categorical,
            'hearing_loss_probs': hearing_loss_probs  # Keep probabilities for loss computation
        }

class ConditionalVAEWithMasking(nn.Module):
    """Enhanced Conditional VAE for ABR data with causal static parameter generation."""
    
    def __init__(self,
                 static_dim: int = 4,
                 masked_features_dim: int = 64,
                 latent_dim: int = 128,
                 condition_dim: int = 128,
                 hidden_dim: int = 256,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 sequence_length: int = 200,
                 beta: float = 1.0,
                 reconstruct_masked_features: bool = True,
                 generate_static_params: bool = True):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.beta = beta
        self.reconstruct_masked_features = reconstruct_masked_features
        self.generate_static_params = generate_static_params
        
        # Masked feature encoder
        self.masked_feature_encoder = MaskedFeatureEncoder(
            num_features=6,  # 3 latency + 3 amplitude
            hidden_dim=hidden_dim // 2,
            output_dim=masked_features_dim,
            dropout=dropout
        )
        
        # Condition encoder (now optional for generation)
        self.condition_encoder = ConditionEncoder(
            static_dim=static_dim,
            masked_features_dim=masked_features_dim,
            hidden_dim=hidden_dim,
            output_dim=condition_dim,
            dropout=dropout
        )
        
        # ABR encoder
        self.abr_encoder = ABREncoder(
            input_dim=1,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            sequence_length=sequence_length
        )
        
        # ABR decoder (now works without explicit conditioning for pure generation)
        self.abr_decoder = ABRDecoder(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            sequence_length=sequence_length
        )
        
        # Static parameter decoder (NEW)
        if generate_static_params:
            self.static_param_decoder = StaticParameterDecoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim // 2,
                dropout=dropout
            )
        
        # Masked feature decoder (optional)
        if reconstruct_masked_features:
            self.masked_feature_decoder = MaskedFeatureDecoder(
                latent_dim=latent_dim,
                condition_dim=condition_dim,
                num_features=6,
                hidden_dim=hidden_dim // 2,
                dropout=dropout
            )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                # He initialization for conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick with clamping for stability."""
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the CVAE.
        
        Args:
            batch: Dictionary containing:
                - time_series: [batch_size, sequence_length]
                - static_params: Dict with age, intensity, stimulus_rate, hear_loss
                - latency_data: Dict with latency values
                - latency_masks: Dict with latency masks
                - amplitude_data: Dict with amplitude values
                - amplitude_masks: Dict with amplitude masks
        
        Returns:
            Dictionary containing model outputs
        """
        # Extract data from batch
        time_series = batch['time_series']
        static_params = batch['static_params']
        latency_data = batch['latency_data']
        latency_masks = batch['latency_masks']
        amplitude_data = batch['amplitude_data']
        amplitude_masks = batch['amplitude_masks']
        
        # Prepare static parameters tensor
        static_tensor = torch.stack([
            static_params['age'],
            static_params['intensity'], 
            static_params['stimulus_rate'],
            static_params['hear_loss'].float()
        ], dim=-1)
        
        # Prepare masked features
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
        
        # Encode masked features
        masked_features_encoded = self.masked_feature_encoder(features, masks)
        
        # Encode conditions
        condition = self.condition_encoder(static_tensor, masked_features_encoded)
        
        # Encode ABR data
        mu, logvar = self.abr_encoder(time_series)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode ABR data
        abr_reconstruction = self.abr_decoder(z, condition)
        
        results = {
            'abr_reconstruction': abr_reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'condition': condition,
            'masked_features_encoded': masked_features_encoded
        }
        
        # Optionally generate static parameters for training
        if self.generate_static_params:
            generated_static_params = self.static_param_decoder(z)
            results['generated_static_params'] = generated_static_params
        
        # Optionally reconstruct masked features
        if self.reconstruct_masked_features:
            masked_features_reconstruction = self.masked_feature_decoder(z, condition)
            results['masked_features_reconstruction'] = masked_features_reconstruction
        
        return results
    
    def generate(self, batch: Dict[str, torch.Tensor] = None, num_samples: int = 1, 
                 use_conditioning: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate new ABR samples with causally consistent static parameters and clinical features.
        
        Args:
            batch: Optional batch containing conditioning information (only used if use_conditioning=True)
            num_samples: Number of samples to generate
            use_conditioning: Whether to use provided conditioning or generate everything from scratch
        
        Returns:
            Dictionary containing:
                - abr_waveforms: [num_samples, sequence_length]
                - generated_static_params: Dict with generated age, intensity, stimulus_rate, hear_loss
                - generated_latencies: [num_samples, 3] (I, III, V latencies)
                - generated_amplitudes: [num_samples, 3] (I, III, V amplitudes)
                - latent_codes: [num_samples, latent_dim]
                - condition_embeddings: [num_samples, condition_dim] (if conditioning used)
        """
        self.eval()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            if use_conditioning and batch is not None:
                # Use provided conditioning (original behavior)
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
                
                # Encode conditions
                masked_features_encoded = self.masked_feature_encoder(features, masks)
                condition = self.condition_encoder(static_tensor, masked_features_encoded)
                
                # Repeat conditions for multiple samples
                batch_size = condition.size(0)
                condition_repeated = condition.repeat_interleave(num_samples, dim=0)
                total_samples = batch_size * num_samples
                
                # Sample from prior
                z = torch.randn(total_samples, self.latent_dim, device=device)
                
                # Generate ABR waveforms with conditioning
                generated_abr = self.abr_decoder(z, condition_repeated)
                
                # Use provided static params
                static_repeated = static_tensor.repeat_interleave(num_samples, dim=0)
                generated_static_params = {
                    'age': static_repeated[:, 0],
                    'intensity': static_repeated[:, 1],
                    'stimulus_rate': static_repeated[:, 2],
                    'hear_loss': static_repeated[:, 3]
                }
                
                results = {
                    'abr_waveforms': generated_abr,
                    'generated_static_params': generated_static_params,
                    'latent_codes': z,
                    'condition_embeddings': condition_repeated
                }
                
            else:
                # Pure generation mode - generate everything from latent space
                # Sample from prior
                z = torch.randn(num_samples, self.latent_dim, device=device)
                
                # Generate static parameters first (these determine ABR characteristics)
                if self.generate_static_params:
                    generated_static_params = self.static_param_decoder(z)
                else:
                    # Fallback to default values if decoder not available
                    generated_static_params = {
                        'age': torch.full((num_samples,), 50.0, device=device),
                        'intensity': torch.full((num_samples,), 80.0, device=device),
                        'stimulus_rate': torch.full((num_samples,), 20.0, device=device),
                        'hear_loss': torch.full((num_samples,), 0.0, device=device)
                    }
                
                # Create condition embedding from generated static params
                static_tensor = torch.stack([
                    generated_static_params['age'],
                    generated_static_params['intensity'],
                    generated_static_params['stimulus_rate'],
                    generated_static_params['hear_loss']
                ], dim=-1)
                
                # Create dummy masked features (all zeros with masks indicating missing)
                dummy_features = [torch.zeros(num_samples, device=device) for _ in range(6)]
                dummy_masks = [torch.zeros(num_samples, device=device) for _ in range(6)]  # All masked
                
                masked_features_encoded = self.masked_feature_encoder(dummy_features, dummy_masks)
                condition = self.condition_encoder(static_tensor, masked_features_encoded)
                
                # Generate ABR waveforms conditioned on generated static params
                generated_abr = self.abr_decoder(z, condition)
                
                results = {
                    'abr_waveforms': generated_abr,
                    'generated_static_params': generated_static_params,
                    'latent_codes': z,
                    'condition_embeddings': condition
                }
            
            # Generate clinical features if decoder is available
            if self.reconstruct_masked_features:
                if use_conditioning and batch is not None:
                    generated_features = self.masked_feature_decoder(z, condition_repeated)
                else:
                    generated_features = self.masked_feature_decoder(z, condition)
                
                results['generated_latencies'] = torch.stack([
                    generated_features[0],  # I Latency
                    generated_features[1],  # III Latency
                    generated_features[2]   # V Latency
                ], dim=1)
                
                results['generated_amplitudes'] = torch.stack([
                    generated_features[3],  # I Amplitude
                    generated_features[4],  # III Amplitude
                    generated_features[5]   # V Amplitude
                ], dim=1)
            
        return results
    
    def generate_with_peak_detection(self, batch: Dict[str, torch.Tensor], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """
        Generate samples and extract peaks from the generated waveforms.
        
        Args:
            batch: Batch containing conditioning information
            num_samples: Number of samples to generate per condition
            
        Returns:
            Dictionary with generated data plus extracted peaks
        """
        # Generate samples
        results = self.generate(batch, num_samples)
        
        # Extract peaks from generated waveforms
        waveforms = results['abr_waveforms'].cpu().numpy()
        extracted_peaks = self._extract_abr_peaks(waveforms)
        
        # Add extracted peaks to results
        results['extracted_latencies'] = torch.tensor(extracted_peaks['latencies'], dtype=torch.float32)
        results['extracted_amplitudes'] = torch.tensor(extracted_peaks['amplitudes'], dtype=torch.float32)
        results['peak_confidence'] = torch.tensor(extracted_peaks['confidence'], dtype=torch.float32)
        
        return results
    
    def _extract_abr_peaks(self, waveforms: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract ABR peaks (I, III, V) from generated waveforms.
        
        Args:
            waveforms: [num_samples, sequence_length] array of ABR waveforms
            
        Returns:
            Dictionary with extracted latencies, amplitudes, and confidence scores
        """
        from scipy.signal import find_peaks
        
        num_samples = waveforms.shape[0]
        latencies = np.zeros((num_samples, 3))  # I, III, V
        amplitudes = np.zeros((num_samples, 3))
        confidence = np.zeros((num_samples, 3))
        
        # Typical ABR peak timing windows (in samples, assuming 200 points for ~10ms)
        # Adjust these based on your sampling rate and sequence length
        peak_windows = {
            'I': (10, 40),    # ~0.5-2ms
            'III': (60, 100), # ~3-5ms  
            'V': (120, 180)   # ~6-9ms
        }
        
        for i, waveform in enumerate(waveforms):
            for peak_idx, (peak_name, (start, end)) in enumerate(peak_windows.items()):
                # Find peaks in the specified window
                window_signal = waveform[start:end]
                peaks, properties = find_peaks(
                    window_signal, 
                    height=np.std(window_signal) * 0.5,  # Minimum height
                    distance=5  # Minimum distance between peaks
                )
                
                if len(peaks) > 0:
                    # Take the highest peak
                    peak_heights = properties['peak_heights']
                    max_peak_idx = np.argmax(peak_heights)
                    peak_position = peaks[max_peak_idx]
                    
                    # Convert to absolute position and time
                    absolute_position = start + peak_position
                    latencies[i, peak_idx] = absolute_position
                    amplitudes[i, peak_idx] = peak_heights[max_peak_idx]
                    confidence[i, peak_idx] = peak_heights[max_peak_idx] / np.max(np.abs(waveform))
                else:
                    # No peak found
                    latencies[i, peak_idx] = -1  # Indicate missing
                    amplitudes[i, peak_idx] = 0
                    confidence[i, peak_idx] = 0
        
        return {
            'latencies': latencies,
            'amplitudes': amplitudes,
            'confidence': confidence
        }
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], reduction: str = 'mean') -> Dict[str, torch.Tensor]:
        """
        Compute the CVAE loss with static parameter generation and numerical stability improvements.
        
        Args:
            batch: Batch of data
            reduction: Loss reduction method
        
        Returns:
            Dictionary containing loss components
        """
        # Forward pass
        outputs = self.forward(batch)
        
        # Original data
        time_series = batch['time_series']
        
        # Check for NaN in outputs
        if torch.isnan(outputs['abr_reconstruction']).any():
            logger.warning("NaN detected in ABR reconstruction")
            return {
                'total_loss': torch.tensor(float('inf'), device=time_series.device),
                'abr_recon_loss': torch.tensor(float('inf'), device=time_series.device),
                'kl_loss': torch.tensor(0.0, device=time_series.device)
            }
        
        # Reconstruction loss for ABR data with stability improvements
        abr_recon_loss = F.mse_loss(
            outputs['abr_reconstruction'], 
            time_series, 
            reduction=reduction
        )
        
        # KL divergence loss with stability improvements
        mu, logvar = outputs['mu'], outputs['logvar']
        
        # Clamp values to prevent numerical issues
        mu = torch.clamp(mu, min=-10, max=10)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
        # Check for NaN in KL loss
        if torch.isnan(kl_loss).any():
            logger.warning("NaN detected in KL loss")
            kl_loss = torch.zeros_like(kl_loss)
        
        if reduction == 'mean':
            kl_loss = kl_loss.mean()
        elif reduction == 'sum':
            kl_loss = kl_loss.sum()
        
        # Ensure beta is reasonable
        beta = max(0.0, min(self.beta, 10.0))
        
        # Total loss with gradient scaling
        total_loss = abr_recon_loss + beta * kl_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'abr_recon_loss': abr_recon_loss,
            'kl_loss': kl_loss
        }
        
        # Static parameter generation loss (NEW)
        if self.generate_static_params:
            # Generate static parameters from latent representation
            z = outputs['z']
            generated_static_params = self.static_param_decoder(z)
            
            # Original static parameters
            orig_static_params = batch['static_params']
            
            # Compute losses for each static parameter
            static_param_loss = 0.0
            
            # Age loss (continuous, MSE)
            age_loss = F.mse_loss(generated_static_params['age'], orig_static_params['age'], reduction=reduction)
            static_param_loss += age_loss
            
            # Intensity loss (continuous, MSE)
            intensity_loss = F.mse_loss(generated_static_params['intensity'], orig_static_params['intensity'], reduction=reduction)
            static_param_loss += intensity_loss
            
            # Stimulus rate loss (continuous, MSE)
            stimulus_rate_loss = F.mse_loss(generated_static_params['stimulus_rate'], orig_static_params['stimulus_rate'], reduction=reduction)
            static_param_loss += stimulus_rate_loss
            
            # Hearing loss loss (categorical, cross-entropy)
            hearing_loss_target = orig_static_params['hear_loss'].long()
            hearing_loss_loss = F.cross_entropy(generated_static_params['hearing_loss_probs'], hearing_loss_target, reduction=reduction)
            static_param_loss += hearing_loss_loss
            
            # Average the static parameter losses
            static_param_loss /= 4.0
            
            # Add to loss dictionary
            loss_dict['static_param_loss'] = static_param_loss
            loss_dict['age_loss'] = age_loss
            loss_dict['intensity_loss'] = intensity_loss
            loss_dict['stimulus_rate_loss'] = stimulus_rate_loss
            loss_dict['hearing_loss_loss'] = hearing_loss_loss
            
            # Add to total loss with weight
            loss_dict['total_loss'] += 0.5 * static_param_loss  # Weight static parameter loss
        
        # Optional masked features reconstruction loss
        if self.reconstruct_masked_features and 'masked_features_reconstruction' in outputs:
            masked_recon_loss = 0.0
            valid_feature_count = 0
            
            # Get original features and masks
            features = [
                batch['latency_data']['I Latancy'],
                batch['latency_data']['III Latancy'],
                batch['latency_data']['V Latancy'],
                batch['amplitude_data']['I Amplitude'],
                batch['amplitude_data']['III Amplitude'],
                batch['amplitude_data']['V Amplitude']
            ]
            
            masks = [
                batch['latency_masks']['I Latancy'],
                batch['latency_masks']['III Latancy'],
                batch['latency_masks']['V Latancy'],
                batch['amplitude_masks']['I Amplitude'],
                batch['amplitude_masks']['III Amplitude'],
                batch['amplitude_masks']['V Amplitude']
            ]
            
            # Compute reconstruction loss only for available features
            for i, (orig_feat, recon_feat, mask) in enumerate(
                zip(features, outputs['masked_features_reconstruction'], masks)
            ):
                if mask.sum() > 0:  # Only if there are available features
                    valid_indices = mask == 1
                    if valid_indices.sum() > 0:
                        feat_loss = F.mse_loss(
                            recon_feat[valid_indices], 
                            orig_feat[valid_indices], 
                            reduction='mean'
                        )
                        if not torch.isnan(feat_loss):
                            masked_recon_loss += feat_loss
                            valid_feature_count += 1
            
            if valid_feature_count > 0:
                masked_recon_loss /= valid_feature_count  # Average over valid features
                loss_dict['masked_features_recon_loss'] = masked_recon_loss
                loss_dict['total_loss'] += 0.1 * masked_recon_loss  # Weight the masked features loss
            else:
                loss_dict['masked_features_recon_loss'] = torch.tensor(0.0, device=total_loss.device)
        
        # Final check for NaN in total loss
        if torch.isnan(loss_dict['total_loss']):
            logger.warning("NaN in total loss, replacing with large value")
            loss_dict['total_loss'] = torch.tensor(100.0, device=total_loss.device)
        
        return loss_dict


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ConditionalVAEWithMasking(
        static_dim=4,
        masked_features_dim=64,
        latent_dim=128,
        condition_dim=128,
        hidden_dim=256,
        sequence_length=200
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
    
    print("Testing model forward pass...")
    
    # Forward pass
    outputs = model(batch)
    print(f"ABR reconstruction shape: {outputs['abr_reconstruction'].shape}")
    print(f"Latent mu shape: {outputs['mu'].shape}")
    print(f"Latent logvar shape: {outputs['logvar'].shape}")
    
    # Compute loss
    loss_dict = model.compute_loss(batch)
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"ABR reconstruction loss: {loss_dict['abr_recon_loss'].item():.4f}")
    print(f"KL loss: {loss_dict['kl_loss'].item():.4f}")
    
    # Generate samples
    generated = model.generate(batch, num_samples=2)
    print(f"Generated samples shape: {generated['abr_waveforms'].shape}")
    
    print("Model test completed successfully!") 