#!/usr/bin/env python3
"""
Model Architecture Visualization Script
======================================

This script creates an SVG visualization of the ABR CVAE model architecture,
showing all components, data flow, and dimensions.
"""

import svgwrite
from svgwrite import cm, mm
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_architecture_svg():
    """Create SVG visualization of the ABR CVAE architecture."""
    
    # Load configuration
    config_path = "outputs/config.json"
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration if file doesn't exist
        config = {
            'static_dim': 4,
            'masked_features_dim': 64,
            'latent_dim': 128,
            'condition_dim': 128,
            'hidden_dim': 256,
            'sequence_length': 200,
            'num_encoder_layers': 4,
            'num_decoder_layers': 4,
            'num_heads': 8
        }
    
    # Create SVG document
    dwg = svgwrite.Drawing('outputs/abr_cvae_architecture.svg', size=('1400px', '1000px'))
    
    # Define colors
    colors = {
        'input': '#E3F2FD',          # Light blue
        'encoder': '#FFF3E0',        # Light orange
        'latent': '#F3E5F5',         # Light purple
        'decoder': '#E8F5E8',        # Light green
        'output': '#FFF9C4',         # Light yellow
        'condition': '#FCE4EC',      # Light pink
        'border': '#424242',         # Dark gray
        'text': '#212121',           # Very dark gray
        'arrow': '#1976D2'           # Blue
    }
    
    # Define styles
    dwg.defs.add(dwg.style("""
        .input-box { fill: %s; stroke: %s; stroke-width: 2; }
        .encoder-box { fill: %s; stroke: %s; stroke-width: 2; }
        .latent-box { fill: %s; stroke: %s; stroke-width: 2; }
        .decoder-box { fill: %s; stroke: %s; stroke-width: 2; }
        .output-box { fill: %s; stroke: %s; stroke-width: 2; }
        .condition-box { fill: %s; stroke: %s; stroke-width: 2; }
        .text { font-family: Arial, sans-serif; font-size: 12px; fill: %s; }
        .title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: %s; }
        .subtitle { font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; fill: %s; }
        .small-text { font-family: Arial, sans-serif; font-size: 10px; fill: %s; }
        .arrow { stroke: %s; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
    """ % (colors['input'], colors['border'],
           colors['encoder'], colors['border'],
           colors['latent'], colors['border'],
           colors['decoder'], colors['border'],
           colors['output'], colors['border'],
           colors['condition'], colors['border'],
           colors['text'], colors['text'], colors['text'], colors['text'],
           colors['arrow'])))
    
    # Add arrowhead marker
    marker = dwg.marker(insert=(0, 0), size=(10, 10), orient='auto', id='arrowhead')
    marker.add(dwg.polygon(points=[(0, 0), (0, 6), (9, 3)], fill=colors['arrow']))
    dwg.defs.add(marker)
    
    # Title
    dwg.add(dwg.text('ABR Conditional VAE Architecture', insert=(700, 30), 
                     text_anchor='middle', class_='title'))
    
    # Input section
    y_offset = 80
    
    # Time series input
    ts_box = dwg.rect(insert=(50, y_offset), size=(180, 60), class_='input-box')
    dwg.add(ts_box)
    dwg.add(dwg.text('Time Series Input', insert=(140, y_offset + 15), 
                     text_anchor='middle', class_='subtitle'))
    dwg.add(dwg.text(f'Shape: [batch, {config["sequence_length"]}]', insert=(140, y_offset + 35), 
                     text_anchor='middle', class_='text'))
    dwg.add(dwg.text('ABR waveform (200 timesteps)', insert=(140, y_offset + 50), 
                     text_anchor='middle', class_='small-text'))
    
    # Static parameters input
    static_box = dwg.rect(insert=(50, y_offset + 120), size=(180, 80), class_='condition-box')
    dwg.add(static_box)
    dwg.add(dwg.text('Static Parameters', insert=(140, y_offset + 135), 
                     text_anchor='middle', class_='subtitle'))
    dwg.add(dwg.text(f'Shape: [batch, {config["static_dim"]}]', insert=(140, y_offset + 155), 
                     text_anchor='middle', class_='text'))
    dwg.add(dwg.text('• Age', insert=(60, y_offset + 175), class_='small-text'))
    dwg.add(dwg.text('• Intensity', insert=(60, y_offset + 187), class_='small-text'))
    dwg.add(dwg.text('• Stimulus Rate', insert=(140, y_offset + 175), class_='small-text'))
    dwg.add(dwg.text('• Hear Loss Type', insert=(140, y_offset + 187), class_='small-text'))
    
    # Masked features input
    masked_box = dwg.rect(insert=(50, y_offset + 220), size=(180, 100), class_='condition-box')
    dwg.add(masked_box)
    dwg.add(dwg.text('Masked Features', insert=(140, y_offset + 235), 
                     text_anchor='middle', class_='subtitle'))
    dwg.add(dwg.text('6 features with masks', insert=(140, y_offset + 255), 
                     text_anchor='middle', class_='text'))
    dwg.add(dwg.text('Latency:', insert=(70, y_offset + 275), class_='small-text', font_weight='bold'))
    dwg.add(dwg.text('• I, III, V', insert=(70, y_offset + 287), class_='small-text'))
    dwg.add(dwg.text('Amplitude:', insert=(140, y_offset + 275), class_='small-text', font_weight='bold'))
    dwg.add(dwg.text('• I, III, V', insert=(140, y_offset + 287), class_='small-text'))
    dwg.add(dwg.text('Availability: 29.4%, 31.5%, 98.4%', insert=(140, y_offset + 305), 
                     text_anchor='middle', class_='small-text'))
    
    # Encoder section
    x_encoder = 300
    
    # ABR Encoder
    abr_encoder_box = dwg.rect(insert=(x_encoder, y_offset), size=(200, 120), class_='encoder-box')
    dwg.add(abr_encoder_box)
    dwg.add(dwg.text('ABR Encoder', insert=(x_encoder + 100, y_offset + 15), 
                     text_anchor='middle', class_='subtitle'))
    
    # Conv layers
    dwg.add(dwg.text('1D Convolutions:', insert=(x_encoder + 10, y_offset + 35), class_='text', font_weight='bold'))
    dwg.add(dwg.text('• Conv1d(1→64, k=7)', insert=(x_encoder + 15, y_offset + 50), class_='small-text'))
    dwg.add(dwg.text('• Conv1d(64→128, k=5, s=2)', insert=(x_encoder + 15, y_offset + 62), class_='small-text'))
    dwg.add(dwg.text('• Conv1d(128→256, k=3, s=2)', insert=(x_encoder + 15, y_offset + 74), class_='small-text'))
    
    # Transformer
    dwg.add(dwg.text('Transformer Encoder:', insert=(x_encoder + 10, y_offset + 90), class_='text', font_weight='bold'))
    dwg.add(dwg.text(f'• {config["num_encoder_layers"]} layers, {config["num_heads"]} heads', 
                     insert=(x_encoder + 15, y_offset + 105), class_='small-text'))
    
    # Masked Feature Encoder
    masked_encoder_box = dwg.rect(insert=(x_encoder, y_offset + 140), size=(200, 80), class_='encoder-box')
    dwg.add(masked_encoder_box)
    dwg.add(dwg.text('Masked Feature Encoder', insert=(x_encoder + 100, y_offset + 155), 
                     text_anchor='middle', class_='subtitle'))
    dwg.add(dwg.text(f'Output: [batch, {config["masked_features_dim"]}]', 
                     insert=(x_encoder + 100, y_offset + 175), text_anchor='middle', class_='text'))
    dwg.add(dwg.text('• Feature encoders', insert=(x_encoder + 15, y_offset + 195), class_='small-text'))
    dwg.add(dwg.text('• Mask embeddings', insert=(x_encoder + 15, y_offset + 207), class_='small-text'))
    
    # Condition Encoder
    cond_encoder_box = dwg.rect(insert=(x_encoder, y_offset + 240), size=(200, 80), class_='encoder-box')
    dwg.add(cond_encoder_box)
    dwg.add(dwg.text('Condition Encoder', insert=(x_encoder + 100, y_offset + 255), 
                     text_anchor='middle', class_='subtitle'))
    dwg.add(dwg.text(f'Output: [batch, {config["condition_dim"]}]', 
                     insert=(x_encoder + 100, y_offset + 275), text_anchor='middle', class_='text'))
    dwg.add(dwg.text('Combines static + masked features', insert=(x_encoder + 100, y_offset + 295), 
                     text_anchor='middle', class_='small-text'))
    
    # Latent space
    x_latent = 550
    
    # Mu and LogVar
    mu_box = dwg.rect(insert=(x_latent, y_offset + 20), size=(120, 40), class_='latent-box')
    dwg.add(mu_box)
    dwg.add(dwg.text('μ (Mean)', insert=(x_latent + 60, y_offset + 35), 
                     text_anchor='middle', class_='subtitle'))
    dwg.add(dwg.text(f'[batch, {config["latent_dim"]}]', insert=(x_latent + 60, y_offset + 50), 
                     text_anchor='middle', class_='small-text'))
    
    logvar_box = dwg.rect(insert=(x_latent, y_offset + 80), size=(120, 40), class_='latent-box')
    dwg.add(logvar_box)
    dwg.add(dwg.text('log σ² (LogVar)', insert=(x_latent + 60, y_offset + 95), 
                     text_anchor='middle', class_='subtitle'))
    dwg.add(dwg.text(f'[batch, {config["latent_dim"]}]', insert=(x_latent + 60, y_offset + 110), 
                     text_anchor='middle', class_='small-text'))
    
    # Reparameterization
    reparam_box = dwg.rect(insert=(x_latent, y_offset + 140), size=(120, 60), class_='latent-box')
    dwg.add(reparam_box)
    dwg.add(dwg.text('Reparameterization', insert=(x_latent + 60, y_offset + 155), 
                     text_anchor='middle', class_='subtitle'))
    dwg.add(dwg.text('z = μ + σ ⊙ ε', insert=(x_latent + 60, y_offset + 175), 
                     text_anchor='middle', class_='text'))
    dwg.add(dwg.text('ε ~ N(0,I)', insert=(x_latent + 60, y_offset + 190), 
                     text_anchor='middle', class_='small-text'))
    
    # Latent vector z
    z_box = dwg.rect(insert=(x_latent, y_offset + 220), size=(120, 50), class_='latent-box')
    dwg.add(z_box)
    dwg.add(dwg.text('Latent Vector z', insert=(x_latent + 60, y_offset + 235), 
                     text_anchor='middle', class_='subtitle'))
    dwg.add(dwg.text(f'[batch, {config["latent_dim"]}]', insert=(x_latent + 60, y_offset + 255), 
                     text_anchor='middle', class_='text'))
    
    # Decoder section
    x_decoder = 720
    
    # ABR Decoder
    abr_decoder_box = dwg.rect(insert=(x_decoder, y_offset), size=(200, 120), class_='decoder-box')
    dwg.add(abr_decoder_box)
    dwg.add(dwg.text('ABR Decoder', insert=(x_decoder + 100, y_offset + 15), 
                     text_anchor='middle', class_='subtitle'))
    
    # Transformer decoder
    dwg.add(dwg.text('Transformer Decoder:', insert=(x_decoder + 10, y_offset + 35), class_='text', font_weight='bold'))
    dwg.add(dwg.text(f'• {config["num_decoder_layers"]} layers, {config["num_heads"]} heads', 
                     insert=(x_decoder + 15, y_offset + 50), class_='small-text'))
    
    # Upsampling
    dwg.add(dwg.text('Upsampling:', insert=(x_decoder + 10, y_offset + 70), class_='text', font_weight='bold'))
    dwg.add(dwg.text('• ConvTranspose1d layers', insert=(x_decoder + 15, y_offset + 85), class_='small-text'))
    dwg.add(dwg.text('• 256→128→64→1 channels', insert=(x_decoder + 15, y_offset + 97), class_='small-text'))
    dwg.add(dwg.text('• Tanh activation', insert=(x_decoder + 15, y_offset + 109), class_='small-text'))
    
    # Masked Feature Decoder (optional)
    masked_decoder_box = dwg.rect(insert=(x_decoder, y_offset + 140), size=(200, 80), class_='decoder-box')
    dwg.add(masked_decoder_box)
    dwg.add(dwg.text('Masked Feature Decoder', insert=(x_decoder + 100, y_offset + 155), 
                     text_anchor='middle', class_='subtitle'))
    dwg.add(dwg.text('(Optional)', insert=(x_decoder + 100, y_offset + 175), 
                     text_anchor='middle', class_='small-text'))
    dwg.add(dwg.text('Reconstructs latency', insert=(x_decoder + 100, y_offset + 190), 
                     text_anchor='middle', class_='small-text'))
    dwg.add(dwg.text('and amplitude values', insert=(x_decoder + 100, y_offset + 202), 
                     text_anchor='middle', class_='small-text'))
    
    # Output section
    x_output = 970
    
    # Reconstructed time series
    recon_box = dwg.rect(insert=(x_output, y_offset), size=(180, 60), class_='output-box')
    dwg.add(recon_box)
    dwg.add(dwg.text('Reconstructed ABR', insert=(x_output + 90, y_offset + 15), 
                     text_anchor='middle', class_='subtitle'))
    dwg.add(dwg.text(f'Shape: [batch, {config["sequence_length"]}]', insert=(x_output + 90, y_offset + 35), 
                     text_anchor='middle', class_='text'))
    dwg.add(dwg.text('Generated waveform', insert=(x_output + 90, y_offset + 50), 
                     text_anchor='middle', class_='small-text'))
    
    # Loss computation
    loss_box = dwg.rect(insert=(x_output, y_offset + 80), size=(180, 100), class_='output-box')
    dwg.add(loss_box)
    dwg.add(dwg.text('Loss Components', insert=(x_output + 90, y_offset + 95), 
                     text_anchor='middle', class_='subtitle'))
    dwg.add(dwg.text('• Reconstruction Loss', insert=(x_output + 10, y_offset + 115), class_='small-text'))
    dwg.add(dwg.text('  MSE(original, recon)', insert=(x_output + 15, y_offset + 127), class_='small-text'))
    dwg.add(dwg.text('• KL Divergence', insert=(x_output + 10, y_offset + 145), class_='small-text'))
    dwg.add(dwg.text('  KL(q(z|x) || p(z))', insert=(x_output + 15, y_offset + 157), class_='small-text'))
    dwg.add(dwg.text(f'• Total: Recon + β×KL', insert=(x_output + 10, y_offset + 175), class_='small-text'))
    
    # Add arrows for data flow
    
    # Time series to ABR encoder
    dwg.add(dwg.line(start=(230, y_offset + 30), end=(300, y_offset + 30), class_='arrow'))
    
    # Static params to condition encoder
    dwg.add(dwg.line(start=(230, y_offset + 160), end=(270, y_offset + 160), class_='arrow'))
    dwg.add(dwg.line(start=(270, y_offset + 160), end=(270, y_offset + 280), class_='arrow'))
    dwg.add(dwg.line(start=(270, y_offset + 280), end=(300, y_offset + 280), class_='arrow'))
    
    # Masked features to masked feature encoder
    dwg.add(dwg.line(start=(230, y_offset + 270), end=(250, y_offset + 270), class_='arrow'))
    dwg.add(dwg.line(start=(250, y_offset + 270), end=(250, y_offset + 180), class_='arrow'))
    dwg.add(dwg.line(start=(250, y_offset + 180), end=(300, y_offset + 180), class_='arrow'))
    
    # ABR encoder to mu/logvar
    dwg.add(dwg.line(start=(500, y_offset + 40), end=(550, y_offset + 40), class_='arrow'))
    dwg.add(dwg.line(start=(500, y_offset + 80), end=(550, y_offset + 100), class_='arrow'))
    
    # Masked feature encoder to condition encoder
    dwg.add(dwg.line(start=(500, y_offset + 180), end=(520, y_offset + 180), class_='arrow'))
    dwg.add(dwg.line(start=(520, y_offset + 180), end=(520, y_offset + 280), class_='arrow'))
    dwg.add(dwg.line(start=(520, y_offset + 280), end=(500, y_offset + 280), class_='arrow'))
    
    # Mu/logvar to reparameterization
    dwg.add(dwg.line(start=(610, y_offset + 50), end=(630, y_offset + 50), class_='arrow'))
    dwg.add(dwg.line(start=(630, y_offset + 50), end=(630, y_offset + 170), class_='arrow'))
    dwg.add(dwg.line(start=(630, y_offset + 170), end=(610, y_offset + 170), class_='arrow'))
    
    dwg.add(dwg.line(start=(610, y_offset + 110), end=(640, y_offset + 110), class_='arrow'))
    dwg.add(dwg.line(start=(640, y_offset + 110), end=(640, y_offset + 170), class_='arrow'))
    dwg.add(dwg.line(start=(640, y_offset + 170), end=(610, y_offset + 170), class_='arrow'))
    
    # Reparameterization to z
    dwg.add(dwg.line(start=(610, y_offset + 190), end=(650, y_offset + 190), class_='arrow'))
    dwg.add(dwg.line(start=(650, y_offset + 190), end=(650, y_offset + 245), class_='arrow'))
    dwg.add(dwg.line(start=(650, y_offset + 245), end=(610, y_offset + 245), class_='arrow'))
    
    # Z and condition to decoder
    dwg.add(dwg.line(start=(670, y_offset + 245), end=(690, y_offset + 245), class_='arrow'))
    dwg.add(dwg.line(start=(690, y_offset + 245), end=(690, y_offset + 60), class_='arrow'))
    dwg.add(dwg.line(start=(690, y_offset + 60), end=(720, y_offset + 60), class_='arrow'))
    
    dwg.add(dwg.line(start=(500, y_offset + 280), end=(680, y_offset + 280), class_='arrow'))
    dwg.add(dwg.line(start=(680, y_offset + 280), end=(680, y_offset + 80), class_='arrow'))
    dwg.add(dwg.line(start=(680, y_offset + 80), end=(720, y_offset + 80), class_='arrow'))
    
    # Decoder to output
    dwg.add(dwg.line(start=(920, y_offset + 30), end=(970, y_offset + 30), class_='arrow'))
    
    # Add model statistics box
    stats_box = dwg.rect(insert=(50, y_offset + 350), size=(300, 120), 
                        fill='#F5F5F5', stroke=colors['border'], stroke_width=2)
    dwg.add(stats_box)
    dwg.add(dwg.text('Model Statistics', insert=(200, y_offset + 370), 
                     text_anchor='middle', class_='subtitle'))
    
    dwg.add(dwg.text('• Total Parameters: 11,505,991 (11.5M)', insert=(60, y_offset + 390), class_='text'))
    dwg.add(dwg.text(f'• Latent Dimension: {config["latent_dim"]}', insert=(60, y_offset + 410), class_='text'))
    dwg.add(dwg.text(f'• Hidden Dimension: {config["hidden_dim"]}', insert=(60, y_offset + 430), class_='text'))
    dwg.add(dwg.text(f'• Sequence Length: {config["sequence_length"]} timesteps', insert=(60, y_offset + 450), class_='text'))
    
    # Add training info box
    training_box = dwg.rect(insert=(370, y_offset + 350), size=(300, 120), 
                           fill='#F5F5F5', stroke=colors['border'], stroke_width=2)
    dwg.add(training_box)
    dwg.add(dwg.text('Training Configuration', insert=(520, y_offset + 370), 
                     text_anchor='middle', class_='subtitle'))
    
    dwg.add(dwg.text('• Learning Rate: 1e-4', insert=(380, y_offset + 390), class_='text'))
    dwg.add(dwg.text('• Batch Size: 8 (conservative)', insert=(380, y_offset + 410), class_='text'))
    dwg.add(dwg.text('• Beta Annealing: 0.0 → 0.1', insert=(380, y_offset + 430), class_='text'))
    dwg.add(dwg.text('• Gradient Clipping: 0.5', insert=(380, y_offset + 450), class_='text'))
    
    # Add data info box
    data_box = dwg.rect(insert=(690, y_offset + 350), size=(300, 120), 
                       fill='#F5F5F5', stroke=colors['border'], stroke_width=2)
    dwg.add(data_box)
    dwg.add(dwg.text('Dataset Information', insert=(840, y_offset + 370), 
                     text_anchor='middle', class_='subtitle'))
    
    dwg.add(dwg.text('• Total Samples: 14,917', insert=(700, y_offset + 390), class_='text'))
    dwg.add(dwg.text('• Train/Val/Test: 70%/15%/15%', insert=(700, y_offset + 410), class_='text'))
    dwg.add(dwg.text('• Time Series: Normalized', insert=(700, y_offset + 430), class_='text'))
    dwg.add(dwg.text('• Missing Data Handling: Masking', insert=(700, y_offset + 450), class_='text'))
    
    # Add legend
    legend_y = y_offset + 500
    dwg.add(dwg.text('Component Legend:', insert=(50, legend_y), class_='subtitle'))
    
    legend_items = [
        ('Input Data', colors['input']),
        ('Encoders', colors['encoder']),
        ('Latent Space', colors['latent']),
        ('Decoders', colors['decoder']),
        ('Outputs', colors['output']),
        ('Conditioning', colors['condition'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        x_pos = 50 + (i * 150)
        if i >= 3:
            x_pos = 50 + ((i - 3) * 150)
            legend_y_pos = legend_y + 40
        else:
            legend_y_pos = legend_y + 20
            
        legend_rect = dwg.rect(insert=(x_pos, legend_y_pos), size=(15, 15), 
                              fill=color, stroke=colors['border'])
        dwg.add(legend_rect)
        dwg.add(dwg.text(label, insert=(x_pos + 20, legend_y_pos + 12), class_='small-text'))
    
    # Save the SVG
    dwg.save()
    logger.info("Architecture diagram saved as outputs/abr_cvae_architecture.svg")
    
    return dwg

def main():
    """Main function to create the architecture visualization."""
    logger.info("Creating ABR CVAE architecture visualization...")
    
    # Create output directory if it doesn't exist
    Path("outputs").mkdir(exist_ok=True)
    
    # Create and save the SVG
    create_architecture_svg()
    
    print("✅ ABR CVAE Architecture diagram created successfully!")
    print("📁 Saved as: outputs/abr_cvae_architecture.svg")
    print("\nThe diagram shows:")
    print("  • Complete data flow from inputs to outputs")
    print("  • All model components with dimensions")
    print("  • Conditioning mechanism for static and masked features")
    print("  • Latent space reparameterization")
    print("  • Loss computation components")
    print("  • Model statistics and training configuration")

if __name__ == "__main__":
    main() 