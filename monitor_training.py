#!/usr/bin/env python3
"""
Training Monitor for ABR CVAE
============================

Monitor training progress in real-time.
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

def monitor_training():
    """Monitor training progress."""
    print("🔍 ABR CVAE Training Monitor")
    print("=" * 50)
    
    # Check for training files
    history_file = Path("outputs_production/training_history.json")
    log_dir = Path("outputs_production/logs")
    
    if not history_file.exists():
        print("❌ No training history found. Training may not have started yet.")
        return
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Get the latest run
        latest_run = max(history.keys(), key=lambda x: history[x]['timestamp'])
        run_data = history[latest_run]
        
        print(f"📊 Latest Run: {latest_run}")
        print(f"⏰ Started: {run_data['timestamp']}")
        print(f"📝 Notes: {run_data.get('notes', 'No notes')}")
        print()
        
        # Training progress
        training_history = run_data.get('training_history', {})
        epochs = training_history.get('epochs', [])
        train_losses = training_history.get('train_loss', [])
        val_losses = training_history.get('val_loss', [])
        
        if epochs:
            current_epoch = max(epochs) + 1
            print(f"🚀 Current Epoch: {current_epoch}")
            print(f"📈 Latest Train Loss: {train_losses[-1]:.4f}")
            print(f"📉 Latest Val Loss: {val_losses[-1]:.4f}")
            
            if len(val_losses) > 1:
                improvement = val_losses[-2] - val_losses[-1]
                if improvement > 0:
                    print(f"✅ Improvement: {improvement:.4f}")
                else:
                    print(f"⚠️ Loss increased: {abs(improvement):.4f}")
        
        # Model info
        model_arch = run_data.get('model_architecture', {})
        if model_arch:
            print(f"🏗️ Model: {model_arch.get('total_parameters', 0):,} parameters")
            print(f"💾 Model Size: {model_arch.get('total_parameters', 0) * 4 / 1e6:.1f} MB")
        
        # Training time estimate
        if 'training_time' in run_data and epochs:
            time_per_epoch = run_data['training_time'] / len(epochs)
            remaining_epochs = 150 - current_epoch  # Assuming 150 total epochs
            estimated_time = remaining_epochs * time_per_epoch / 3600
            print(f"⏱️ Estimated time remaining: {estimated_time:.1f} hours")
        
    except Exception as e:
        print(f"❌ Error reading training data: {e}")

def plot_training_curves():
    """Plot training curves if data is available."""
    history_file = Path("outputs_production/training_history.json")
    
    if not history_file.exists():
        print("❌ No training history found for plotting.")
        return
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Get the latest run
        latest_run = max(history.keys(), key=lambda x: history[x]['timestamp'])
        run_data = history[latest_run]
        training_history = run_data.get('training_history', {})
        
        epochs = training_history.get('epochs', [])
        train_losses = training_history.get('train_loss', [])
        val_losses = training_history.get('val_loss', [])
        
        if len(epochs) < 2:
            print("❌ Not enough data for plotting yet.")
            return
        
        # Create plot
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot([e+1 for e in epochs], train_losses, 'b-', label='Train Loss', alpha=0.7)
        plt.plot([e+1 for e in epochs], val_losses, 'r-', label='Val Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Recent progress (last 20 epochs)
        plt.subplot(1, 2, 2)
        recent_epochs = epochs[-20:] if len(epochs) > 20 else epochs
        recent_train = train_losses[-20:] if len(train_losses) > 20 else train_losses
        recent_val = val_losses[-20:] if len(val_losses) > 20 else val_losses
        
        plt.plot([e+1 for e in recent_epochs], recent_train, 'b-', label='Train Loss', alpha=0.7)
        plt.plot([e+1 for e in recent_epochs], recent_val, 'r-', label='Val Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Recent Progress (Last 20 Epochs)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📊 Training curves saved as 'training_progress.png'")
        
    except Exception as e:
        print(f"❌ Error plotting training curves: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        plot_training_curves()
    else:
        monitor_training() 