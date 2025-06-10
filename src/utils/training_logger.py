import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import hashlib
import os
import shutil
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TrainingRun:
    """Data class for storing training run information."""
    run_id: str
    timestamp: str
    config: Dict[str, Any]
    model_architecture: Dict[str, Any]
    final_metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    best_epoch: int
    total_epochs: int
    training_time: float
    device: str
    git_commit: Optional[str] = None
    notes: Optional[str] = None
    data_hash: Optional[str] = None
    model_path: Optional[str] = None
    evaluation_path: Optional[str] = None

class TrainingHistoryLogger:
    """
    Comprehensive training history logger for tracking all training runs.
    
    Features:
    - Automatic run ID generation
    - Detailed metadata tracking
    - Training metrics history
    - Model and evaluation artifact storage
    - Easy comparison between runs
    - Export capabilities
    """
    
    def __init__(self, base_dir: str = "outputs", history_file: str = "training_history.json"):
        """
        Initialize the training history logger.
        
        Args:
            base_dir: Base directory for storing training outputs
            history_file: Name of the history file
        """
        self.base_dir = Path(base_dir)
        self.history_file = self.base_dir / history_file
        self.runs_dir = self.base_dir / "training_runs"
        
        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing history
        self.training_history = self._load_history()
        
        logger.info(f"Training history logger initialized. Found {len(self.training_history)} previous runs.")
    
    def _load_history(self) -> Dict[str, TrainingRun]:
        """Load existing training history."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to TrainingRun objects
                history = {}
                for run_id, run_data in data.items():
                    history[run_id] = TrainingRun(**run_data)
                
                return history
            except Exception as e:
                logger.warning(f"Could not load training history: {e}")
                return {}
        return {}
    
    def _save_history(self):
        """Save training history to file."""
        try:
            # Convert TrainingRun objects to dictionaries
            data = {}
            for run_id, run in self.training_history.items():
                data[run_id] = asdict(run)
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Training history saved to {self.history_file}")
        except Exception as e:
            logger.error(f"Could not save training history: {e}")
    
    def _generate_run_id(self, config: Dict[str, Any]) -> str:
        """Generate unique run ID based on timestamp and config hash."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create config hash for uniqueness
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"run_{timestamp}_{config_hash}"
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash if available."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def _calculate_data_hash(self, data_path: str) -> Optional[str]:
        """Calculate hash of training data for reproducibility tracking."""
        try:
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    data_hash = hashlib.md5(f.read()).hexdigest()
                return data_hash
        except:
            pass
        return None
    
    def start_training_run(self, 
                          config: Dict[str, Any], 
                          model_architecture: Dict[str, Any],
                          data_path: Optional[str] = None,
                          notes: Optional[str] = None) -> str:
        """
        Start a new training run and return the run ID.
        
        Args:
            config: Training configuration
            model_architecture: Model architecture details
            data_path: Path to training data for hash calculation
            notes: Optional notes about this training run
            
        Returns:
            run_id: Unique identifier for this training run
        """
        run_id = self._generate_run_id(config)
        
        # Create run directory
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate data hash if path provided
        data_hash = None
        if data_path:
            data_hash = self._calculate_data_hash(data_path)
        
        # Create training run object
        training_run = TrainingRun(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config=config,
            model_architecture=model_architecture,
            final_metrics={},
            training_history={},
            best_epoch=0,
            total_epochs=0,
            training_time=0.0,
            device=str(config.get('device', 'unknown')),
            git_commit=self._get_git_commit(),
            notes=notes,
            data_hash=data_hash,
            model_path=None,
            evaluation_path=None
        )
        
        # Store in history
        self.training_history[run_id] = training_run
        
        # Save config to run directory
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Save model architecture
        with open(run_dir / 'model_architecture.json', 'w') as f:
            json.dump(model_architecture, f, indent=2, default=str)
        
        logger.info(f"Started training run: {run_id}")
        return run_id
    
    def update_training_progress(self, 
                               run_id: str, 
                               epoch: int, 
                               train_loss: float, 
                               val_loss: float,
                               additional_metrics: Optional[Dict[str, float]] = None):
        """
        Update training progress for a run.
        
        Args:
            run_id: Training run identifier
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            additional_metrics: Additional metrics to track
        """
        if run_id not in self.training_history:
            logger.warning(f"Run ID {run_id} not found in history")
            return
        
        run = self.training_history[run_id]
        
        # Initialize history if empty
        if not run.training_history:
            run.training_history = {
                'train_loss': [],
                'val_loss': [],
                'epochs': []
            }
        
        # Update history
        run.training_history['epochs'].append(epoch)
        run.training_history['train_loss'].append(train_loss)
        run.training_history['val_loss'].append(val_loss)
        
        # Add additional metrics
        if additional_metrics:
            for key, value in additional_metrics.items():
                if key not in run.training_history:
                    run.training_history[key] = []
                run.training_history[key].append(value)
        
        run.total_epochs = epoch + 1
        
        # Update best epoch if validation loss improved
        val_losses = run.training_history['val_loss']
        if not val_losses or len(val_losses) == 1 or val_loss < min(val_losses[:-1]):
            run.best_epoch = epoch
    
    def finish_training_run(self, 
                          run_id: str, 
                          final_metrics: Dict[str, float],
                          training_time: float,
                          model_path: Optional[str] = None,
                          evaluation_path: Optional[str] = None):
        """
        Finish a training run and save final results.
        
        Args:
            run_id: Training run identifier
            final_metrics: Final evaluation metrics
            training_time: Total training time in seconds
            model_path: Path to saved model
            evaluation_path: Path to evaluation results
        """
        if run_id not in self.training_history:
            logger.warning(f"Run ID {run_id} not found in history")
            return
        
        run = self.training_history[run_id]
        run.final_metrics = final_metrics
        run.training_time = training_time
        run.model_path = model_path
        run.evaluation_path = evaluation_path
        
        # Save training history to run directory
        run_dir = self.runs_dir / run_id
        with open(run_dir / 'training_history.json', 'w') as f:
            json.dump(run.training_history, f, indent=2, default=str)
        
        with open(run_dir / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2, default=str)
        
        # Copy model and evaluation files if provided
        if model_path and os.path.exists(model_path):
            shutil.copy2(model_path, run_dir / 'model.pth')
        
        if evaluation_path and os.path.exists(evaluation_path):
            shutil.copy2(evaluation_path, run_dir / 'evaluation.json')
        
        # Save updated history
        self._save_history()
        
        logger.info(f"Finished training run: {run_id}")
        logger.info(f"Final metrics: {final_metrics}")
    
    def get_run_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific training run."""
        if run_id not in self.training_history:
            return None
        
        run = self.training_history[run_id]
        
        summary = {
            'run_id': run.run_id,
            'timestamp': run.timestamp,
            'total_epochs': run.total_epochs,
            'best_epoch': run.best_epoch,
            'training_time': run.training_time,
            'device': run.device,
            'final_metrics': run.final_metrics,
            'notes': run.notes
        }
        
        # Add best validation loss if available
        if run.training_history.get('val_loss'):
            summary['best_val_loss'] = min(run.training_history['val_loss'])
        
        return summary
    
    def list_all_runs(self, sort_by: str = 'timestamp') -> pd.DataFrame:
        """
        List all training runs as a DataFrame.
        
        Args:
            sort_by: Column to sort by ('timestamp', 'best_val_loss', etc.)
            
        Returns:
            DataFrame with run summaries
        """
        if not self.training_history:
            return pd.DataFrame()
        
        summaries = []
        for run_id in self.training_history:
            summary = self.get_run_summary(run_id)
            if summary:
                summaries.append(summary)
        
        df = pd.DataFrame(summaries)
        
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        
        return df
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple training runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            'runs': {},
            'best_run': None,
            'metrics_comparison': {}
        }
        
        best_val_loss = float('inf')
        
        for run_id in run_ids:
            if run_id in self.training_history:
                run = self.training_history[run_id]
                summary = self.get_run_summary(run_id)
                comparison['runs'][run_id] = summary
                
                # Track best run
                if summary and 'best_val_loss' in summary:
                    if summary['best_val_loss'] < best_val_loss:
                        best_val_loss = summary['best_val_loss']
                        comparison['best_run'] = run_id
        
        return comparison
    
    def plot_training_comparison(self, run_ids: List[str], save_path: Optional[str] = None):
        """
        Plot training curves for multiple runs.
        
        Args:
            run_ids: List of run IDs to plot
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))
        
        for i, run_id in enumerate(run_ids):
            if run_id not in self.training_history:
                continue
            
            run = self.training_history[run_id]
            history = run.training_history
            
            if not history:
                continue
            
            color = colors[i]
            label = f"{run_id[:12]}..."
            
            # Training loss
            if 'train_loss' in history:
                axes[0, 0].plot(history['epochs'], history['train_loss'], 
                              color=color, label=label, alpha=0.8)
            
            # Validation loss
            if 'val_loss' in history:
                axes[0, 1].plot(history['epochs'], history['val_loss'], 
                              color=color, label=label, alpha=0.8)
            
            # KL loss if available
            if 'kl_loss' in history:
                axes[1, 0].plot(history['epochs'], history['kl_loss'], 
                              color=color, label=label, alpha=0.8)
            
            # Beta schedule if available
            if 'beta' in history:
                axes[1, 1].plot(history['epochs'], history['beta'], 
                              color=color, label=label, alpha=0.8)
        
        # Customize plots
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('KL Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Beta Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Beta')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training comparison plot saved to {save_path}")
        
        plt.show()
    
    def export_run_data(self, run_id: str, export_path: str):
        """
        Export all data for a specific run.
        
        Args:
            run_id: Run ID to export
            export_path: Path to export directory
        """
        if run_id not in self.training_history:
            logger.error(f"Run ID {run_id} not found")
            return
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        run = self.training_history[run_id]
        
        # Export run data
        with open(export_dir / f'{run_id}_summary.json', 'w') as f:
            json.dump(asdict(run), f, indent=2, default=str)
        
        # Copy run directory if it exists
        run_dir = self.runs_dir / run_id
        if run_dir.exists():
            shutil.copytree(run_dir, export_dir / run_id, dirs_exist_ok=True)
        
        logger.info(f"Run {run_id} exported to {export_path}")
    
    def get_best_runs(self, metric: str = 'best_val_loss', top_k: int = 5) -> List[str]:
        """
        Get the best performing runs based on a metric.
        
        Args:
            metric: Metric to rank by
            top_k: Number of top runs to return
            
        Returns:
            List of run IDs sorted by performance
        """
        df = self.list_all_runs()
        
        if df.empty or metric not in df.columns:
            return []
        
        # Sort by metric (ascending for loss metrics)
        ascending = 'loss' in metric.lower() or 'error' in metric.lower()
        sorted_df = df.sort_values(metric, ascending=ascending)
        
        return sorted_df['run_id'].head(top_k).tolist()
    
    def cleanup_old_runs(self, keep_best: int = 10, keep_recent: int = 5):
        """
        Clean up old training runs, keeping only the best and most recent.
        
        Args:
            keep_best: Number of best runs to keep
            keep_recent: Number of recent runs to keep
        """
        if len(self.training_history) <= keep_best + keep_recent:
            return
        
        # Get best and recent runs
        best_runs = set(self.get_best_runs(top_k=keep_best))
        
        df = self.list_all_runs(sort_by='timestamp')
        recent_runs = set(df['run_id'].head(keep_recent).tolist())
        
        # Runs to keep
        keep_runs = best_runs.union(recent_runs)
        
        # Remove old runs
        runs_to_remove = []
        for run_id in self.training_history:
            if run_id not in keep_runs:
                runs_to_remove.append(run_id)
        
        for run_id in runs_to_remove:
            # Remove run directory
            run_dir = self.runs_dir / run_id
            if run_dir.exists():
                shutil.rmtree(run_dir)
            
            # Remove from history
            del self.training_history[run_id]
        
        # Save updated history
        self._save_history()
        
        logger.info(f"Cleaned up {len(runs_to_remove)} old training runs")


# Convenience functions for easy access
def get_training_logger(base_dir: str = "outputs") -> TrainingHistoryLogger:
    """Get a training history logger instance."""
    return TrainingHistoryLogger(base_dir)

def list_previous_trainings(base_dir: str = "outputs") -> pd.DataFrame:
    """Quick function to list all previous trainings."""
    logger = get_training_logger(base_dir)
    return logger.list_all_runs()

def compare_training_runs(run_ids: List[str], base_dir: str = "outputs") -> Dict[str, Any]:
    """Quick function to compare training runs."""
    logger = get_training_logger(base_dir)
    return logger.compare_runs(run_ids)

def plot_training_history(run_ids: List[str], base_dir: str = "outputs", save_path: Optional[str] = None):
    """Quick function to plot training history."""
    logger = get_training_logger(base_dir)
    logger.plot_training_comparison(run_ids, save_path) 