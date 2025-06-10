"""
Training Module - ABR CVAE Project
=================================

Training loops, optimization, and model fitting utilities.

Components:
    - trainer: Main training class with logging and checkpointing
    - utils: Training utilities and helper functions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .trainer import CVAETrainer

__all__ = [
    'CVAETrainer'
] 