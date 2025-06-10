"""
Visualization Module - ABR CVAE Project
======================================

Plotting and visualization tools for ABR data and model results.

Components:
    - architecture: Model architecture diagrams
    - plots: Data and result plotting utilities
    - reports: Automated report generation with visualizations
"""

from .architecture import create_architecture_diagram, convert_svg_to_png

__all__ = [
    'create_architecture_diagram',
    'convert_svg_to_png'
] 