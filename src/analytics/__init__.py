"""
Analytics Package
Generic analytics engine for multi-source metrics processing
"""

from .generic_metrics_engine import AnalysisResult, GenericMetricsEngine, SLOResult, SLOTarget

__all__ = ["GenericMetricsEngine", "SLOTarget", "SLOResult", "AnalysisResult"]
