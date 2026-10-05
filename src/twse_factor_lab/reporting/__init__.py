"""Read-only research reporting from canonical research-cycle artifacts."""

from .loader import ReportLoaderError, load_research_report_model
from .model import ReportSection, ResearchReportModel
from .performance_adapter import PerformanceData
from .performance_metrics import PerformanceMetrics
from .performance_tearsheet import (
    PerformanceCharts,
    PerformanceReport,
    generate_performance_report,
)
from .runner import ReportResult, generate_research_report

__all__ = [
    "ReportLoaderError",
    "ReportResult",
    "ReportSection",
    "PerformanceCharts",
    "PerformanceData",
    "PerformanceMetrics",
    "PerformanceReport",
    "ResearchReportModel",
    "generate_performance_report",
    "generate_research_report",
    "load_research_report_model",
]
