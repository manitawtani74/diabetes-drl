"""
Data models for CGM readings and processed data.

Defines the structure for raw input readings and processed readings with
annotations, flags, and trust scores.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class RawReading:
    """
    Represents a raw CGM reading from the input CSV.
    
    Attributes:
        timestamp: ISO8601 timestamp string or datetime object
        glucose: Glucose value in mg/dL
    """
    timestamp: datetime
    glucose: float
    
    def __post_init__(self):
        """Ensure glucose is a float."""
        self.glucose = float(self.glucose)


@dataclass
class ProcessedReading:
    """
    Represents a processed CGM reading with reliability annotations.
    
    This is the output of the DRL pipeline, containing both the original
    data and all computed reliability metrics.
    
    Attributes:
        timestamp: Timestamp of the reading
        raw_glucose: Original glucose value from input
        corrected_glucose: Corrected glucose value (may be same as raw if trusted)
        is_anomaly: Whether this reading was flagged as anomalous
        anomaly_reasons: List of reasons why this reading is anomalous
        is_filled: Whether this reading was interpolated to fill a gap
        trust_score: Reliability score between 0.0 and 1.0
    """
    timestamp: datetime
    raw_glucose: float
    corrected_glucose: Optional[float] = None
    is_anomaly: bool = False
    anomaly_reasons: List[str] = field(default_factory=list)
    is_filled: bool = False
    trust_score: float = 1.0
    
    def __post_init__(self):
        """Ensure corrected_glucose defaults to raw_glucose if not set."""
        if self.corrected_glucose is None:
            self.corrected_glucose = self.raw_glucose

