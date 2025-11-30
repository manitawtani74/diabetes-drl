"""
Anomaly Detector

Detects suspicious patterns in CGM readings that may indicate:
- Rate-of-change violations (sudden spikes/drops)
- Flatline regions (sensor stuck or compression)
- Compression lows (sharp drop with immediate recovery)
- Stale readings (timestamps too far apart)

This module identifies potential data quality issues that could
affect the reliability of glucose monitoring.
"""

from datetime import datetime, timedelta
from typing import List

from .models import RawReading, ProcessedReading


# Default thresholds for anomaly detection
MAX_DELTA_MGDL = 50.0  # Maximum allowed glucose change in mg/dL
MAX_DELTA_MINUTES = 5  # Time window for rate-of-change calculation
FLATLINE_THRESHOLD = 5  # Number of consecutive identical values to trigger flatline
FLATLINE_TOLERANCE = 0.1  # Tolerance for "identical" values (mg/dL)
COMPRESSION_DROP_THRESHOLD = 30.0  # Minimum drop to detect compression low
COMPRESSION_RECOVERY_THRESHOLD = 20.0  # Minimum recovery after drop


def detect_anomalies(
    readings: List[RawReading],
    max_delta_mgdl: float = MAX_DELTA_MGDL,
    max_delta_minutes: int = MAX_DELTA_MINUTES,
    flatline_threshold: int = FLATLINE_THRESHOLD,
    flatline_tolerance: float = FLATLINE_TOLERANCE
) -> List[ProcessedReading]:
    """
    Detect anomalies in a stream of CGM readings.
    
    Anomalies detected:
    1. Rate-of-change violations: Sudden changes exceeding threshold
    2. Flatline regions: Long stretches of identical/near-identical values
    3. Compression lows: Sharp drop followed by immediate recovery
    
    Args:
        readings: List of normalized raw readings
        max_delta_mgdl: Maximum allowed glucose change (mg/dL) in time window
        max_delta_minutes: Time window (minutes) for ROC calculation
        flatline_threshold: Number of consecutive identical values for flatline
        flatline_tolerance: Tolerance for "identical" values (mg/dL)
        
    Returns:
        List of ProcessedReading objects with anomaly flags set
    """
    if not readings:
        return []
    
    # Convert to ProcessedReading objects
    processed = [
        ProcessedReading(
            timestamp=r.timestamp,
            raw_glucose=r.glucose,
            corrected_glucose=r.glucose
        )
        for r in readings
    ]
    
    # Detect rate-of-change violations
    _detect_rate_of_change_anomalies(
        processed, max_delta_mgdl, max_delta_minutes
    )
    
    # Detect flatline regions
    _detect_flatlines(processed, flatline_threshold, flatline_tolerance)
    
    # Detect compression lows
    _detect_compression_lows(processed)
    
    return processed


def _detect_rate_of_change_anomalies(
    processed: List[ProcessedReading],
    max_delta_mgdl: float,
    max_delta_minutes: int
) -> None:
    """
    Detect readings where rate-of-change exceeds threshold.
    
    For each reading, check if the change from previous reading
    (within the time window) exceeds the maximum allowed delta.
    """
    if len(processed) < 2:
        return
    
    time_window = timedelta(minutes=max_delta_minutes)
    
    for i in range(1, len(processed)):
        current = processed[i]
        prev = processed[i - 1]
        
        time_diff = (current.timestamp - prev.timestamp).total_seconds() / 60.0
        
        # Only check if readings are within the time window
        if time_diff <= max_delta_minutes:
            glucose_delta = abs(current.raw_glucose - prev.raw_glucose)
            
            # Calculate rate of change per minute
            if time_diff > 0:
                roc_per_minute = glucose_delta / time_diff
                roc_in_window = roc_per_minute * max_delta_minutes
            else:
                roc_in_window = glucose_delta
            
            if roc_in_window > max_delta_mgdl:
                current.is_anomaly = True
                current.anomaly_reasons.append(
                    f"roc_exceeded: {roc_in_window:.1f} mg/dL in {time_diff:.1f} min"
                )


def _detect_flatlines(
    processed: List[ProcessedReading],
    threshold: int,
    tolerance: float
) -> None:
    """
    Detect flatline regions where sensor may be stuck.
    
    A flatline is defined as N consecutive readings with values
    within tolerance of each other.
    """
    if len(processed) < threshold:
        return
    
    i = 0
    while i < len(processed) - threshold + 1:
        # Check if we have a sequence of identical values
        base_value = processed[i].raw_glucose
        flatline_length = 1
        
        for j in range(i + 1, len(processed)):
            if abs(processed[j].raw_glucose - base_value) <= tolerance:
                flatline_length += 1
            else:
                break
        
        # If we found a flatline, mark all readings in that region
        if flatline_length >= threshold:
            for k in range(i, i + flatline_length):
                processed[k].is_anomaly = True
                if "flatline" not in processed[k].anomaly_reasons:
                    processed[k].anomaly_reasons.append(
                        f"flatline: {flatline_length} consecutive values"
                    )
            i += flatline_length
        else:
            i += 1


def _detect_compression_lows(processed: List[ProcessedReading]) -> None:
    """
    Detect compression lows: sharp drop followed by immediate recovery.
    
    Pattern: Reading drops significantly, then recovers quickly.
    This often indicates sensor compression (pressure on sensor site).
    """
    if len(processed) < 3:
        return
    
    for i in range(1, len(processed) - 1):
        prev = processed[i - 1]
        current = processed[i]
        next_reading = processed[i + 1]
        
        # Check for drop
        drop = prev.raw_glucose - current.raw_glucose
        if drop < COMPRESSION_DROP_THRESHOLD:
            continue
        
        # Check for recovery
        recovery = next_reading.raw_glucose - current.raw_glucose
        if recovery >= COMPRESSION_RECOVERY_THRESHOLD:
            # This looks like a compression low
            current.is_anomaly = True
            if "compression_low" not in current.anomaly_reasons:
                current.anomaly_reasons.append(
                    f"compression_low: drop {drop:.1f} mg/dL, recovery {recovery:.1f} mg/dL"
                )

