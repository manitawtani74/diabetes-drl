"""
Main Pipeline Orchestrator

Orchestrates the complete DRL processing pipeline:
1. Stream normalization
2. Anomaly detection
3. Gap filling
4. Trust scoring
5. Correction generation
6. Summary metrics computation

This is the main entry point for processing CGM data streams.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics

from .models import RawReading, ProcessedReading


class EmptyWindowError(Exception):
    """Raised when time window filtering results in no readings."""
    pass
from .stream_normalizer import normalize_stream
from .anomaly_detector import detect_anomalies
from .gap_filler import fill_gaps
from .trust_score import score_trust
from .sensor_report import compute_sensor_health_report
from .bolus_risk import analyze_bolus_risks
from .failure_predictor import predict_failure
from .glucose_insights import generate_glucose_insights


# Default configuration
DEFAULT_MAX_DELTA_MGDL = 50.0
DEFAULT_MAX_DELTA_MINUTES = 5
DEFAULT_EXPECTED_INTERVAL_MINUTES = 5
DEFAULT_GAP_THRESHOLD_MINUTES = 10
DEFAULT_TRUST_CORRECTION_THRESHOLD = 0.4


def run_pipeline(
    raw_readings: List[RawReading],
    max_delta_mgdl: float = DEFAULT_MAX_DELTA_MGDL,
    max_delta_minutes: int = DEFAULT_MAX_DELTA_MINUTES,
    expected_interval_minutes: int = DEFAULT_EXPECTED_INTERVAL_MINUTES,
    gap_threshold_minutes: int = DEFAULT_GAP_THRESHOLD_MINUTES,
    trust_correction_threshold: float = DEFAULT_TRUST_CORRECTION_THRESHOLD,
    glucose_targets: Optional[Dict[str, float]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> tuple[List[ProcessedReading], Dict[str, Any]]:
    """
    Run the complete DRL pipeline on raw CGM readings.
    
    Pipeline steps:
    1. Normalize stream (sort, deduplicate, fix timestamps)
    2. Detect anomalies (ROC, flatline, compression)
    3. Fill gaps (interpolate missing readings)
    4. Score trust (compute reliability scores)
    5. Generate corrections (replace low-trust readings)
    6. Compute summary metrics
    
    Args:
        raw_readings: List of raw CGM readings from input
        max_delta_mgdl: Maximum allowed glucose change (mg/dL) for ROC detection
        max_delta_minutes: Time window (minutes) for ROC calculation
        expected_interval_minutes: Expected time between readings (minutes)
        gap_threshold_minutes: Minimum gap size to trigger filling (minutes)
        trust_correction_threshold: Trust score below which readings are corrected
        glucose_targets: Dictionary with custom glucose thresholds (defaults used if None)
        start_time: Optional start time for filtering readings (inclusive)
        end_time: Optional end time for filtering readings (exclusive)
        
    Returns:
        Tuple of:
        - List of fully processed readings
        - Dictionary containing:
          - Basic metrics (num_readings, anomalies, etc.)
          - Sensor health report (health_grade, detailed metrics)
          - Bolus risk analysis (high_risk_readings, risk_summary)
          - Failure prediction (SFRI, TTF, etc.)
          - Insights (patient-friendly summaries)
    """
    # Set default glucose targets if not provided
    if glucose_targets is None:
        glucose_targets = {
            "time_in_range_low": 70.0,
            "time_in_range_high": 180.0,
            "low_threshold": 70.0,
            "very_low_threshold": 54.0,
            "high_threshold": 180.0,
            "very_high_threshold": 250.0,
        }
    
    if not raw_readings:
        return [], {
            "basic_metrics": _empty_metrics(),
            "sensor_health": {},
            "bolus_risks": {},
            "failure_prediction": {},
            "insights": []
        }
    
    # Step 1: Normalize stream
    normalized = normalize_stream(raw_readings)
    
    # Step 2: Detect anomalies
    processed = detect_anomalies(
        normalized,
        max_delta_mgdl=max_delta_mgdl,
        max_delta_minutes=max_delta_minutes
    )
    
    # Step 3: Fill gaps
    processed = fill_gaps(
        processed,
        expected_interval_minutes=expected_interval_minutes,
        gap_threshold_minutes=gap_threshold_minutes
    )
    
    # Step 4: Score trust
    processed = score_trust(processed)
    
    # Step 5: Generate corrections
    processed = _generate_corrections(processed, trust_correction_threshold)
    
    # Step 6: Compute summary metrics
    basic_metrics = _compute_metrics(processed, raw_readings)
    
    # Step 7: Compute sensor health report
    sensor_health = compute_sensor_health_report(processed)
    
    # Step 8: Analyze bolus risks
    bolus_risks = analyze_bolus_risks(processed, glucose_targets=glucose_targets)
    
    # Step 9: Predict failure (only if we have enough data)
    if len(processed) >= 5:
        failure_prediction = predict_failure(processed)
    else:
        failure_prediction = {
            "not_enough_data": True,
            "sfri": {
                "sfri_score": None,
                "risk_category": "N/A",
            },
            "ttf": {
                "ttf_category": "Not enough data",
                "ttf_confidence": "N/A",
                "reasoning": "Not enough data to compute a reliable failure prediction. Try using at least several hours or a full day of CGM data.",
            },
        }
    
    # Step 10: Generate patient-friendly insights
    insights = generate_glucose_insights(processed, glucose_targets=glucose_targets)
    
    return processed, {
        "basic_metrics": basic_metrics,
        "sensor_health": sensor_health,
        "bolus_risks": bolus_risks,
        "failure_prediction": failure_prediction,
        "insights": insights,
    }


def _generate_corrections(
    processed: List[ProcessedReading],
    trust_threshold: float
) -> List[ProcessedReading]:
    """
    Generate corrected glucose values for low-trust readings.
    
    If a reading's trust_score < threshold, replace its corrected_glucose
    with a linear interpolation between neighboring trusted readings.
    
    Args:
        processed: List of processed readings with trust scores
        trust_threshold: Trust score below which readings are corrected
        
    Returns:
        Same list with corrected_glucose values updated
    """
    if len(processed) < 2:
        return processed
    
    for i in range(len(processed)):
        reading = processed[i]
        
        if reading.trust_score < trust_threshold:
            # Find neighboring trusted readings
            prev_trusted = None
            next_trusted = None
            
            # Look backward for trusted reading
            for j in range(i - 1, -1, -1):
                if processed[j].trust_score >= trust_threshold:
                    prev_trusted = processed[j]
                    break
            
            # Look forward for trusted reading
            for j in range(i + 1, len(processed)):
                if processed[j].trust_score >= trust_threshold:
                    next_trusted = processed[j]
                    break
            
            # Interpolate if we have neighbors
            if prev_trusted and next_trusted:
                # Linear interpolation
                time_diff = (next_trusted.timestamp - prev_trusted.timestamp).total_seconds()
                if time_diff > 0:
                    current_time_diff = (reading.timestamp - prev_trusted.timestamp).total_seconds()
                    ratio = current_time_diff / time_diff
                    reading.corrected_glucose = (
                        prev_trusted.corrected_glucose +
                        (next_trusted.corrected_glucose - prev_trusted.corrected_glucose) * ratio
                    )
            elif prev_trusted:
                # Only previous trusted reading, use its value
                reading.corrected_glucose = prev_trusted.corrected_glucose
            elif next_trusted:
                # Only next trusted reading, use its value
                reading.corrected_glucose = next_trusted.corrected_glucose
            # If no trusted neighbors, keep original value
    
    return processed


def _compute_metrics(
    processed: List[ProcessedReading],
    raw_readings: List[RawReading]
) -> Dict[str, Any]:
    """
    Compute summary metrics for the processed data.
    
    Returns a dictionary with:
    - Number of raw readings
    - Number of anomalies detected
    - Number of filled readings
    - Average, min, max trust scores
    - Number of corrected readings
    """
    if not processed:
        return _empty_metrics()
    
    anomaly_count = sum(1 for r in processed if r.is_anomaly)
    filled_count = sum(1 for r in processed if r.is_filled)
    corrected_count = sum(
        1 for r in processed
        if abs(r.corrected_glucose - r.raw_glucose) > 0.01
    )
    
    trust_scores = [r.trust_score for r in processed]
    avg_trust = statistics.mean(trust_scores) if trust_scores else 0.0
    min_trust = min(trust_scores) if trust_scores else 0.0
    max_trust = max(trust_scores) if trust_scores else 0.0
    
    return {
        "num_raw_readings": len(raw_readings),
        "num_processed_readings": len(processed),
        "num_anomalies": anomaly_count,
        "num_filled": filled_count,
        "num_corrected": corrected_count,
        "avg_trust_score": round(avg_trust, 3),
        "min_trust_score": round(min_trust, 3),
        "max_trust_score": round(max_trust, 3),
    }


def _empty_metrics() -> Dict[str, Any]:
    """Return empty metrics dictionary."""
    return {
        "num_raw_readings": 0,
        "num_processed_readings": 0,
        "num_anomalies": 0,
        "num_filled": 0,
        "num_corrected": 0,
        "avg_trust_score": 0.0,
        "min_trust_score": 0.0,
        "max_trust_score": 0.0,
    }

