"""
CGM Failure Prediction Engine

Predicts sensor failure risk using deterministic, rule-based analysis of:
- Trust score trends (slope, variance, instability)
- Glucose drift patterns
- Compression-low risk patterns
- Overall sensor degradation indicators

All predictions are based on statistical analysis and pattern detection,
not machine learning.
"""

from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import statistics
import math

from .models import ProcessedReading


# Configuration constants
TRUST_TREND_WINDOW_SIZE = 10  # Number of readings for sliding window
MIN_READINGS_FOR_TREND = 5  # Minimum readings needed for trend analysis
DRIFT_DETECTION_WINDOW = 20  # Readings to analyze for drift
DRIFT_THRESHOLD_MGDL = 15.0  # Minimum change to consider as drift
COMPRESSION_PATTERN_WINDOW = 5  # Readings to check for compression patterns
INSTABILITY_VARIANCE_THRESHOLD = 0.15  # Variance threshold for instability


def compute_trust_trend(readings: List[ProcessedReading]) -> Dict[str, Any]:
    """
    Compute trust score trend analysis.
    
    Analyzes trust scores over time to detect:
    - Average trust score
    - Trust score slope (downward = degrading sensor)
    - Trust score variance (higher = instability)
    
    Args:
        readings: List of processed readings with trust scores
        
    Returns:
        Dictionary containing:
        - avg_trust: Average trust score across all readings
        - trust_slope: Slope of trust score over time (negative = degrading)
        - trust_variance: Variance of trust scores
        - recent_avg_trust: Average trust in most recent window
        - trend_direction: "improving", "degrading", or "stable"
    """
    if not readings or len(readings) < MIN_READINGS_FOR_TREND:
        return {
            "avg_trust": None,
            "trust_slope": 0.0,
            "trust_variance": 0.0,
            "recent_avg_trust": None,
            "trend_direction": "insufficient_data",
        }
    
    # Extract trust scores, filtering out None values
    trust_scores = [r.trust_score for r in readings if r.trust_score is not None]
    
    if not trust_scores:
        return {
            "avg_trust": None,
            "trust_slope": 0.0,
            "trust_variance": 0.0,
            "recent_avg_trust": None,
            "trend_direction": "insufficient_data",
        }
    
    if len(trust_scores) < 2:
        avg_trust = trust_scores[0] if trust_scores else None
        return {
            "avg_trust": round(avg_trust, 3) if avg_trust is not None else None,
            "trust_slope": 0.0,
            "trust_variance": 0.0,
            "recent_avg_trust": round(avg_trust, 3) if avg_trust is not None else None,
            "trend_direction": "insufficient_data",
        }
    
    if not trust_scores:
        avg_trust = 0.0
        trust_variance = 0.0
    else:
        avg_trust = statistics.mean(trust_scores)
        trust_variance = statistics.variance(trust_scores) if len(trust_scores) > 1 else 0.0
    
    # Compute trust slope using linear regression
    n = len(readings)
    x_values = list(range(n))  # Time indices
    y_values = trust_scores
    
    # Simple linear regression: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x_squared = sum(x * x for x in x_values)
    
    denominator = n * sum_x_squared - sum_x * sum_x
    if denominator != 0:
        trust_slope = (n * sum_xy - sum_x * sum_y) / denominator
    else:
        trust_slope = 0.0
    
    # Recent average trust (last window)
    window_size = min(TRUST_TREND_WINDOW_SIZE, len(trust_scores))
    if window_size > 0:
        recent_trust_scores = trust_scores[-window_size:]
        recent_avg_trust = statistics.mean(recent_trust_scores) if recent_trust_scores else None
    else:
        recent_avg_trust = None
    
    # Determine trend direction
    if trust_slope < -0.01:  # Significant negative slope
        trend_direction = "degrading"
    elif trust_slope > 0.01:  # Significant positive slope
        trend_direction = "improving"
    else:
        trend_direction = "stable"
    
    return {
        "avg_trust": round(avg_trust, 3) if avg_trust is not None else None,
        "trust_slope": round(trust_slope, 5),
        "trust_variance": round(trust_variance, 3),
        "recent_avg_trust": round(recent_avg_trust, 3) if recent_avg_trust is not None else None,
        "trend_direction": trend_direction,
    }


def compute_drift(readings: List[ProcessedReading]) -> Dict[str, Any]:
    """
    Detect and quantify glucose drift patterns.
    
    Analyzes glucose values over time to detect:
    - Upward drift (sensor reading higher than expected)
    - Downward drift (sensor reading lower than expected)
    - No significant drift
    
    Args:
        readings: List of processed readings
        
    Returns:
        Dictionary containing:
        - drift_direction: "upward", "downward", or "none"
        - drift_severity: "none", "mild", "moderate", or "severe"
        - drift_magnitude_mgdl: Magnitude of drift in mg/dL
        - drift_start_index: Index where drift begins
        - drift_duration_minutes: Duration of drift period
    """
    if not readings or len(readings) < 2:
        return {
            "drift_direction": "none",
            "drift_severity": "none",
            "drift_magnitude_mgdl": 0.0,
            "drift_start_index": None,
            "drift_duration_minutes": 0.0,
        }
    
    if len(readings) < DRIFT_DETECTION_WINDOW:
        return {
            "drift_direction": "none",
            "drift_severity": "none",
            "drift_magnitude_mgdl": 0.0,
            "drift_start_index": None,
            "drift_duration_minutes": 0.0,
        }
    
    # Use corrected glucose values for drift detection
    glucose_values = [r.corrected_glucose for r in readings if r.corrected_glucose is not None]
    
    if not glucose_values or len(glucose_values) < 2:
        return {
            "drift_direction": "none",
            "drift_severity": "none",
            "drift_magnitude_mgdl": 0.0,
            "drift_start_index": None,
            "drift_duration_minutes": 0.0,
        }
    
    # Compare early vs recent readings
    early_window_size = min(DRIFT_DETECTION_WINDOW // 2, len(glucose_values) // 3)
    recent_window_size = min(DRIFT_DETECTION_WINDOW // 2, len(glucose_values) // 3)
    
    if early_window_size < 3 or recent_window_size < 3:
        return {
            "drift_direction": "none",
            "drift_severity": "none",
            "drift_magnitude_mgdl": 0.0,
            "drift_start_index": None,
            "drift_duration_minutes": 0.0,
        }
    
    early_glucose = glucose_values[:early_window_size]
    recent_glucose = glucose_values[-recent_window_size:]
    
    if not early_glucose or not recent_glucose:
        return {
            "drift_direction": "none",
            "drift_severity": "none",
            "drift_magnitude_mgdl": 0.0,
            "drift_start_index": None,
            "drift_duration_minutes": 0.0,
        }
    
    if not early_glucose or not recent_glucose:
        return {
            "drift_direction": "none",
            "drift_severity": "none",
            "drift_magnitude_mgdl": 0.0,
            "drift_start_index": None,
            "drift_duration_minutes": 0.0,
        }
    
    early_avg = statistics.mean(early_glucose)
    recent_avg = statistics.mean(recent_glucose)
    
    drift_magnitude = recent_avg - early_avg
    
    # Determine drift direction
    if abs(drift_magnitude) < DRIFT_THRESHOLD_MGDL:
        drift_direction = "none"
        drift_severity = "none"
    elif drift_magnitude > 0:
        drift_direction = "upward"
        if drift_magnitude < 30:
            drift_severity = "mild"
        elif drift_magnitude < 60:
            drift_severity = "moderate"
        else:
            drift_severity = "severe"
    else:
        drift_direction = "downward"
        if abs(drift_magnitude) < 30:
            drift_severity = "mild"
        elif abs(drift_magnitude) < 60:
            drift_severity = "moderate"
        else:
            drift_severity = "severe"
    
    # Estimate drift start (find where trend changes)
    drift_start_index = _find_drift_start(readings, drift_direction)
    
    if drift_start_index is not None:
        drift_duration = (readings[-1].timestamp - readings[drift_start_index].timestamp).total_seconds() / 60.0
    else:
        drift_duration = 0.0
    
    return {
        "drift_direction": drift_direction,
        "drift_severity": drift_severity,
        "drift_magnitude_mgdl": round(abs(drift_magnitude), 2),
        "drift_start_index": drift_start_index,
        "drift_duration_minutes": round(drift_duration, 1),
    }


def _find_drift_start(readings: List[ProcessedReading], drift_direction: str) -> int:
    """
    Find the index where drift begins.
    
    Uses a sliding window approach to detect when the trend changes.
    """
    if len(readings) < 10:
        return None
    
    window_size = 5
    glucose_values = [r.corrected_glucose for r in readings if r.corrected_glucose is not None]
    
    if len(glucose_values) < window_size * 2:
        return None
    
    # Look for the point where trend becomes consistent
    for i in range(window_size, len(glucose_values) - window_size):
        early_window = glucose_values[i - window_size:i]
        recent_window = glucose_values[i:i + window_size]
        
        if not early_window or not recent_window:
            continue
        
        try:
            early_avg = statistics.mean(early_window)
            recent_avg = statistics.mean(recent_window)
        except statistics.StatisticsError:
            continue
        
        change = recent_avg - early_avg
        
        if drift_direction == "upward" and change > DRIFT_THRESHOLD_MGDL / 2:
            return i
        elif drift_direction == "downward" and change < -DRIFT_THRESHOLD_MGDL / 2:
            return i
    
    return None


def compute_compression_low_risk(readings: List[ProcessedReading]) -> Dict[str, Any]:
    """
    Estimate compression-low risk using deterministic patterns.
    
    Analyzes patterns that indicate sensor compression:
    - Sharp drops followed by recovery
    - Frequency of compression-like patterns
    - Risk level based on pattern frequency
    
    Args:
        readings: List of processed readings
        
    Returns:
        Dictionary containing:
        - compression_risk_score: Risk score 0.0-1.0
        - compression_events: Number of detected compression events
        - compression_frequency: Events per hour
        - risk_level: "low", "moderate", "high", or "critical"
    """
    if not readings or len(readings) < COMPRESSION_PATTERN_WINDOW + 2:
        return {
            "compression_risk_score": 0.0,
            "compression_events": 0,
            "compression_frequency": 0.0,
            "risk_level": "low",
            "compression_patterns": [],
        }
    
    compression_events = 0
    compression_patterns = []
    
    # Look for compression patterns: drop followed by recovery
    for i in range(1, len(readings) - 1):
        prev = readings[i - 1]
        current = readings[i]
        next_reading = readings[i + 1]
        
        # Check for drop (handle None values)
        if prev.corrected_glucose is None or current.corrected_glucose is None or next_reading.corrected_glucose is None:
            continue
        
        drop = prev.corrected_glucose - current.corrected_glucose
        if drop < 20.0:  # Minimum drop threshold
            continue
        
        # Check for recovery
        recovery = next_reading.corrected_glucose - current.corrected_glucose
        if recovery >= 15.0:  # Minimum recovery threshold
            # This looks like a compression low
            compression_events += 1
            compression_patterns.append({
                "index": i,
                "timestamp": current.timestamp.isoformat(),
                "drop_mgdl": round(drop, 2),
                "recovery_mgdl": round(recovery, 2),
            })
    
    # Calculate frequency (events per hour)
    if len(readings) > 1:
        try:
            total_hours = (readings[-1].timestamp - readings[0].timestamp).total_seconds() / 3600.0
            if total_hours > 0:
                compression_frequency = compression_events / total_hours
            else:
                compression_frequency = 0.0
        except (AttributeError, TypeError):
            compression_frequency = 0.0
    else:
        compression_frequency = 0.0
    
    # Compute risk score (0.0-1.0)
    # Higher frequency = higher risk
    if compression_frequency == 0:
        compression_risk_score = 0.0
        risk_level = "low"
    elif compression_frequency < 0.5:  # Less than 0.5 events per hour
        compression_risk_score = 0.2
        risk_level = "low"
    elif compression_frequency < 1.0:  # 0.5-1.0 events per hour
        compression_risk_score = 0.5
        risk_level = "moderate"
    elif compression_frequency < 2.0:  # 1.0-2.0 events per hour
        compression_risk_score = 0.75
        risk_level = "high"
    else:  # > 2.0 events per hour
        compression_risk_score = 1.0
        risk_level = "critical"
    
    return {
        "compression_risk_score": round(compression_risk_score, 3),
        "compression_events": compression_events,
        "compression_frequency": round(compression_frequency, 2),
        "risk_level": risk_level,
        "compression_patterns": compression_patterns[:10],  # Limit to first 10
    }


def analyze_instability_windows(readings: List[ProcessedReading]) -> Dict[str, Any]:
    """
    Analyze trust score variance and identify instability windows.
    
    Identifies time periods where trust scores are highly variable,
    indicating sensor instability.
    
    Args:
        readings: List of processed readings
        
    Returns:
        Dictionary containing:
        - instability_windows: List of time ranges with high variance
        - max_variance: Maximum variance found
        - avg_variance: Average variance across windows
        - instability_duration_minutes: Total duration of instability
    """
    if not readings or len(readings) < TRUST_TREND_WINDOW_SIZE:
        return {
            "instability_windows": [],
            "max_variance": 0.0,
            "avg_variance": 0.0,
            "instability_duration_minutes": 0.0,
        }
    
    window_size = TRUST_TREND_WINDOW_SIZE
    instability_windows = []
    variances = []
    
    # Slide window across readings
    for i in range(len(readings) - window_size + 1):
        window_readings = readings[i:i + window_size]
        trust_scores = [r.trust_score for r in window_readings]
        
        if len(trust_scores) > 1:
            try:
                variance = statistics.variance(trust_scores)
                variances.append(variance)
            except (statistics.StatisticsError, ValueError, ZeroDivisionError):
                variances.append(0.0)
            
            # Check if variance exceeds threshold
            if variance > INSTABILITY_VARIANCE_THRESHOLD:
                start_time = window_readings[0].timestamp
                end_time = window_readings[-1].timestamp
                
                try:
                    avg_trust = statistics.mean(trust_scores) if trust_scores else 0.0
                    instability_windows.append({
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                        "variance": round(variance, 3),
                        "avg_trust": round(avg_trust, 3),
                    })
                except statistics.StatisticsError:
                    instability_windows.append({
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                        "variance": round(variance, 3),
                        "avg_trust": 0.0,
                    })
    
    # Calculate total instability duration
    instability_duration = 0.0
    if instability_windows:
        # Merge overlapping windows
        merged_windows = _merge_overlapping_windows(instability_windows)
        for window in merged_windows:
            start = datetime.fromisoformat(window["start"])
            end = datetime.fromisoformat(window["end"])
            duration = (end - start).total_seconds() / 60.0
            instability_duration += duration
    
    max_variance = max(variances) if variances else 0.0
    avg_variance = statistics.mean(variances) if len(variances) > 0 else 0.0
    
    return {
        "instability_windows": instability_windows[:20],  # Limit to first 20
        "max_variance": round(max_variance, 3),
        "avg_variance": round(avg_variance, 3),
        "instability_duration_minutes": round(instability_duration, 1),
    }


def _merge_overlapping_windows(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge overlapping time windows."""
    if not windows:
        return []
    
    # Sort by start time
    sorted_windows = sorted(windows, key=lambda w: w["start"])
    merged = [sorted_windows[0]]
    
    for current in sorted_windows[1:]:
        last = merged[-1]
        current_start = datetime.fromisoformat(current["start"])
        last_end = datetime.fromisoformat(last["end"])
        
        if current_start <= last_end:
            # Overlapping, merge
            if datetime.fromisoformat(current["end"]) > last_end:
                last["end"] = current["end"]
            # Use higher variance
            if current["variance"] > last["variance"]:
                last["variance"] = current["variance"]
        else:
            # Not overlapping, add as new window
            merged.append(current)
    
    return merged


def compute_sensor_failure_risk_index(
    readings: List[ProcessedReading],
    trust_trend: Dict[str, Any],
    drift: Dict[str, Any],
    compression_risk: Dict[str, Any],
    instability: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute Sensor Failure Risk Index (SFRI) on 0.0-1.0 scale.
    
    Combines multiple indicators to produce an overall failure risk score:
    - Trust trend degradation
    - Drift severity
    - Compression risk
    - Instability
    
    Args:
        readings: List of processed readings
        trust_trend: Results from compute_trust_trend
        drift: Results from compute_drift
        compression_risk: Results from compute_compression_low_risk
        instability: Results from analyze_instability_windows
        
    Returns:
        Dictionary containing:
        - sfri_score: Overall risk index 0.0-1.0
        - risk_factors: Breakdown of contributing factors
        - risk_category: "low", "moderate", "high", or "critical"
    """
    risk_factors = {}
    
    # Factor 1: Trust trend (0.0-0.3 weight)
    trust_weight = 0.3
    if trust_trend.get("trend_direction") == "degrading":
        trust_slope = trust_trend.get("trust_slope", 0.0) or 0.0
        trust_risk = min(0.3, abs(trust_slope) * 100)  # Scale slope
        recent_trust = trust_trend.get("recent_avg_trust")
        if recent_trust is not None and recent_trust < 0.5:
            trust_risk = 0.3  # Max weight if recent trust is very low
    elif trust_trend.get("trend_direction") == "stable":
        trust_risk = 0.1
    else:
        trust_risk = 0.0
    
    risk_factors["trust_trend"] = round(trust_risk * trust_weight, 3)
    
    # Factor 2: Drift (0.0-0.25 weight)
    drift_weight = 0.25
    if drift["drift_severity"] == "severe":
        drift_risk = 0.25
    elif drift["drift_severity"] == "moderate":
        drift_risk = 0.15
    elif drift["drift_severity"] == "mild":
        drift_risk = 0.08
    else:
        drift_risk = 0.0
    
    risk_factors["drift"] = round(drift_risk * drift_weight, 3)
    
    # Factor 3: Compression risk (0.0-0.25 weight)
    compression_weight = 0.25
    compression_risk_score = compression_risk["compression_risk_score"]
    risk_factors["compression"] = round(compression_risk_score * compression_weight, 3)
    
    # Factor 4: Instability (0.0-0.2 weight)
    instability_weight = 0.2
    if instability["max_variance"] > 0.3:
        instability_risk = 0.2
    elif instability["max_variance"] > 0.2:
        instability_risk = 0.12
    elif instability["max_variance"] > INSTABILITY_VARIANCE_THRESHOLD:
        instability_risk = 0.06
    else:
        instability_risk = 0.0
    
    risk_factors["instability"] = round(instability_risk * instability_weight, 3)
    
    # Calculate total SFRI
    sfri_score = sum(risk_factors.values())
    sfri_score = min(1.0, max(0.0, sfri_score))  # Clamp to 0.0-1.0
    
    # Determine risk category
    if sfri_score < 0.25:
        risk_category = "low"
    elif sfri_score < 0.5:
        risk_category = "moderate"
    elif sfri_score < 0.75:
        risk_category = "high"
    else:
        risk_category = "critical"
    
    return {
        "sfri_score": round(sfri_score, 3),
        "risk_factors": risk_factors,
        "risk_category": risk_category,
    }


def estimate_time_to_failure(
    readings: List[ProcessedReading],
    sfri: Dict[str, Any],
    trust_trend: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Estimate Time-To-Failure (TTF) based on current indicators.
    
    Categories:
    - 0-3 hours: Immediate risk
    - 3-6 hours: Near-term risk
    - 6-12 hours: Medium-term risk
    - Low Risk: No immediate failure expected
    
    Args:
        readings: List of processed readings
        sfri: Results from compute_sensor_failure_risk_index
        trust_trend: Results from compute_trust_trend
        
    Returns:
        Dictionary containing:
        - ttf_category: "0-3 hours", "3-6 hours", "6-12 hours", or "Low Risk"
        - ttf_confidence: Confidence level in the estimate
        - reasoning: Explanation of the estimate
    """
    sfri_score = sfri.get("sfri_score", 0.0) or 0.0
    trust_slope = trust_trend.get("trust_slope", 0.0) or 0.0
    recent_trust = trust_trend.get("recent_avg_trust")
    
    # Handle insufficient data case
    if trust_trend.get("trend_direction") == "insufficient_data" or recent_trust is None:
        return {
            "ttf_category": "Low Risk",
            "ttf_confidence": "low",
            "reasoning": "Not enough data to compute a reliable time-to-failure estimate. Try using at least several hours or a full day of CGM data.",
        }
    
    # Determine TTF based on multiple factors
    if sfri_score >= 0.75 or recent_trust < 0.3:
        ttf_category = "0-3 hours"
        confidence = "high"
        reasoning = "Critical risk indicators detected. The sensor shows severe degradation with very low trust scores according to our analysis."
    elif sfri_score >= 0.5 or (recent_trust < 0.5 and trust_slope < -0.02):
        ttf_category = "3-6 hours"
        confidence = "moderate"
        reasoning = "High risk indicators detected. Trust scores are declining rapidly or multiple failure patterns were found."
    elif sfri_score >= 0.25 or trust_slope < -0.01:
        ttf_category = "6-12 hours"
        confidence = "moderate"
        reasoning = "Moderate risk indicators detected. Some degradation patterns were found, but the sensor may continue functioning."
    else:
        ttf_category = "Low Risk"
        confidence = "high"
        reasoning = "No significant failure indicators detected. The sensor appears stable according to our rule-based analysis."
    
    return {
        "ttf_category": ttf_category,
        "ttf_confidence": confidence,
        "reasoning": reasoning,
    }


def predict_failure(readings: List[ProcessedReading]) -> Dict[str, Any]:
    """
    Main function to run complete failure prediction analysis.
    
    Orchestrates all failure prediction components and returns
    comprehensive results.
    
    Args:
        readings: List of processed readings
        
    Returns:
        Dictionary containing all failure prediction results
    """
    if not readings or len(readings) < 5:
        return {
            "trust_trend": {
                "avg_trust": None,
                "trust_slope": 0.0,
                "trust_variance": 0.0,
                "recent_avg_trust": None,
                "trend_direction": "insufficient_data",
            },
            "drift": {
                "drift_direction": "none",
                "drift_severity": "none",
                "drift_magnitude_mgdl": 0.0,
                "drift_start_index": None,
                "drift_duration_minutes": 0.0,
            },
            "compression_risk": {
                "compression_risk_score": 0.0,
                "compression_events": 0,
                "compression_frequency": 0.0,
                "risk_level": "low",
                "compression_patterns": [],
            },
            "instability": {
                "instability_windows": [],
                "max_variance": 0.0,
                "avg_variance": 0.0,
                "instability_duration_minutes": 0.0,
            },
            "sfri": {
                "sfri_score": 0.0,
                "risk_factors": {},
                "risk_category": "low",
            },
            "ttf": {
                "ttf_category": "Low Risk",
                "ttf_confidence": "low",
                "reasoning": "Not enough data to compute a reliable time-to-failure estimate. Try using at least several hours or a full day of CGM data.",
            },
            "not_enough_data": True,
        }
    
    # Run all analyses
    trust_trend = compute_trust_trend(readings)
    drift = compute_drift(readings)
    compression_risk = compute_compression_low_risk(readings)
    instability = analyze_instability_windows(readings)
    
    # Compute SFRI
    sfri = compute_sensor_failure_risk_index(
        readings, trust_trend, drift, compression_risk, instability
    )
    
    # Estimate TTF
    ttf = estimate_time_to_failure(readings, sfri, trust_trend)
    
    return {
        "trust_trend": trust_trend,
        "drift": drift,
        "compression_risk": compression_risk,
        "instability": instability,
        "sfri": sfri,
        "ttf": ttf,
    }

