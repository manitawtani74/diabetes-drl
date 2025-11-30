"""
Sensor Health Report

Computes comprehensive sensor health metrics and assigns a health grade
based on data quality indicators.

This module analyzes processed CGM readings to assess sensor reliability
and identify potential sensor malfunctions or data quality issues.
"""

from typing import Dict, List, Any
import statistics

from .models import ProcessedReading


# Health grading thresholds
MAX_ANOMALY_RATIO_FOR_A = 0.02  # 2% or less anomalies = A grade
MAX_ANOMALY_RATIO_FOR_B = 0.05  # 5% or less anomalies = B grade
MAX_ANOMALY_RATIO_FOR_C = 0.10  # 10% or less anomalies = C grade
MAX_ANOMALY_RATIO_FOR_D = 0.20  # 20% or less anomalies = D grade
# Above 20% = F grade

MIN_AVG_TRUST_SCORE_FOR_A = 0.90  # Average trust score >= 0.90 for A
MIN_AVG_TRUST_SCORE_FOR_B = 0.80  # Average trust score >= 0.80 for B
MIN_AVG_TRUST_SCORE_FOR_C = 0.70  # Average trust score >= 0.70 for C
MIN_AVG_TRUST_SCORE_FOR_D = 0.60  # Average trust score >= 0.60 for D
# Below 0.60 = F grade

MAX_GAP_MINUTES_FOR_A = 30  # Total gap time <= 30 minutes for A
MAX_GAP_MINUTES_FOR_B = 60  # Total gap time <= 60 minutes for B
MAX_GAP_MINUTES_FOR_C = 120  # Total gap time <= 120 minutes for C
MAX_GAP_MINUTES_FOR_D = 240  # Total gap time <= 240 minutes for D
# Above 240 minutes = F grade

MAX_FLATLINE_DURATION_FOR_A = 60  # Flatline duration <= 60 minutes for A
MAX_FLATLINE_DURATION_FOR_B = 120  # Flatline duration <= 120 minutes for B
MAX_FLATLINE_DURATION_FOR_C = 240  # Flatline duration <= 240 minutes for C
MAX_FLATLINE_DURATION_FOR_D = 480  # Flatline duration <= 480 minutes for D
# Above 480 minutes = F grade


def compute_sensor_health_report(
    processed_readings: List[ProcessedReading]
) -> Dict[str, Any]:
    """
    Compute comprehensive sensor health metrics and grade.
    
    Args:
        processed_readings: List of processed readings with annotations
        
    Returns:
        Dictionary containing:
        - total_readings: Total number of readings
        - num_anomalies: Count of anomalous readings
        - num_flatline_anomalies: Count of flatline anomalies
        - num_roc_anomalies: Count of rate-of-change anomalies
        - num_compression_anomalies: Count of compression low anomalies
        - num_gaps_filled: Count of filled gap readings
        - total_gap_minutes: Total minutes of gaps that were filled
        - avg_trust_score: Average trust score across all readings
        - min_trust_score: Minimum trust score
        - max_trust_score: Maximum trust score
        - anomaly_ratio: Ratio of anomalies to total readings
        - flatline_duration_minutes: Total duration of flatline regions
        - health_grade: Sensor health grade (A, B, C, D, or F)
        - grade_explanation: Explanation of the grade
    """
    if not processed_readings:
        return _empty_health_report()
    
    total_readings = len(processed_readings)
    
    # Count anomalies by type
    num_anomalies = sum(1 for r in processed_readings if r.is_anomaly)
    num_flatline_anomalies = sum(
        1 for r in processed_readings
        if r.is_anomaly and any("flatline" in reason for reason in r.anomaly_reasons)
    )
    num_roc_anomalies = sum(
        1 for r in processed_readings
        if r.is_anomaly and any("roc_exceeded" in reason for reason in r.anomaly_reasons)
    )
    num_compression_anomalies = sum(
        1 for r in processed_readings
        if r.is_anomaly and any("compression_low" in reason for reason in r.anomaly_reasons)
    )
    
    # Count filled gaps
    num_gaps_filled = sum(1 for r in processed_readings if r.is_filled)
    
    # Calculate total gap minutes
    total_gap_minutes = _calculate_total_gap_minutes(processed_readings)
    
    # Calculate flatline duration
    flatline_duration_minutes = _calculate_flatline_duration(processed_readings)
    
    # Trust score statistics
    trust_scores = [r.trust_score for r in processed_readings]
    avg_trust_score = statistics.mean(trust_scores) if trust_scores else 0.0
    min_trust_score = min(trust_scores) if trust_scores else 0.0
    max_trust_score = max(trust_scores) if trust_scores else 0.0
    
    # Calculate anomaly ratio
    anomaly_ratio = num_anomalies / total_readings if total_readings > 0 else 0.0
    
    # Compute health grade
    health_grade, grade_explanation = _compute_health_grade(
        anomaly_ratio=anomaly_ratio,
        avg_trust_score=avg_trust_score,
        total_gap_minutes=total_gap_minutes,
        flatline_duration_minutes=flatline_duration_minutes
    )
    
    return {
        "total_readings": total_readings,
        "num_anomalies": num_anomalies,
        "num_flatline_anomalies": num_flatline_anomalies,
        "num_roc_anomalies": num_roc_anomalies,
        "num_compression_anomalies": num_compression_anomalies,
        "num_gaps_filled": num_gaps_filled,
        "total_gap_minutes": round(total_gap_minutes, 1),
        "flatline_duration_minutes": round(flatline_duration_minutes, 1),
        "avg_trust_score": round(avg_trust_score, 3),
        "min_trust_score": round(min_trust_score, 3),
        "max_trust_score": round(max_trust_score, 3),
        "anomaly_ratio": round(anomaly_ratio, 3),
        "anomaly_ratio_percent": round(anomaly_ratio * 100, 1),
        "health_grade": health_grade,
        "grade_explanation": grade_explanation,
    }


def _calculate_total_gap_minutes(processed_readings: List[ProcessedReading]) -> float:
    """
    Calculate total minutes of gaps that were filled.
    
    This estimates the total data loss by summing up the time differences
    between consecutive readings where gaps were filled.
    """
    if len(processed_readings) < 2:
        return 0.0
    
    total_gap_minutes = 0.0
    expected_interval_minutes = 5.0  # Standard CGM interval
    
    for i in range(len(processed_readings) - 1):
        current = processed_readings[i]
        next_reading = processed_readings[i + 1]
        
        # If next reading is filled, calculate the gap
        if next_reading.is_filled:
            time_diff = (next_reading.timestamp - current.timestamp).total_seconds() / 60.0
            if time_diff > expected_interval_minutes:
                # Estimate gap size (subtract expected interval)
                gap_size = time_diff - expected_interval_minutes
                total_gap_minutes += gap_size
        # Also check if there's a large gap between non-filled readings
        else:
            time_diff = (next_reading.timestamp - current.timestamp).total_seconds() / 60.0
            if time_diff > 10.0:  # Gap threshold
                gap_size = time_diff - expected_interval_minutes
                total_gap_minutes += gap_size
    
    return total_gap_minutes


def _calculate_flatline_duration(processed_readings: List[ProcessedReading]) -> float:
    """
    Calculate total duration of flatline regions in minutes.
    
    Identifies contiguous sequences of flatline readings and sums their duration.
    """
    if not processed_readings:
        return 0.0
    
    flatline_duration = 0.0
    in_flatline = False
    flatline_start_idx = None
    
    for i, reading in enumerate(processed_readings):
        is_flatline = (
            reading.is_anomaly and
            any("flatline" in reason for reason in reading.anomaly_reasons)
        )
        
        if is_flatline and not in_flatline:
            # Start of flatline region
            in_flatline = True
            flatline_start_idx = i
        elif not is_flatline and in_flatline:
            # End of flatline region
            if flatline_start_idx is not None:
                start_reading = processed_readings[flatline_start_idx]
                end_reading = processed_readings[i - 1]
                duration = (end_reading.timestamp - start_reading.timestamp).total_seconds() / 60.0
                flatline_duration += duration
            in_flatline = False
            flatline_start_idx = None
    
    # Handle case where flatline extends to end of data
    if in_flatline and flatline_start_idx is not None:
        start_reading = processed_readings[flatline_start_idx]
        end_reading = processed_readings[-1]
        duration = (end_reading.timestamp - start_reading.timestamp).total_seconds() / 60.0
        flatline_duration += duration
    
    return flatline_duration


def _compute_health_grade(
    anomaly_ratio: float,
    avg_trust_score: float,
    total_gap_minutes: float,
    flatline_duration_minutes: float
) -> tuple[str, str]:
    """
    Compute sensor health grade based on multiple factors.
    
    Grade is determined by the worst factor:
    - Anomaly ratio
    - Average trust score
    - Total gap minutes
    - Flatline duration
    
    Returns:
        Tuple of (grade, explanation)
    """
    grades = []
    reasons = []
    
    # Grade based on anomaly ratio
    if anomaly_ratio <= MAX_ANOMALY_RATIO_FOR_A:
        grades.append("A")
    elif anomaly_ratio <= MAX_ANOMALY_RATIO_FOR_B:
        grades.append("B")
        reasons.append(f"Anomaly ratio {anomaly_ratio:.1%} exceeds A threshold ({MAX_ANOMALY_RATIO_FOR_A:.1%})")
    elif anomaly_ratio <= MAX_ANOMALY_RATIO_FOR_C:
        grades.append("C")
        reasons.append(f"Anomaly ratio {anomaly_ratio:.1%} exceeds B threshold ({MAX_ANOMALY_RATIO_FOR_B:.1%})")
    elif anomaly_ratio <= MAX_ANOMALY_RATIO_FOR_D:
        grades.append("D")
        reasons.append(f"Anomaly ratio {anomaly_ratio:.1%} exceeds C threshold ({MAX_ANOMALY_RATIO_FOR_C:.1%})")
    else:
        grades.append("F")
        reasons.append(f"Anomaly ratio {anomaly_ratio:.1%} exceeds D threshold ({MAX_ANOMALY_RATIO_FOR_D:.1%})")
    
    # Grade based on average trust score
    if avg_trust_score >= MIN_AVG_TRUST_SCORE_FOR_A:
        grades.append("A")
    elif avg_trust_score >= MIN_AVG_TRUST_SCORE_FOR_B:
        grades.append("B")
        reasons.append(f"Average trust score {avg_trust_score:.2f} below A threshold ({MIN_AVG_TRUST_SCORE_FOR_A})")
    elif avg_trust_score >= MIN_AVG_TRUST_SCORE_FOR_C:
        grades.append("C")
        reasons.append(f"Average trust score {avg_trust_score:.2f} below B threshold ({MIN_AVG_TRUST_SCORE_FOR_B})")
    elif avg_trust_score >= MIN_AVG_TRUST_SCORE_FOR_D:
        grades.append("D")
        reasons.append(f"Average trust score {avg_trust_score:.2f} below C threshold ({MIN_AVG_TRUST_SCORE_FOR_C})")
    else:
        grades.append("F")
        reasons.append(f"Average trust score {avg_trust_score:.2f} below D threshold ({MIN_AVG_TRUST_SCORE_FOR_D})")
    
    # Grade based on total gap minutes
    if total_gap_minutes <= MAX_GAP_MINUTES_FOR_A:
        grades.append("A")
    elif total_gap_minutes <= MAX_GAP_MINUTES_FOR_B:
        grades.append("B")
        reasons.append(f"Total gap time {total_gap_minutes:.1f} min exceeds A threshold ({MAX_GAP_MINUTES_FOR_A} min)")
    elif total_gap_minutes <= MAX_GAP_MINUTES_FOR_C:
        grades.append("C")
        reasons.append(f"Total gap time {total_gap_minutes:.1f} min exceeds B threshold ({MAX_GAP_MINUTES_FOR_B} min)")
    elif total_gap_minutes <= MAX_GAP_MINUTES_FOR_D:
        grades.append("D")
        reasons.append(f"Total gap time {total_gap_minutes:.1f} min exceeds C threshold ({MAX_GAP_MINUTES_FOR_B} min)")
    else:
        grades.append("F")
        reasons.append(f"Total gap time {total_gap_minutes:.1f} min exceeds D threshold ({MAX_GAP_MINUTES_FOR_D} min)")
    
    # Grade based on flatline duration
    if flatline_duration_minutes <= MAX_FLATLINE_DURATION_FOR_A:
        grades.append("A")
    elif flatline_duration_minutes <= MAX_FLATLINE_DURATION_FOR_B:
        grades.append("B")
        reasons.append(f"Flatline duration {flatline_duration_minutes:.1f} min exceeds A threshold ({MAX_FLATLINE_DURATION_FOR_A} min)")
    elif flatline_duration_minutes <= MAX_FLATLINE_DURATION_FOR_C:
        grades.append("C")
        reasons.append(f"Flatline duration {flatline_duration_minutes:.1f} min exceeds B threshold ({MAX_FLATLINE_DURATION_FOR_B} min)")
    elif flatline_duration_minutes <= MAX_FLATLINE_DURATION_FOR_D:
        grades.append("D")
        reasons.append(f"Flatline duration {flatline_duration_minutes:.1f} min exceeds C threshold ({MAX_FLATLINE_DURATION_FOR_C} min)")
    else:
        grades.append("F")
        reasons.append(f"Flatline duration {flatline_duration_minutes:.1f} min exceeds D threshold ({MAX_FLATLINE_DURATION_FOR_D} min)")
    
    # Final grade is the worst (lowest) grade
    grade_order = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
    final_grade = min(grades, key=lambda g: grade_order[g])
    
    # Build explanation
    if final_grade == "A":
        explanation = "Excellent sensor health. All metrics are within acceptable ranges."
    else:
        explanation = f"Grade {final_grade} assigned because: " + "; ".join(reasons[:3])  # Limit to 3 reasons
    
    return final_grade, explanation


def _empty_health_report() -> Dict[str, Any]:
    """Return empty health report structure."""
    return {
        "total_readings": 0,
        "num_anomalies": 0,
        "num_flatline_anomalies": 0,
        "num_roc_anomalies": 0,
        "num_compression_anomalies": 0,
        "num_gaps_filled": 0,
        "total_gap_minutes": 0.0,
        "flatline_duration_minutes": 0.0,
        "avg_trust_score": 0.0,
        "min_trust_score": 0.0,
        "max_trust_score": 0.0,
        "anomaly_ratio": 0.0,
        "anomaly_ratio_percent": 0.0,
        "health_grade": "N/A",
        "grade_explanation": "No data available",
    }

