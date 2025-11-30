"""
Bolus Risk Analysis

Identifies dangerous scenarios where unreliable CGM readings could lead
to incorrect insulin bolus decisions.

This module analyzes processed readings to find:
- Low glucose readings with low trust scores (risk of over-bolusing)
- High glucose readings with low trust scores (risk of under-bolusing)
- Rapid changes in unreliable readings (could trigger incorrect corrections)
- Critical zones where sensor reliability is compromised
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .models import ProcessedReading


# Default risk thresholds (can be overridden by glucose_targets parameter)
CRITICAL_LOW_THRESHOLD_DEFAULT = 54.0   # very low
LOW_THRESHOLD_DEFAULT = 70.0           # low
HIGH_THRESHOLD_DEFAULT = 180.0        # high
VERY_HIGH_THRESHOLD_DEFAULT = 250.0   # very high

LOW_TRUST_THRESHOLD = 0.5  # Trust score below this is considered unreliable
CRITICAL_TRUST_THRESHOLD = 0.3  # Trust score below this is highly unreliable

RISK_WINDOW_MINUTES = 15  # Time window for analyzing risk patterns


def analyze_bolus_risks(
    processed_readings: List[ProcessedReading],
    glucose_targets: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Analyze bolus risks in processed CGM readings.
    
    Identifies dangerous scenarios where unreliable readings could lead
    to incorrect insulin dosing decisions.
    
    Args:
        processed_readings: List of processed readings with trust scores
        glucose_targets: Dictionary with custom glucose thresholds (defaults used if None)
        
    Returns:
        Dictionary containing:
        - high_risk_readings: List of high-risk reading indices/details
        - low_glucose_unreliable: Count of low glucose readings with low trust
        - high_glucose_unreliable: Count of high glucose readings with low trust
        - critical_low_unreliable: Count of critical low readings with low trust
        - critical_high_unreliable: Count of critical high readings with low trust
        - rapid_change_unreliable: Count of rapid changes in unreliable readings
        - risk_zones: List of time ranges with elevated risk
        - risk_summary: Text summary of risks
    """
    # Compute local thresholds using user's targets if provided, otherwise fall back to defaults
    critical_low = (glucose_targets or {}).get("very_low_threshold", CRITICAL_LOW_THRESHOLD_DEFAULT)
    low = (glucose_targets or {}).get("low_threshold", LOW_THRESHOLD_DEFAULT)
    high = (glucose_targets or {}).get("high_threshold", HIGH_THRESHOLD_DEFAULT)
    very_high = (glucose_targets or {}).get("very_high_threshold", VERY_HIGH_THRESHOLD_DEFAULT)
    
    if not processed_readings:
        return _empty_risk_analysis()
    
    high_risk_readings = []
    low_glucose_unreliable = 0
    high_glucose_unreliable = 0
    critical_low_unreliable = 0
    critical_high_unreliable = 0
    rapid_change_unreliable = 0
    
    risk_zones = []
    current_risk_zone_start = None
    
    for i, reading in enumerate(processed_readings):
        risk_level = _assess_reading_risk(reading, low, critical_low, high, very_high)
        
        if risk_level > 0:
            # Count risk types
            if reading.raw_glucose < critical_low and reading.trust_score < CRITICAL_TRUST_THRESHOLD:
                critical_low_unreliable += 1
            elif reading.raw_glucose < low and reading.trust_score < LOW_TRUST_THRESHOLD:
                low_glucose_unreliable += 1
            elif reading.raw_glucose > very_high and reading.trust_score < CRITICAL_TRUST_THRESHOLD:
                critical_high_unreliable += 1
            elif reading.raw_glucose > high and reading.trust_score < LOW_TRUST_THRESHOLD:
                high_glucose_unreliable += 1
            
            # Check for rapid changes in unreliable readings
            if i > 0:
                prev_reading = processed_readings[i - 1]
                if (reading.trust_score < LOW_TRUST_THRESHOLD and
                    prev_reading.trust_score < LOW_TRUST_THRESHOLD):
                    time_diff = (reading.timestamp - prev_reading.timestamp).total_seconds() / 60.0
                    if time_diff <= RISK_WINDOW_MINUTES:
                        glucose_change = abs(reading.raw_glucose - prev_reading.raw_glucose)
                        if glucose_change > 30:  # Significant change
                            rapid_change_unreliable += 1
            
            # Record high-risk reading
            if risk_level >= 2:  # High or critical risk
                high_risk_readings.append({
                    "index": i,
                    "timestamp": reading.timestamp.isoformat(),
                    "raw_glucose": round(reading.raw_glucose, 2),
                    "corrected_glucose": round(reading.corrected_glucose, 2),
                    "trust_score": round(reading.trust_score, 3),
                    "risk_level": risk_level,
                    "risk_reason": _get_risk_reason(
                        reading,
                        low,
                        critical_low,
                        high,
                        very_high
                    ),
                })
            
            # Track risk zones
            if current_risk_zone_start is None:
                current_risk_zone_start = reading.timestamp
        else:
            # End of risk zone
            if current_risk_zone_start is not None:
                risk_zones.append({
                    "start": current_risk_zone_start.isoformat(),
                    "end": processed_readings[i - 1].timestamp.isoformat(),
                })
                current_risk_zone_start = None
    
    # Handle risk zone extending to end
    if current_risk_zone_start is not None:
        risk_zones.append({
            "start": current_risk_zone_start.isoformat(),
            "end": processed_readings[-1].timestamp.isoformat(),
        })
    
    # Generate risk summary
    risk_summary = _generate_risk_summary(
        low_glucose_unreliable,
        high_glucose_unreliable,
        critical_low_unreliable,
        critical_high_unreliable,
        rapid_change_unreliable,
        len(high_risk_readings),
        low,
        critical_low,
        high,
        very_high
    )
    
    return {
        "high_risk_readings": high_risk_readings,
        "low_glucose_unreliable": low_glucose_unreliable,
        "high_glucose_unreliable": high_glucose_unreliable,
        "critical_low_unreliable": critical_low_unreliable,
        "critical_high_unreliable": critical_high_unreliable,
        "rapid_change_unreliable": rapid_change_unreliable,
        "risk_zones": risk_zones,
        "risk_summary": risk_summary,
        "total_high_risk": len(high_risk_readings),
    }


def _assess_reading_risk(
    reading: ProcessedReading,
    low_threshold: float,
    critical_low_threshold: float,
    high_threshold: float,
    critical_high_threshold: float
) -> int:
    """
    Assess risk level for a single reading.
    
    Returns:
        0: No risk
        1: Low risk
        2: High risk
        3: Critical risk
    """
    # Critical risk: extreme glucose values with very low trust
    if ((reading.raw_glucose < critical_low_threshold or reading.raw_glucose > critical_high_threshold) and
        reading.trust_score < CRITICAL_TRUST_THRESHOLD):
        return 3
    
    # High risk: extreme glucose values with low trust, or normal range with very low trust
    if ((reading.raw_glucose < low_threshold or reading.raw_glucose > high_threshold) and
        reading.trust_score < LOW_TRUST_THRESHOLD):
        return 2
    
    # Low risk: borderline values with low trust
    if reading.trust_score < LOW_TRUST_THRESHOLD:
        return 1
    
    return 0


def _get_risk_reason(
    reading: ProcessedReading,
    low_threshold: float,
    critical_low_threshold: float,
    high_threshold: float,
    critical_high_threshold: float
) -> str:
    """Get human-readable reason for risk."""
    reasons = []
    
    if reading.raw_glucose < critical_low_threshold:
        reasons.append(f"Critical low glucose ({reading.raw_glucose:.1f} mg/dL)")
    elif reading.raw_glucose < low_threshold:
        reasons.append(f"Low glucose ({reading.raw_glucose:.1f} mg/dL)")
    elif reading.raw_glucose > critical_high_threshold:
        reasons.append(f"Critical high glucose ({reading.raw_glucose:.1f} mg/dL)")
    elif reading.raw_glucose > high_threshold:
        reasons.append(f"High glucose ({reading.raw_glucose:.1f} mg/dL)")
    
    if reading.trust_score < CRITICAL_TRUST_THRESHOLD:
        reasons.append(f"Very low trust ({reading.trust_score:.2f})")
    elif reading.trust_score < LOW_TRUST_THRESHOLD:
        reasons.append(f"Low trust ({reading.trust_score:.2f})")
    
    if reading.is_anomaly:
        reasons.append("Anomaly detected")
    
    if reading.is_filled:
        reasons.append("Interpolated reading")
    
    return "; ".join(reasons) if reasons else "Unreliable reading"


def _generate_risk_summary(
    low_glucose_unreliable: int,
    high_glucose_unreliable: int,
    critical_low_unreliable: int,
    critical_high_unreliable: int,
    rapid_change_unreliable: int,
    total_high_risk: int,
    low_threshold: float,
    critical_low_threshold: float,
    high_threshold: float,
    critical_high_threshold: float
) -> str:
    """Generate human-readable risk summary."""
    if total_high_risk == 0:
        return "✅ No high-risk readings detected. Sensor data appears generally reliable according to our rule-based analysis."
    
    summary_parts = []
    
    if critical_low_unreliable > 0:
        summary_parts.append(
            f"⚠️ CRITICAL: {critical_low_unreliable} critical low glucose reading{'s' if critical_low_unreliable > 1 else ''} "
            f"(below {critical_low_threshold:.0f} mg/dL) with unreliable trust scores. "
            f"These readings may not be accurate and could lead to incorrect dosing decisions."
        )
    
    if critical_high_unreliable > 0:
        summary_parts.append(
            f"⚠️ CRITICAL: {critical_high_unreliable} critical high glucose reading{'s' if critical_high_unreliable > 1 else ''} "
            f"(above {critical_high_threshold:.0f} mg/dL) with unreliable trust scores. "
            f"These readings may not be accurate and could lead to incorrect dosing decisions."
        )
    
    if low_glucose_unreliable > 0:
        summary_parts.append(
            f"⚠️ WARNING: {low_glucose_unreliable} low glucose reading{'s' if low_glucose_unreliable > 1 else ''} "
            f"(below {low_threshold:.0f} mg/dL) with low trust scores. "
            f"These readings may be less reliable."
        )
    
    if high_glucose_unreliable > 0:
        summary_parts.append(
            f"⚠️ WARNING: {high_glucose_unreliable} high glucose reading{'s' if high_glucose_unreliable > 1 else ''} "
            f"(above {high_threshold:.0f} mg/dL) with low trust scores. "
            f"These readings may be less reliable."
        )
    
    if rapid_change_unreliable > 0:
        summary_parts.append(
            f"⚠️ WARNING: {rapid_change_unreliable} instance{'s' if rapid_change_unreliable > 1 else ''} of rapid glucose changes "
            f"in unreliable readings. These patterns may indicate sensor issues."
        )
    
    summary_parts.append(
        f"\n📊 Total high-risk readings: {total_high_risk}. "
        f"These patterns may be worth discussing with your diabetes care team."
    )
    
    return "\n\n".join(summary_parts)


def _empty_risk_analysis() -> Dict[str, Any]:
    """Return empty risk analysis structure."""
    return {
        "high_risk_readings": [],
        "low_glucose_unreliable": 0,
        "high_glucose_unreliable": 0,
        "critical_low_unreliable": 0,
        "critical_high_unreliable": 0,
        "rapid_change_unreliable": 0,
        "risk_zones": [],
        "risk_summary": "No data available for risk analysis.",
        "total_high_risk": 0,
    }

