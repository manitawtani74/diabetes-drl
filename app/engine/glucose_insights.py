"""
Glucose Insights Generator

Generates patient-friendly, easy-to-understand insights from CGM data.
All insights describe patterns only - no medical advice or dosing suggestions.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import statistics

from .models import ProcessedReading


@dataclass
class Insight:
    """
    Represents a single patient-friendly insight.
    
    Attributes:
        title: Short, clear title for the insight
        severity: "info", "warning", or "critical"
        message: Patient-friendly explanation (1-3 sentences)
        details: Additional data for reference
    """
    title: str
    severity: str  # "info", "warning", or "critical"
    message: str
    details: Dict[str, Any]


def generate_glucose_insights(
    readings: List[ProcessedReading],
    glucose_targets: Optional[Dict[str, float]] = None
) -> List[Insight]:
    """
    Generate patient-friendly insights from processed CGM readings.
    
    All insights describe patterns only - no medical advice.
    
    Args:
        readings: List of processed readings
        glucose_targets: Dictionary with custom glucose thresholds (defaults used if None)
        
    Returns:
        List of Insight objects
    """
    # Set default targets if not provided
    if glucose_targets is None:
        glucose_targets = {
            "time_in_range_low": 70.0,
            "time_in_range_high": 180.0,
            "low_threshold": 70.0,
            "very_low_threshold": 54.0,
            "high_threshold": 180.0,
            "very_high_threshold": 250.0,
        }
    
    if not readings:
        return []
    
    insights = []
    
    # A) Glucose Range Summary
    range_insight = _generate_range_summary(readings, glucose_targets)
    if range_insight:
        insights.append(range_insight)
    
    # B) Rapid Glucose Spikes
    spikes_insight = _detect_rapid_spikes(readings, glucose_targets)
    if spikes_insight:
        insights.append(spikes_insight)
    
    # C) Long High Glucose Periods
    high_periods_insight = _detect_long_high_periods(readings, glucose_targets)
    if high_periods_insight:
        insights.append(high_periods_insight)
    
    # D) Low Glucose Events
    low_events_insight = _detect_low_glucose_events(readings, glucose_targets)
    if low_events_insight:
        insights.append(low_events_insight)
    
    # E) Sensor Reliability Context
    reliability_insight = _generate_reliability_context(readings)
    if reliability_insight:
        insights.append(reliability_insight)
    
    return insights


def _generate_range_summary(
    readings: List[ProcessedReading],
    targets: Dict[str, float]
) -> Optional[Insight]:
    """Generate insight about time in range, highs, and lows."""
    if not readings:
        return None
    
    # Extract thresholds from targets
    tir_low = targets["time_in_range_low"]
    tir_high = targets["time_in_range_high"]
    low_thresh = targets["low_threshold"]
    very_low_thresh = targets["very_low_threshold"]
    high_thresh = targets["high_threshold"]
    very_high_thresh = targets["very_high_threshold"]
    
    # Use corrected glucose values
    glucose_values = [r.corrected_glucose for r in readings]
    timestamps = [r.timestamp for r in readings]
    
    # Count readings in each range
    in_range = sum(1 for g in glucose_values if tir_low <= g <= tir_high)
    above_high = sum(1 for g in glucose_values if g > high_thresh)
    above_very_high = sum(1 for g in glucose_values if g > very_high_thresh)
    below_low = sum(1 for g in glucose_values if g < low_thresh)
    below_very_low = sum(1 for g in glucose_values if g < very_low_thresh)
    
    total = len(readings)
    
    # Calculate percentages
    pct_in_range = (in_range / total * 100) if total > 0 else 0
    pct_above_high = (above_high / total * 100) if total > 0 else 0
    pct_above_very_high = (above_very_high / total * 100) if total > 0 else 0
    pct_below_low = (below_low / total * 100) if total > 0 else 0
    pct_below_very_low = (below_very_low / total * 100) if total > 0 else 0
    
    # Find highest and lowest values
    max_glucose = max(glucose_values)
    min_glucose = min(glucose_values)
    max_idx = glucose_values.index(max_glucose)
    min_idx = glucose_values.index(min_glucose)
    max_time = timestamps[max_idx].strftime("%H:%M")
    min_time = timestamps[min_idx].strftime("%H:%M")
    
    # Build message with bullet points
    message_lines = ["Over this period:"]
    
    message_lines.append(f"• Time in range ({tir_low:.0f}–{tir_high:.0f} mg/dL): {pct_in_range:.0f}%")
    
    if pct_above_high > 0:
        message_lines.append(f"• Above range (>{high_thresh:.0f} mg/dL): {pct_above_high:.0f}%")
    if pct_above_very_high > 0:
        message_lines.append(f"• Very high (>{very_high_thresh:.0f} mg/dL): {pct_above_very_high:.0f}%")
    if pct_below_low > 0:
        message_lines.append(f"• Below range (<{low_thresh:.0f} mg/dL): {pct_below_low:.0f}%")
    if pct_below_very_low > 0:
        message_lines.append(f"• Very low (<{very_low_thresh:.0f} mg/dL): {pct_below_very_low:.0f}%")
    
    message_lines.append("")
    message_lines.append(
        f"Your highest glucose was {max_glucose:.0f} mg/dL at {max_time}, "
        f"and your lowest was {min_glucose:.0f} mg/dL at {min_time}."
    )
    
    message = "\n".join(message_lines)
    
    # Determine severity
    severity = "info"
    if pct_above_very_high > 10 or pct_below_very_low > 5:
        severity = "warning"
    elif pct_below_very_low > 0:
        severity = "warning"
    
    return Insight(
        title="Glucose Range Summary",
        severity=severity,
        message=message,
        details={
            "pct_in_range": round(pct_in_range, 1),
            "pct_above_high": round(pct_above_high, 1),
            "pct_above_very_high": round(pct_above_very_high, 1),
            "pct_below_low": round(pct_below_low, 1),
            "pct_below_very_low": round(pct_below_very_low, 1),
            "max_glucose": round(max_glucose, 1),
            "min_glucose": round(min_glucose, 1),
            "max_time": max_time,
            "min_time": min_time,
        }
    )


def _detect_rapid_spikes(
    readings: List[ProcessedReading],
    targets: Dict[str, float]
) -> Optional[Insight]:
    """
    Detect rapid glucose spikes (rise ≥ 60 mg/dL within 30-45 minutes).
    """
    if len(readings) < 3:
        return None
    
    high_thresh = targets["high_threshold"]
    
    spikes = []
    spike_window_minutes = 45
    
    i = 0
    while i < len(readings) - 1:
        start_reading = readings[i]
        start_glucose = start_reading.corrected_glucose
        
        # Look ahead for rapid rise
        for j in range(i + 1, len(readings)):
            current_reading = readings[j]
            time_diff = (current_reading.timestamp - start_reading.timestamp).total_seconds() / 60.0
            
            # Only check within the time window
            if time_diff > spike_window_minutes:
                break
            
            current_glucose = current_reading.corrected_glucose
            rise = current_glucose - start_glucose
            
            # Check if this is a spike (rise ≥ 60 mg/dL and peaks above high threshold)
            if rise >= 60 and current_glucose > high_thresh:
                # Find the peak
                peak_glucose = current_glucose
                peak_idx = j
                peak_time = current_reading.timestamp
                
                # Look ahead to find when it falls below high threshold
                fall_below_high = None
                for k in range(j + 1, len(readings)):
                    if readings[k].corrected_glucose < high_thresh:
                        fall_below_high = (readings[k].timestamp - peak_time).total_seconds() / 60.0
                        break
                
                spikes.append({
                    "start_time": start_reading.timestamp,
                    "peak_glucose": peak_glucose,
                    "peak_time": peak_time,
                    "rise_mgdl": rise,
                    "time_to_peak_minutes": time_diff,
                    "time_to_fall_minutes": fall_below_high,
                })
                
                # Skip ahead to avoid double-counting
                i = j
                break
        
        i += 1
    
    if not spikes:
        return None
    
    # Calculate statistics
    if not spikes:
        avg_peak = 0.0
        avg_fall_time = None
    else:
        peak_values = [s["peak_glucose"] for s in spikes]
        avg_peak = statistics.mean(peak_values) if peak_values else 0.0
        
        fall_times = [s["time_to_fall_minutes"] for s in spikes if s["time_to_fall_minutes"] is not None]
        avg_fall_time = statistics.mean(fall_times) if fall_times else None
    
    # Find typical time of day
    peak_hours = [s["peak_time"].hour for s in spikes]
    if peak_hours:
        most_common_hour = statistics.mode(peak_hours) if len(set(peak_hours)) < len(peak_hours) else peak_hours[0]
        typical_time = f"{most_common_hour}:00"
    else:
        typical_time = "various times"
    
    # Build message
    spike_count = len(spikes)
    message = (
        f"We detected {spike_count} rapid glucose spike{'s' if spike_count > 1 else ''} "
        f"(rise of 60+ mg/dL within 30–45 minutes). "
        f"{'They' if spike_count > 1 else 'It'} typically peaked around {avg_peak:.0f} mg/dL"
    )
    
    if avg_fall_time:
        hours = avg_fall_time / 60.0
        if hours >= 1:
            message += f" and took about {hours:.1f} hour{'s' if hours >= 2 else ''} to return below {high_thresh:.0f} mg/dL."
        else:
            message += f" and took about {avg_fall_time:.0f} minutes to return below {high_thresh:.0f} mg/dL."
    else:
        message += "."
    
    # Determine severity
    severity = "warning" if spike_count >= 3 else "info"
    
    return Insight(
        title="Rapid Glucose Spikes",
        severity=severity,
        message=message,
        details={
            "spike_count": spike_count,
            "avg_peak": round(avg_peak, 1),
            "avg_fall_time_minutes": round(avg_fall_time, 1) if avg_fall_time else None,
            "typical_time": typical_time,
        }
    )


def _detect_long_high_periods(
    readings: List[ProcessedReading],
    targets: Dict[str, float]
) -> Optional[Insight]:
    """
    Detect periods where glucose stays above high threshold for more than 2 hours.
    """
    if not readings:
        return None
    
    high_thresh = targets["high_threshold"]
    
    high_periods = []
    in_high_period = False
    period_start = None
    period_start_idx = None
    
    for i, reading in enumerate(readings):
        glucose = reading.corrected_glucose
        
        if glucose > high_thresh:
            if not in_high_period:
                # Start of high period
                in_high_period = True
                period_start = reading.timestamp
                period_start_idx = i
        else:
            if in_high_period:
                # End of high period
                period_end = readings[i - 1].timestamp
                duration_minutes = (period_end - period_start).total_seconds() / 60.0
                
                if duration_minutes > 120:  # More than 2 hours
                    high_periods.append({
                        "start": period_start,
                        "end": period_end,
                        "duration_minutes": duration_minutes,
                    })
                
                in_high_period = False
    
    # Handle case where high period extends to end
    if in_high_period and period_start:
        period_end = readings[-1].timestamp
        duration_minutes = (period_end - period_start).total_seconds() / 60.0
        if duration_minutes > 120:
            high_periods.append({
                "start": period_start,
                "end": period_end,
                "duration_minutes": duration_minutes,
            })
    
    if not high_periods:
        return None
    
    # Find longest period
    longest = max(high_periods, key=lambda p: p["duration_minutes"])
    longest_hours = longest["duration_minutes"] / 60.0
    
    # Determine time of day for longest period
    longest_hour = longest["start"].hour
    if 6 <= longest_hour < 12:
        time_of_day = "morning"
    elif 12 <= longest_hour < 18:
        time_of_day = "afternoon"
    elif 18 <= longest_hour < 22:
        time_of_day = "evening"
    else:
        time_of_day = "night"
    
    period_count = len(high_periods)
    message = (
        f"Glucose stayed above {high_thresh:.0f} mg/dL for more than 2 hours on {period_count} "
        f"occasion{'s' if period_count > 1 else ''}. "
        f"The longest period lasted {longest_hours:.1f} hours and occurred in the {time_of_day}."
    )
    
    return Insight(
        title="Long High Glucose Periods",
        severity="warning",
        message=message,
        details={
            "period_count": period_count,
            "longest_duration_hours": round(longest_hours, 1),
            "longest_start": longest["start"].isoformat(),
        }
    )


def _detect_low_glucose_events(
    readings: List[ProcessedReading],
    targets: Dict[str, float]
) -> Optional[Insight]:
    """
    Detect low glucose events and check for night-time clustering.
    """
    if not readings:
        return None
    
    low_thresh = targets["low_threshold"]
    very_low_thresh = targets["very_low_threshold"]
    
    lows_below_low = []
    lows_below_very_low = []
    
    for reading in readings:
        glucose = reading.corrected_glucose
        hour = reading.timestamp.hour
        
        if glucose < very_low_thresh:
            lows_below_very_low.append({
                "glucose": glucose,
                "timestamp": reading.timestamp,
                "hour": hour,
            })
        elif glucose < low_thresh:
            lows_below_low.append({
                "glucose": glucose,
                "timestamp": reading.timestamp,
                "hour": hour,
            })
    
    if not lows_below_low and not lows_below_very_low:
        return None
    
    # Check for night-time clustering (00:00-06:00)
    night_lows = [
        low for low in (lows_below_low + lows_below_very_low)
        if 0 <= low["hour"] < 6
    ]
    
    # Build message
    if lows_below_very_low:
        count = len(lows_below_very_low)
        message = (
            f"You had {count} reading{'s' if count > 1 else ''} below {very_low_thresh:.0f} mg/dL "
            f"(very low glucose)."
        )
        if night_lows:
            night_count = len([l for l in lows_below_very_low if 0 <= l["hour"] < 6])
            if night_count > 0:
                message += f" {night_count} of these occurred between 2am–4am."
        message += " Night-time lows can be important to review with your diabetes care team."
        severity = "critical"
    elif lows_below_low:
        count = len(lows_below_low)
        message = (
            f"You had {count} reading{'s' if count > 1 else ''} below {low_thresh:.0f} mg/dL "
            f"(low glucose)."
        )
        if night_lows:
            night_count = len([l for l in lows_below_low if 0 <= l["hour"] < 6])
            if night_count > 0:
                message += f" {night_count} of these occurred between 2am–4am."
        message += " These patterns may be worth discussing with your diabetes care team."
        severity = "warning"
    else:
        message = "No low glucose events detected."
        severity = "info"
    
    return Insight(
        title="Low Glucose Events",
        severity=severity,
        message=message,
        details={
            "count_below_low": len(lows_below_low),
            "count_below_very_low": len(lows_below_very_low),
            "night_time_count": len(night_lows),
        }
    )


def _generate_reliability_context(readings: List[ProcessedReading]) -> Optional[Insight]:
    """
    Provide context about sensor reliability/trust scores.
    """
    if not readings:
        return None
    
    trust_scores = [r.trust_score for r in readings if r.trust_score is not None]
    if not trust_scores:
        return None
    avg_trust = statistics.mean(trust_scores)
    
    # Determine reliability description
    if avg_trust >= 0.9:
        reliability_desc = "high"
        reliability_qualifier = "very reliable"
    elif avg_trust >= 0.7:
        reliability_desc = "good"
        reliability_qualifier = "generally reliable"
    elif avg_trust >= 0.5:
        reliability_desc = "moderate"
        reliability_qualifier = "somewhat reliable"
    else:
        reliability_desc = "low"
        reliability_qualifier = "less reliable"
    
    message = (
        f"The sensor data in this report had a {reliability_desc} trust score "
        f"(average: {avg_trust:.2f}), meaning the readings were {reliability_qualifier} "
        f"according to our rule-based analysis."
    )
    
    return Insight(
        title="Sensor Reliability Context",
        severity="info",
        message=message,
        details={
            "avg_trust_score": round(avg_trust, 3),
            "reliability_level": reliability_desc,
        }
    )

