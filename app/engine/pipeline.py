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
from datetime import datetime, date, time
from calendar import monthrange
import statistics
import re

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


def format_datetime_for_summary(dt: datetime) -> str:
    """
    Format datetime for summary text - always includes date and time.
    
    Args:
        dt: Datetime to format
        
    Returns:
        Formatted string: "1:00 PM on Jan 5, 2025"
    """
    hour = dt.hour
    minute = dt.minute
    
    # Format time in 12-hour format
    if hour == 0:
        time_str = "12:00 AM" if minute == 0 else f"12:{minute:02d} AM"
    elif hour == 12:
        time_str = "12:00 PM" if minute == 0 else f"12:{minute:02d} PM"
    elif hour < 12:
        time_str = f"{hour}:{minute:02d} AM"
    else:
        time_str = f"{hour - 12}:{minute:02d} PM"
    
    # Format date without leading zero on day
    month_str = dt.strftime("%b")
    day = dt.day
    year = dt.year
    date_str = f"{month_str} {day}, {year}"
    
    return f"{time_str} on {date_str}"


def format_period_description(
    view_scope: str,
    first_ts: Optional[datetime],
    last_ts: Optional[datetime],
    view_day: Optional[str] = None,
    view_start: Optional[str] = None,
    view_end: Optional[str] = None,
    view_month: Optional[str] = None
) -> str:
    """
    Format a human-readable description of the analysis period.
    
    Args:
        view_scope: One of "full", "day", "range", "month"
        first_ts: First timestamp in the filtered data
        last_ts: Last timestamp in the filtered data
        view_day: Date string (YYYY-MM-DD) when view_scope == "day"
        view_start: Start date string when view_scope == "range"
        view_end: End date string when view_scope == "range"
        view_month: Month string (YYYY-MM) when view_scope == "month"
        
    Returns:
        Formatted string like "on Jan 5, 2025" or "from Jan 5, 2025 to Jan 12, 2025"
    """
    if view_scope == "day" and view_day:
        try:
            target_date = datetime.strptime(view_day, "%Y-%m-%d").date()
            month_str = target_date.strftime("%b")
            return f"on {month_str} {target_date.day}, {target_date.year}"
        except (ValueError, AttributeError):
            pass
    
    elif view_scope == "range" and view_start and view_end:
        try:
            start_date = datetime.strptime(view_start, "%Y-%m-%d").date()
            end_date = datetime.strptime(view_end, "%Y-%m-%d").date()
            start_str = f"{start_date.strftime('%b')} {start_date.day}, {start_date.year}"
            end_str = f"{end_date.strftime('%b')} {end_date.day}, {end_date.year}"
            return f"from {start_str} to {end_str}"
        except (ValueError, AttributeError):
            pass
    
    elif view_scope == "month" and view_month:
        try:
            year, month = map(int, view_month.split("-"))
            month_name = datetime(year, month, 1).strftime("%B")
            return f"in {month_name} {year}"
        except (ValueError, AttributeError):
            pass
    
    # Default: use actual data range
    if first_ts and last_ts:
        first_date = first_ts.date()
        last_date = last_ts.date()
        
        if first_date == last_date:
            # Single day
            month_str = first_date.strftime("%b")
            return f"on {month_str} {first_date.day}, {first_date.year}"
        else:
            # Date range
            start_str = f"{first_date.strftime('%b')} {first_date.day}, {first_date.year}"
            end_str = f"{last_date.strftime('%b')} {last_date.day}, {last_date.year}"
            return f"from {start_str} to {end_str}"
    
    return "in the selected period"


def _parse_date_string(date_str: str) -> date:
    """
    Robustly parse a date string in various formats.
    
    Supports:
    - ISO format: "2025-01-03" (YYYY-MM-DD)
    - US format: "01/03/2025" (MM/DD/YYYY)
    - European format: "03/01/2025" (DD/MM/YYYY)
    - Alternative separators: dots, dashes
    
    Args:
        date_str: Date string in any supported format
        
    Returns:
        date object
        
    Raises:
        ValueError: If date string cannot be parsed
    """
    if not date_str or not date_str.strip():
        raise ValueError("Empty date string")
    
    date_str = date_str.strip()
    
    # Try ISO format first (YYYY-MM-DD) - most common from HTML date inputs
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        pass
    
    # Try US format (MM/DD/YYYY or MM-DD-YYYY)
    for sep in ['/', '-', '.']:
        try:
            parts = date_str.split(sep)
            if len(parts) == 3:
                # Try MM/DD/YYYY first (US format)
                month, day, year = map(int, parts)
                if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                    return date(year, month, day)
        except (ValueError, IndexError):
            continue
    
    # Try European format (DD/MM/YYYY or DD-MM-YYYY)
    for sep in ['/', '-', '.']:
        try:
            parts = date_str.split(sep)
            if len(parts) == 3:
                # Try DD/MM/YYYY (European format)
                day, month, year = map(int, parts)
                if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                    return date(year, month, day)
        except (ValueError, IndexError):
            continue
    
    # If all else fails, try pandas to_datetime (very flexible)
    try:
        import pandas as pd
        dt = pd.to_datetime(date_str, errors='raise')
        if isinstance(dt, pd.Timestamp):
            return dt.date()
        elif isinstance(dt, datetime):
            return dt.date()
    except (ImportError, ValueError, TypeError):
        pass
    
    raise ValueError(f"Could not parse date string: {date_str}")


def parse_date_input(value: Optional[str]) -> Optional[date]:
    """
    Parse a date string from the HTML form into a date object.
    
    Accepts multiple common formats:
    - 'YYYY-MM-DD' (ISO, what <input type="date"> sends)
    - 'MM/DD/YYYY'
    - 'DD/MM/YYYY'
    
    Returns None if parsing fails.
    """
    if not value:
        return None
    
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    
    # Debug while developing
    print("parse_date_input: could not parse date string:", repr(value))
    return None


def filter_by_view_scope(
    readings: List[ProcessedReading],
    view_scope: str,
    view_day: Optional[str] = None,
    view_start: Optional[str] = None,
    view_end: Optional[str] = None,
    view_month: Optional[str] = None
) -> List[ProcessedReading]:
    """
    Filter readings based on view scope using explicit datetime ranges.
    
    Args:
        readings: List of processed readings
        view_scope: One of "full", "whole_file", "day", "range", "month"
        view_day: Date string (YYYY-MM-DD or other formats) when view_scope == "day"
        view_start: Start date string (YYYY-MM-DD or other formats) when view_scope == "range"
        view_end: End date string (YYYY-MM-DD or other formats) when view_scope == "range"
        view_month: Month string (YYYY-MM) when view_scope == "month"
        
    Returns:
        Filtered list of readings
    """
    # Debug
    print(
        "filter_by_view_scope called with:",
        "view_scope =", view_scope,
        "view_day =", view_day,
        "view_start =", view_start,
        "view_end =", view_end,
        "view_month =", view_month,
    )
    
    if not readings:
        print("filter_by_view_scope: no readings provided, returning empty list")
        return readings
    
    # --- WHOLE FILE / FULL ---------------------------------------------------
    if view_scope in ("full", "whole_file"):
        # Do not filter at all
        print(f"filter_by_view_scope[{view_scope}]: using all rows: {len(readings)}")
        return readings
    
    # --- SINGLE DAY ---------------------------------------------------------
    if view_scope == "day" and view_day:
        target = parse_date_input(view_day)
        if target is None:
            print("filter_by_view_scope[day]: invalid view_day, returning unfiltered readings")
            return readings
        
        start_dt = datetime.combine(target, time.min)
        end_dt = datetime.combine(target, time.max)
        
        filtered = [
            r for r in readings
            if start_dt <= r.timestamp <= end_dt
        ]
        
        print(
            f"filter_by_view_scope[day]: rows before = {len(readings)}, "
            f"rows after = {len(filtered)}"
        )
        return filtered
    
    # --- DATE RANGE ---------------------------------------------------------
    if view_scope == "range" and view_start and view_end:
        start_date = parse_date_input(view_start)
        end_date = parse_date_input(view_end)
        if start_date is None or end_date is None:
            print("filter_by_view_scope[range]: invalid range, returning unfiltered readings")
            return readings
        
        start_dt = datetime.combine(start_date, time.min)
        end_dt = datetime.combine(end_date, time.max)
        
        filtered = [
            r for r in readings
            if start_dt <= r.timestamp <= end_dt
        ]
        
        print(
            f"filter_by_view_scope[range]: rows before = {len(readings)}, "
            f"rows after = {len(filtered)}"
        )
        return filtered
    
    # --- MONTH --------------------------------------------------------------
    if view_scope == "month" and view_month:
        try:
            year, month = map(int, view_month.split("-"))
            start_dt = datetime(year, month, 1)
            last_day = monthrange(year, month)[1]
            end_dt = datetime(year, month, last_day, 23, 59, 59)
            
            filtered = [
                r for r in readings
                if start_dt <= r.timestamp <= end_dt
            ]
            
            print(
                f"filter_by_view_scope[month]: rows before = {len(readings)}, "
                f"rows after = {len(filtered)}"
            )
            return filtered
        except Exception as e:
            print(f"filter_by_view_scope[month]: failed for {view_month} -> {e}")
            return readings
    
    # If the view_scope value is unknown, do not filter.
    print(f"filter_by_view_scope: unknown view_scope '{view_scope}', returning unfiltered readings")
    return readings


def run_pipeline(
    raw_readings: List[RawReading],
    max_delta_mgdl: float = DEFAULT_MAX_DELTA_MGDL,
    max_delta_minutes: int = DEFAULT_MAX_DELTA_MINUTES,
    expected_interval_minutes: int = DEFAULT_EXPECTED_INTERVAL_MINUTES,
    gap_threshold_minutes: int = DEFAULT_GAP_THRESHOLD_MINUTES,
    trust_correction_threshold: float = DEFAULT_TRUST_CORRECTION_THRESHOLD,
    glucose_targets: Optional[Dict[str, float]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    view_scope: str = "full",
    view_day: Optional[str] = None,
    view_start: Optional[str] = None,
    view_end: Optional[str] = None,
    view_month: Optional[str] = None
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
    
    # Step 5.5: Filter by view scope if specified
    filtered_processed = filter_by_view_scope(
        processed,
        view_scope=view_scope,
        view_day=view_day,
        view_start=view_start,
        view_end=view_end,
        view_month=view_month
    )
    
    if not filtered_processed:
        # Return a special structure indicating no data
        return [], {
            "has_data": False,
            "error": "NO_DATA_IN_RANGE",
            "basic_metrics": _empty_metrics(),
            "sensor_health": {},
            "bolus_risks": {},
            "failure_prediction": {},
            "insights": [],
            "summary": {
                "period_description": "No data in selected window",
                "highest_glucose_text": "",
                "lowest_glucose_text": "",
            },
        }
    
    # Step 6: Compute summary metrics (using filtered data)
    basic_metrics = _compute_metrics(filtered_processed, raw_readings)
    
    # Step 7: Compute sensor health report (using filtered data)
    sensor_health = compute_sensor_health_report(filtered_processed)
    
    # Step 8: Analyze bolus risks (using filtered data)
    bolus_risks = analyze_bolus_risks(filtered_processed, glucose_targets=glucose_targets)
    
    # Step 9: Predict failure (only if we have enough data, using filtered data)
    if len(filtered_processed) >= 5:
        failure_prediction = predict_failure(filtered_processed)
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
    
    # Step 10: Generate patient-friendly insights (using filtered data)
    insights = generate_glucose_insights(filtered_processed, glucose_targets=glucose_targets)
    
    # Step 11: Find highest and lowest glucose values for summary text
    # (filtered_processed is guaranteed to be non-empty at this point)
    glucose_values = [r.corrected_glucose for r in filtered_processed]
    max_glucose = max(glucose_values)
    min_glucose = min(glucose_values)
    max_idx = glucose_values.index(max_glucose)
    min_idx = glucose_values.index(min_glucose)
    max_dt = filtered_processed[max_idx].timestamp
    min_dt = filtered_processed[min_idx].timestamp
    
    highest_glucose_text = f"Your highest glucose was {max_glucose:.0f} mg/dL at {format_datetime_for_summary(max_dt)}."
    lowest_glucose_text = f"Your lowest was {min_glucose:.0f} mg/dL at {format_datetime_for_summary(min_dt)}."
    
    # Compute period description from actual filtered data range
    first_ts_filtered = filtered_processed[0].timestamp
    last_ts_filtered = filtered_processed[-1].timestamp
    # Use actual filtered data range for period description
    period_description = f"from {first_ts_filtered.strftime('%b %d, %Y')} to {last_ts_filtered.strftime('%b %d, %Y')}"
    
    return filtered_processed, {
        "has_data": True,
        "basic_metrics": basic_metrics,
        "summary": {
            "period_description": period_description,
            "highest_glucose_text": highest_glucose_text,
            "lowest_glucose_text": lowest_glucose_text,
        },
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

