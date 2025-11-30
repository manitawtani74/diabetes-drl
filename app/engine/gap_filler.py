"""
Gap Filler

Identifies gaps in CGM data streams and fills them with interpolated readings.

Gaps are defined as time intervals between consecutive readings that exceed
the expected interval. Interpolated readings are marked with is_filled=True
and use linear interpolation between neighboring readings.
"""

from datetime import datetime, timedelta
from typing import List

from .models import ProcessedReading


def fill_gaps(
    processed: List[ProcessedReading],
    expected_interval_minutes: int = 5,
    gap_threshold_minutes: int = 10
) -> List[ProcessedReading]:
    """
    Fill gaps in the processed readings with interpolated values.
    
    A gap is identified when the time difference between consecutive
    readings exceeds the gap threshold. The gap is filled with linearly
    interpolated readings at the expected interval.
    
    Args:
        processed: List of processed readings (may already have anomalies flagged)
        expected_interval_minutes: Expected time between readings (minutes)
        gap_threshold_minutes: Minimum gap size to trigger filling (minutes)
        
    Returns:
        List of processed readings with gaps filled (new readings have is_filled=True)
    """
    if len(processed) < 2:
        return processed
    
    filled_readings = []
    expected_interval = timedelta(minutes=expected_interval_minutes)
    gap_threshold = timedelta(minutes=gap_threshold_minutes)
    
    for i in range(len(processed)):
        # Always add the current reading
        filled_readings.append(processed[i])
        
        # Check if there's a gap before the next reading
        if i < len(processed) - 1:
            current = processed[i]
            next_reading = processed[i + 1]
            
            time_diff = next_reading.timestamp - current.timestamp
            
            if time_diff > gap_threshold:
                # Fill the gap with interpolated readings
                interpolated = _interpolate_gap(
                    current,
                    next_reading,
                    expected_interval
                )
                filled_readings.extend(interpolated)
    
    return filled_readings


def _interpolate_gap(
    start: ProcessedReading,
    end: ProcessedReading,
    interval: timedelta
) -> List[ProcessedReading]:
    """
    Generate interpolated readings between two readings.
    
    Uses linear interpolation to fill the gap at regular intervals.
    
    Args:
        start: Reading at the start of the gap
        end: Reading at the end of the gap
        interval: Time interval between interpolated readings
        
    Returns:
        List of interpolated ProcessedReading objects with is_filled=True
    """
    interpolated = []
    
    # Calculate number of intervals needed
    time_diff = end.timestamp - start.timestamp
    num_intervals = int(time_diff / interval)
    
    # If gap is exactly one interval or less, no interpolation needed
    if num_intervals < 2:
        return []
    
    # Calculate glucose delta per interval
    glucose_delta = end.raw_glucose - start.raw_glucose
    glucose_per_interval = glucose_delta / num_intervals
    
    # Generate interpolated readings
    for i in range(1, num_intervals):
        timestamp = start.timestamp + (interval * i)
        glucose = start.raw_glucose + (glucose_per_interval * i)
        
        interpolated_reading = ProcessedReading(
            timestamp=timestamp,
            raw_glucose=glucose,
            corrected_glucose=glucose,
            is_filled=True,
            trust_score=0.5  # Initial trust score for filled readings (will be adjusted)
        )
        interpolated.append(interpolated_reading)
    
    return interpolated

