"""
Stream Normalizer

Normalizes raw CGM data streams by:
- Sorting readings by timestamp
- Removing exact duplicates
- Fixing minor timestamp ordering issues
- Optionally resampling to fixed intervals

This is the first step in the reliability pipeline, ensuring data
is in a consistent, ordered format for downstream processing.
"""

from datetime import datetime, timedelta
from typing import List

from .models import RawReading


def normalize_stream(readings: List[RawReading]) -> List[RawReading]:
    """
    Normalize a stream of raw CGM readings.
    
    Steps:
    1. Sort by timestamp (ascending)
    2. Remove exact duplicates (same timestamp + same glucose value)
    3. Fix minor timestamp ordering issues (small out-of-order corrections)
    4. Ensure monotonic timestamps
    
    Args:
        readings: List of raw readings to normalize
        
    Returns:
        Normalized list of readings, sorted by timestamp
    """
    if not readings:
        return []
    
    # Step 1: Sort by timestamp
    sorted_readings = sorted(readings, key=lambda r: r.timestamp)
    
    # Step 2: Remove exact duplicates (same timestamp + same glucose)
    seen = set()
    deduplicated = []
    for reading in sorted_readings:
        key = (reading.timestamp, reading.glucose)
        if key not in seen:
            seen.add(key)
            deduplicated.append(reading)
    
    # Step 3: Fix minor timestamp ordering issues
    # If a reading is out of order by less than 5 minutes, we assume
    # it's a minor clock drift and keep it, but ensure overall monotonicity
    fixed_readings = []
    for i, reading in enumerate(deduplicated):
        if i == 0:
            fixed_readings.append(reading)
            continue
        
        prev_timestamp = fixed_readings[-1].timestamp
        
        # If timestamp is before previous, but within 5 minutes, adjust it
        # to be 1 second after previous (handles minor clock issues)
        if reading.timestamp < prev_timestamp:
            time_diff = (prev_timestamp - reading.timestamp).total_seconds()
            if time_diff < 300:  # Less than 5 minutes
                # Adjust to 1 second after previous
                reading.timestamp = prev_timestamp + timedelta(seconds=1)
            # If more than 5 minutes out of order, we keep it as-is
            # (might be a real data issue that anomaly detector will catch)
        
        fixed_readings.append(reading)
    
    return fixed_readings

