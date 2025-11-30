"""
Trust Score Engine

Computes reliability scores (0.0 to 1.0) for each processed reading.

Trust scores are based on:
- Anomaly flags (rate-of-change, flatline, compression)
- Whether reading was interpolated (filled)
- Proximity to gaps
- Local stability (variance in neighborhood)

Higher scores indicate more reliable readings.
"""

from typing import List
import statistics

from .models import ProcessedReading


# Trust score penalties
PENALTY_ROC_ANOMALY = 0.4
PENALTY_FLATLINE = 0.3
PENALTY_COMPRESSION_LOW = 0.3
PENALTY_FILLED = 0.2
PENALTY_NEAR_GAP = 0.1


def score_trust(processed: List[ProcessedReading]) -> List[ProcessedReading]:
    """
    Compute trust scores for all processed readings.
    
    Trust score starts at 1.0 and penalties are subtracted for:
    - Rate-of-change anomalies
    - Flatline regions
    - Compression lows
    - Filled (interpolated) readings
    - Readings near gaps
    
    Final score is clamped between 0.0 and 1.0.
    
    Args:
        processed: List of processed readings (may have anomalies and filled flags)
        
    Returns:
        Same list with trust_score field populated
    """
    if not processed:
        return processed
    
    # First pass: compute base trust scores
    for reading in processed:
        reading.trust_score = _compute_base_trust_score(reading)
    
    # Second pass: adjust for local stability
    _adjust_for_local_stability(processed)
    
    # Ensure scores are clamped
    for reading in processed:
        reading.trust_score = max(0.0, min(1.0, reading.trust_score))
    
    return processed


def _compute_base_trust_score(reading: ProcessedReading) -> float:
    """
    Compute base trust score for a single reading.
    
    Starts at 1.0 and subtracts penalties based on flags.
    """
    score = 1.0
    
    # Check anomaly reasons
    if reading.is_anomaly:
        if any("roc_exceeded" in reason for reason in reading.anomaly_reasons):
            score -= PENALTY_ROC_ANOMALY
        if any("flatline" in reason for reason in reading.anomaly_reasons):
            score -= PENALTY_FLATLINE
        if any("compression_low" in reason for reason in reading.anomaly_reasons):
            score -= PENALTY_COMPRESSION_LOW
    
    # Penalty for filled readings
    if reading.is_filled:
        score -= PENALTY_FILLED
    
    return score


def _adjust_for_local_stability(processed: List[ProcessedReading]) -> None:
    """
    Adjust trust scores based on local stability (variance in neighborhood).
    
    Readings in stable regions (low variance) get a small boost.
    Readings in unstable regions (high variance) get a small penalty.
    """
    if len(processed) < 3:
        return
    
    # Window size for local stability calculation
    window_size = 3  # Look at 3 readings before and after
    
    for i in range(len(processed)):
        # Collect neighboring readings
        neighbors = []
        start_idx = max(0, i - window_size)
        end_idx = min(len(processed), i + window_size + 1)
        
        for j in range(start_idx, end_idx):
            if j != i:  # Exclude current reading
                neighbors.append(processed[j].raw_glucose)
        
        if len(neighbors) < 2:
            continue
        
        # Calculate local variance
        try:
            if len(neighbors) > 1:
                try:
                    variance = statistics.variance(neighbors)
                except (statistics.StatisticsError, ValueError):
                    variance = 0.0
            else:
                variance = 0.0
            std_dev = variance ** 0.5
            
            # Normalize: if std_dev > 20 mg/dL, it's quite unstable
            # If std_dev < 5 mg/dL, it's quite stable
            if std_dev > 20:
                # Unstable region: small penalty
                processed[i].trust_score -= 0.05
            elif std_dev < 5 and not processed[i].is_anomaly:
                # Stable region: small boost (only if not already anomalous)
                processed[i].trust_score += 0.05
        except statistics.StatisticsError:
            # Not enough data for variance calculation
            pass

