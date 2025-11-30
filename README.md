# Diabetes Reliability Layer (DRL)

A fault-tolerant, safety-focused middleware layer for CGM (continuous glucose monitor) data streams. DRL processes raw CGM time series data through a reliability pipeline and outputs a corrected, annotated stream with trust scores and metrics.

## Overview

DRL applies deterministic algorithms to:
- Normalize and clean raw CGM data streams
- Detect anomalies (rate-of-change violations, flatlines, compression artifacts)
- Fill gaps with interpolated values
- Compute trust scores for each reading
- Generate corrected glucose values for unreliable readings

## Project Structure

```
drLayer/
├── app/
│   ├── main.py                    # FastAPI application
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── models.py              # Data models
│   │   ├── stream_normalizer.py   # Stream normalization
│   │   ├── anomaly_detector.py    # Anomaly detection
│   │   ├── gap_filler.py          # Gap filling
│   │   ├── trust_score.py         # Trust scoring
│   │   └── pipeline.py            # Main pipeline orchestrator
│   ├── templates/
│   │   └── upload_and_report.html # Web UI
│   └── static/                    # Static assets (if needed)
├── sample_data/
│   └── glucose_raw_example.csv    # Sample test data
├── requirements.txt
└── README.md
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

Then navigate to `http://localhost:8000` in your browser.

## Usage

1. Upload a CSV file with at least two columns:
   - `timestamp` (ISO8601 format)
   - `glucose` (mg/dL, float or int)

2. Optionally configure:
   - **ROC Threshold**: Maximum allowed glucose change (mg/dL) per time window
   - **ROC Time Window**: Time window (minutes) for ROC calculation
   - **Expected Interval**: Expected time between readings (minutes)
   - **Trust Threshold**: Trust score below which readings are corrected

3. View the results:
   - Summary metrics (anomaly count, trust scores, etc.)
   - Detailed table of all readings
   - Interactive chart comparing raw vs corrected glucose

## Configurable Thresholds

You can adjust these parameters in the web UI or by modifying the default values in `engine/pipeline.py`:

- **MAX_DELTA_MGDL** (default: 50): Maximum glucose change in mg/dL within the time window
- **MAX_DELTA_MINUTES** (default: 5): Time window for rate-of-change detection
- **FLATLINE_THRESHOLD** (default: 5): Number of consecutive identical values to trigger flatline
- **EXPECTED_INTERVAL_MINUTES** (default: 5): Expected interval between readings
- **GAP_THRESHOLD_MINUTES** (default: 10): Gap size threshold for interpolation
- **TRUST_CORRECTION_THRESHOLD** (default: 0.4): Trust score below which readings are corrected

## Sample Data

A sample CSV file is included in `sample_data/glucose_raw_example.csv` with:
- A flatline region (sensor stuck)
- A large spike (rate-of-change anomaly)
- A gap in readings (missing data)

## Architecture

The pipeline processes data in the following order:

1. **Stream Normalizer**: Sorts, deduplicates, and fixes timestamp issues
2. **Anomaly Detector**: Identifies suspicious readings based on patterns
3. **Gap Filler**: Interpolates missing readings in large gaps
4. **Trust Score Engine**: Computes reliability scores for each reading
5. **Correction Engine**: Replaces low-trust readings with interpolated values

## Methods

### Data Processing Pipeline

DRL uses a deterministic, rule-based approach to process CGM data. All algorithms are explicitly defined and do not rely on machine learning or external APIs.

#### 1. Stream Normalization
- **Sorting**: Readings are sorted chronologically by timestamp
- **Deduplication**: Exact duplicates (same timestamp + same glucose value) are removed
- **Timestamp Correction**: Small timestamp drifts or out-of-order readings within a tolerance window are corrected

#### 2. Anomaly Detection
Three types of anomalies are detected:

- **Rate-of-Change (ROC) Anomalies**: 
  - Detects when glucose changes exceed a threshold (default: 50 mg/dL) within a time window (default: 5 minutes)
  - Formula: `|glucose[i] - glucose[i-1]| / time_delta > threshold`
  
- **Flatline Detection**:
  - Identifies consecutive readings with identical or near-identical values
  - Indicates potential sensor stuck or compression artifacts
  - Threshold: 5+ consecutive identical values
  
- **Compression Lows**:
  - Detects sharp drops followed by immediate recovery
  - Pattern: drop > 30 mg/dL within 5 minutes, then recovery > 20 mg/dL within 10 minutes

#### 3. Gap Filling
- **Gap Detection**: Identifies gaps where time difference between consecutive readings exceeds the expected interval (default: 10 minutes)
- **Interpolation**: Uses linear interpolation between neighboring readings to fill gaps
- **Marking**: Filled readings are marked with `is_filled = True`

#### 4. Trust Scoring
Each reading receives a trust score between 0.0 and 1.0:

- **Baseline**: 1.0 (fully trusted)
- **Penalties**:
  - ROC anomaly: -0.4
  - Flatline region: -0.3
  - Compression anomaly: -0.3
  - Filled gap: -0.2
  - Proximity to gap: -0.1 (if within 2 readings of a gap)
- **Final Score**: Clamped between 0.0 and 1.0

#### 5. Correction
- Readings with trust score < threshold (default: 0.4) are replaced with linearly interpolated values from neighboring trusted readings
- Original values are preserved in `raw_glucose` field

### Sensor Health Grading

Sensor health is graded A-F based on:

- **Anomaly Ratio**: Percentage of readings flagged as anomalies
- **Average Trust Score**: Mean trust score across all readings
- **Gap Time**: Total minutes of missing data
- **Flatline Duration**: Total time spent in flatline regions

**Grade Thresholds**:
- **A**: Anomaly ratio < 5%, avg trust > 0.9, gaps < 60 min
- **B**: Anomaly ratio < 10%, avg trust > 0.8, gaps < 120 min
- **C**: Anomaly ratio < 20%, avg trust > 0.7, gaps < 240 min
- **D**: Anomaly ratio < 30%, avg trust > 0.6, gaps < 360 min
- **F**: Exceeds D thresholds

### Failure Prediction

The system computes a **Sensor Failure Risk Index (SFRI)** on a 0.0-1.0 scale:

**Components**:
1. **Trust Trend**: Moving average and slope of trust scores over time
2. **Drift Analysis**: Direction and magnitude of glucose drift (upward/downward)
3. **Compression Risk**: Frequency and severity of compression-like events
4. **Instability Windows**: Periods of high variance in trust scores

**SFRI Calculation**:
- Weighted combination of normalized risk factors
- Categories: LOW (< 0.25), MODERATE (0.25-0.5), HIGH (0.5-0.75), CRITICAL (> 0.75)

**Time-to-Failure (TTF) Estimation**:
- Based on SFRI and trend analysis
- Categories: 0-3 hours, 3-6 hours, 6-12 hours, Low Risk (> 12 hours)

### Bolus Risk Analysis

Identifies high-risk periods where:
- Glucose is in critical ranges (< 54 mg/dL or > 250 mg/dL) with low trust scores
- Rapid changes occur in unreliable readings
- Sensor data quality is compromised during critical decision points

### CSV Format Support

DRL supports multiple CGM export formats:

- **Generic**: `timestamp`, `glucose` columns
- **Dexcom Clarity**: Detects `GlucoseDisplayTime`, `Glucose Value`, etc.
- **Libre/LibreView**: Detects `Device Timestamp`, `Sensor Glucose (mg/dL)`, etc.
- **Medtronic/CareLink**: Combines `Date` and `Time` columns, detects `Sensor Glucose (mg/dL)`

Column detection is case-insensitive and handles variations in naming conventions.

### Limitations

- **Deterministic Rules Only**: No machine learning or statistical models
- **Not Clinically Validated**: All algorithms are experimental and rule-based
- **Educational Purpose**: This tool is for research and educational use only
- **No Medical Advice**: Results should not be used for treatment decisions

## License

This is a portfolio project for demonstration purposes.

