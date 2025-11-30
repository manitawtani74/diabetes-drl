"""
CSV Loader for CGM Data

Handles multiple CGM export formats and normalizes them into RawReading format.
Supports: Generic, Dexcom, Libre/LibreView, Medtronic/CareLink
"""

import csv
import io
from datetime import datetime
from typing import List, Optional, Tuple, Iterable
import re

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from .models import RawReading


def load_raw_readings_from_upload(file_bytes: bytes) -> List[RawReading]:
    """
    Load CGM data from uploaded file bytes.
    
    This is the main entry point for CSV loading from file uploads.
    Uses pandas for robust parsing of complex CSV structures (e.g., Dexcom multi-block files).
    
    Args:
        file_bytes: Raw bytes from file upload
        
    Returns:
        List of RawReading objects, sorted by timestamp
        
    Raises:
        ValueError: If CSV cannot be parsed or contains no usable glucose data
    """
    # Try pandas-based parsing first (better for complex Dexcom files)
    if PANDAS_AVAILABLE:
        try:
            readings = _load_with_pandas(file_bytes)
            if readings:
                return readings
        except Exception:
            # Fall back to traditional CSV parsing if pandas fails
            pass
    
    # Fallback to traditional CSV parsing
    try:
        # Decode to text (handle UTF-8, replace errors)
        try:
            raw_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = file_bytes.decode("utf-8", errors="replace")
        
        # Handle BOM if present
        if raw_text.startswith('\ufeff'):
            raw_text = raw_text[1:]
        
        # Create file-like object
        file_obj = io.StringIO(raw_text)
        
        # Delegate to load_cgm_csv
        return load_cgm_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")


def _load_with_pandas(file_bytes: bytes) -> List[RawReading]:
    """
    Load CGM data using pandas for robust parsing of complex CSV structures.
    
    Specifically designed to handle Dexcom Clarity multi-block exports.
    
    Args:
        file_bytes: Raw bytes from file upload
        
    Returns:
        List of RawReading objects, sorted by timestamp
    """
    if not PANDAS_AVAILABLE:
        return []
    
    # Read CSV with pandas using python engine to handle weird rows
    try:
        # Try with on_bad_lines parameter (pandas >= 1.3.0)
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", on_bad_lines='skip')
        except TypeError:
            # Fallback for older pandas versions
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", error_bad_lines=False, warn_bad_lines=False)
    except Exception:
        # If pandas can't parse, return empty list to trigger fallback
        return []
    
    if df.empty:
        return []
    
    # Normalize all column names: lowercase, remove spaces, remove punctuation
    df.columns = [_normalize_column_name(col) for col in df.columns]
    
    # Detect timestamp and glucose columns using normalized names
    timestamp_col = _find_timestamp_column(df.columns)
    glucose_col = _find_glucose_column(df.columns)
    
    if not timestamp_col or not glucose_col:
        return []
    
    # Extract valid glucose readings
    raw_readings = []
    
    # Convert timestamp column to datetime, coercing errors to NaT
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    
    # Convert glucose column to numeric, coercing errors to NaN
    df[glucose_col] = pd.to_numeric(df[glucose_col], errors='coerce')
    
    # Filter rows: must have valid timestamp AND valid glucose value
    valid_mask = df[timestamp_col].notna() & df[glucose_col].notna()
    valid_df = df[valid_mask].copy()
    
    # Additional filtering: glucose should be in reasonable range (0-600 mg/dL)
    valid_df = valid_df[(valid_df[glucose_col] >= 0) & (valid_df[glucose_col] <= 600)]
    
    # Extract readings
    for _, row in valid_df.iterrows():
        try:
            timestamp = row[timestamp_col]
            glucose = float(row[glucose_col])
            
            # Check for valid values
            if pd.notna(timestamp) and pd.notna(glucose):
                # Convert pandas Timestamp to datetime if needed
                if isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.to_pydatetime()
                elif hasattr(timestamp, 'to_pydatetime'):
                    timestamp = timestamp.to_pydatetime()
                
                raw_readings.append(RawReading(timestamp=timestamp, glucose=glucose))
        except (ValueError, TypeError, AttributeError):
            continue
    
    # Sort by timestamp and remove duplicates
    raw_readings.sort(key=lambda r: r.timestamp)
    
    # Remove duplicates (same timestamp + same glucose)
    seen = set()
    unique_readings = []
    for reading in raw_readings:
        key = (reading.timestamp, reading.glucose)
        if key not in seen:
            seen.add(key)
            unique_readings.append(reading)
    
    return unique_readings


def _normalize_column_name(name: str) -> str:
    """
    Aggressively normalize column name for matching.
    
    - Convert to lowercase
    - Remove spaces
    - Remove punctuation: [,.;():"']
    """
    if not name:
        return ""
    
    # Handle pandas NaN
    if PANDAS_AVAILABLE and pd.isna(name):
        return ""
    
    name = str(name).lower()
    # Remove spaces
    name = name.replace(" ", "")
    # Remove punctuation
    name = re.sub(r'[,.;():"\']', '', name)
    
    return name


def _find_timestamp_column(columns: List[str]) -> Optional[str]:
    """
    Find timestamp column from normalized column names.
    
    Looks for: glucosedisplaytime, glucosinternaltime, timestamp, systemtime, displaytime
    Priority order matters - earlier matches are preferred.
    """
    # Priority order: more specific first
    timestamp_candidates = [
        "glucosedisplaytime",      # Dexcom specific
        "glucosinternaltime",      # Dexcom specific
        "displaytime",             # Dexcom
        "systemtime",              # Dexcom
        "timestamp",               # Generic
        "eventtime",               # Event-related
        "datetime",                # Generic
        "date",                    # Generic
        "time",                    # Generic (lowest priority)
    ]
    
    # First pass: exact matches or contains
    for candidate in timestamp_candidates:
        for col in columns:
            normalized = _normalize_column_name(col)
            if candidate == normalized or candidate in normalized:
                return col
    
    # Second pass: partial matches (e.g., "time" in "eventtime")
    for col in columns:
        normalized = _normalize_column_name(col)
        if "time" in normalized and len(normalized) > 3:
            # Make sure it's not something like "eventtype"
            if "type" not in normalized:
                return col
    
    return None


def _find_glucose_column(columns: List[str]) -> Optional[str]:
    """
    Find glucose column from normalized column names.
    
    Looks for: glucosevalue, glucose, value, bgreadingsmgdl
    Priority order matters - more specific matches first.
    """
    # Priority order: more specific first
    glucose_candidates = [
        "glucosevalue",            # Dexcom specific
        "bgreadingsmgdl",          # Dexcom with units
        "sensorglucose",           # Sensor-specific
        "glucosereading",          # Reading-specific
        "glucose",                 # Generic glucose
        "value",                   # Generic value (lowest priority)
    ]
    
    # Exclude columns that are clearly not glucose values
    exclude_patterns = [
        "eventtype",
        "recordtype",
        "eventvalue",  # Event values are not glucose readings
        "event",
        "type",
    ]
    
    # First pass: exact matches or contains (with exclusions)
    for candidate in glucose_candidates:
        for col in columns:
            normalized = _normalize_column_name(col)
            
            # Skip if it matches exclusion patterns
            if any(exclude in normalized for exclude in exclude_patterns):
                continue
            
            if candidate == normalized or candidate in normalized:
                return col
    
    # Second pass: look for "glucose" anywhere in column name
    for col in columns:
        normalized = _normalize_column_name(col)
        if "glucose" in normalized:
            # Make sure it's not an event type or record type column
            if not any(exclude in normalized for exclude in exclude_patterns):
                return col
    
    return None


def load_cgm_csv(file_obj) -> List[RawReading]:
    """
    Load CGM data from a CSV file, supporting multiple vendor formats.
    
    Supports:
    - Generic format (timestamp, glucose columns)
    - Dexcom Clarity exports
    - Libre/LibreView exports
    - Medtronic/CareLink exports
    
    Args:
        file_obj: File-like object (text mode, UTF-8)
        
    Returns:
        List of RawReading objects, sorted by timestamp
        
    Raises:
        ValueError: If timestamp and glucose columns cannot be detected
    """
    # Read and normalize the CSV
    try:
        # Handle BOM if present
        content = file_obj.read()
        if content.startswith('\ufeff'):
            content = content[1:]  # Remove BOM
        
        file_obj = io.StringIO(content)
        reader = csv.DictReader(file_obj)
        
        # Normalize headers: lowercase, strip, remove quotes
        normalized_headers = {}
        original_headers = reader.fieldnames
        if not original_headers:
            raise ValueError("CSV file appears to be empty or has no headers.")
        
        for header in original_headers:
            normalized = header.lower().strip().strip('"').strip("'")
            normalized_headers[normalized] = header
        
        # Try to detect timestamp and glucose columns
        timestamp_col, glucose_col = _detect_columns(normalized_headers, original_headers)
        
        if not timestamp_col or not glucose_col:
            # Try pandas-based parsing as fallback
            try:
                content = file_obj.read()
                file_obj.seek(0)  # Reset for potential retry
                file_bytes = content.encode('utf-8') if isinstance(content, str) else content
                pandas_readings = _load_with_pandas(file_bytes)
                if pandas_readings:
                    return pandas_readings
            except Exception:
                pass
            
            raise ValueError(
                "Uploaded file parsed successfully but contained no usable glucose rows. "
                "This file may be a Dexcom Event Log without glucose values."
            )
        
        # Parse rows
        raw_readings = []
        for row in reader:
            try:
                timestamp = _parse_timestamp(row, timestamp_col, normalized_headers)
                glucose = _parse_glucose(row, glucose_col, normalized_headers)
                
                if timestamp and glucose is not None:
                    raw_readings.append(RawReading(timestamp=timestamp, glucose=glucose))
            except (ValueError, TypeError) as e:
                # Skip rows that can't be parsed
                continue
        
        # Sort by timestamp
        raw_readings.sort(key=lambda r: r.timestamp)
        
        return raw_readings
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error reading CSV file: {str(e)}")


def _detect_columns(normalized_headers: dict, original_headers: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect timestamp and glucose columns from normalized headers.
    
    Returns:
        Tuple of (timestamp_column_name, glucose_column_name) from original headers
    """
    timestamp_col = None
    glucose_col = None
    
    # A) Generic format: exact match for "timestamp" and "glucose"
    if "timestamp" in normalized_headers and "glucose" in normalized_headers:
        return normalized_headers["timestamp"], normalized_headers["glucose"]
    
    # B) Dexcom Clarity style
    timestamp_col, glucose_col = _detect_dexcom_columns(normalized_headers, original_headers)
    if timestamp_col and glucose_col:
        return timestamp_col, glucose_col
    
    # C) Libre/LibreView style
    timestamp_col, glucose_col = _detect_libre_columns(normalized_headers, original_headers)
    if timestamp_col and glucose_col:
        return timestamp_col, glucose_col
    
    # D) Medtronic/CareLink style
    timestamp_col, glucose_col = _detect_medtronic_columns(normalized_headers, original_headers)
    if timestamp_col and glucose_col:
        return timestamp_col, glucose_col
    
    return None, None


def _detect_dexcom_columns(normalized_headers: dict, original_headers: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Detect Dexcom Clarity format columns."""
    timestamp_col = None
    glucose_col = None
    
    # Find timestamp column - Dexcom uses various formats
    # Priority: "timestamp" > "event time" > "date" + "time" > anything with "time"
    timestamp_candidates = []
    for norm, orig in normalized_headers.items():
        if norm == "timestamp":
            timestamp_candidates.insert(0, orig)  # Highest priority
        elif "event time" in norm or "eventtime" in norm:
            timestamp_candidates.insert(1 if len(timestamp_candidates) > 0 else 0, orig)
        elif "timestamp" in norm:
            timestamp_candidates.append(orig)
        elif ("date" in norm and "time" in norm) or norm == "time":
            timestamp_candidates.append(orig)
    
    if timestamp_candidates:
        timestamp_col = timestamp_candidates[0]
    
    # Find glucose column - Dexcom uses various glucose column names
    # Priority: "glucose value (mg/dl)" > "sensor glucose (mg/dl)" > "glucose" with "mg/dl"
    glucose_candidates = []
    for norm, orig in normalized_headers.items():
        if "glucose value" in norm and "mg/dl" in norm:
            glucose_candidates.insert(0, orig)  # Highest priority
        elif "sensor glucose" in norm and "mg/dl" in norm:
            glucose_candidates.insert(1 if len(glucose_candidates) > 0 else 0, orig)
        elif "glucose" in norm and "mg/dl" in norm:
            glucose_candidates.append(orig)
        elif norm == "glucose" or norm == "glucose value":
            # Sometimes mg/dL is in a separate unit column or implied
            glucose_candidates.append(orig)
    
    if glucose_candidates:
        glucose_col = glucose_candidates[0]
    
    return timestamp_col, glucose_col


def _detect_libre_columns(normalized_headers: dict, original_headers: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Detect Libre/LibreView format columns."""
    timestamp_col = None
    glucose_col = None
    
    # Find timestamp column
    for norm, orig in normalized_headers.items():
        if "device timestamp" in norm or ("timestamp" in norm and "device" in norm):
            timestamp_col = orig
            break
        elif "timestamp" in norm and not timestamp_col:
            timestamp_col = orig
    
    # Find glucose column (prefer in order: Scan > Sensor > Historic)
    glucose_candidates = []
    for norm, orig in normalized_headers.items():
        if "scan glucose" in norm and "mg/dl" in norm:
            glucose_candidates.insert(0, orig)  # Highest priority
        elif "sensor glucose" in norm and "mg/dl" in norm:
            glucose_candidates.insert(1 if len(glucose_candidates) > 0 else 0, orig)
        elif "historic glucose" in norm and "mg/dl" in norm:
            glucose_candidates.append(orig)  # Lowest priority
    
    if glucose_candidates:
        glucose_col = glucose_candidates[0]
    
    return timestamp_col, glucose_col


def _detect_medtronic_columns(normalized_headers: dict, original_headers: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Detect Medtronic/CareLink format columns."""
    timestamp_col = None
    glucose_col = None
    
    # Check for separate date and time columns
    date_col = None
    time_col = None
    
    for norm, orig in normalized_headers.items():
        if norm == "date":
            date_col = orig
        elif norm == "time":
            time_col = orig
    
    if date_col and time_col:
        # We'll combine them in _parse_timestamp
        timestamp_col = (date_col, time_col)
    else:
        # Look for a single timestamp column
        for norm, orig in normalized_headers.items():
            if "timestamp" in norm or ("date" in norm and "time" in norm):
                timestamp_col = orig
                break
    
    # Find glucose column
    for norm, orig in normalized_headers.items():
        if "sensor glucose" in norm and "mg/dl" in norm:
            glucose_col = orig
            break
    
    return timestamp_col, glucose_col


def _parse_timestamp(row: dict, timestamp_col, normalized_headers: dict) -> Optional[datetime]:
    """
    Parse timestamp from row.
    
    Handles:
    - ISO8601 strings
    - Separate date and time columns (Medtronic)
    - Various date/time formats
    """
    try:
        # Handle tuple (date_col, time_col) for Medtronic
        if isinstance(timestamp_col, tuple):
            date_col, time_col = timestamp_col
            date_str = row.get(date_col, "").strip()
            time_str = row.get(time_col, "").strip()
            
            if date_str and time_str:
                # Combine into ISO format
                combined = f"{date_str}T{time_str}"
                # Try parsing
                try:
                    return datetime.fromisoformat(combined.replace(" ", "T"))
                except ValueError:
                    # Try common formats
                    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"]:
                        try:
                            return datetime.strptime(combined, fmt)
                        except ValueError:
                            continue
            return None
        
        # Single timestamp column
        timestamp_str = row.get(timestamp_col, "").strip()
        if not timestamp_str:
            return None
        
        # Try ISO8601 first
        try:
            # Handle Z timezone
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1] + "+00:00"
            return datetime.fromisoformat(timestamp_str)
        except ValueError:
            pass
        
        # Try common formats (including Dexcom formats)
        formats = [
            "%Y-%m-%d %H:%M:%S",      # Standard format
            "%m/%d/%Y %H:%M:%S",      # US format
            "%Y-%m-%dT%H:%M:%S",      # ISO without timezone
            "%d/%m/%Y %H:%M:%S",      # European format
            "%Y-%m-%d %H:%M",         # Without seconds
            "%m/%d/%Y %H:%M",         # US without seconds
            "%Y-%m-%d %H:%M:%S.%f",   # With microseconds
            "%m/%d/%Y %I:%M %p",      # 12-hour format
            "%Y-%m-%d %I:%M:%S %p",   # 12-hour with seconds
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        return None
        
    except Exception:
        return None


def _parse_glucose(row: dict, glucose_col: str, normalized_headers: dict) -> Optional[float]:
    """Parse glucose value from row."""
    try:
        glucose_str = row.get(glucose_col, "").strip()
        if not glucose_str:
            return None
        
        # Remove common non-numeric characters (commas, spaces, units)
        glucose_str = glucose_str.replace(",", "").replace(" ", "")
        # Remove common unit suffixes
        glucose_str = re.sub(r'\s*(mg/dl|mg/dL|mg|mmol/L|mmol/l).*$', '', glucose_str, flags=re.IGNORECASE)
        
        # Try to convert to float
        try:
            glucose = float(glucose_str)
            # Sanity check: glucose should be in reasonable range
            # Allow 0-600 mg/dL (some sensors report 0 for errors, 600 is very high but possible)
            if 0 <= glucose <= 600:
                return glucose
        except ValueError:
            pass
        
        return None
        
    except Exception:
        return None

