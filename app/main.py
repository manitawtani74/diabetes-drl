"""
FastAPI Application for Diabetes Reliability Layer

Main web application that provides:
- CSV file upload interface
- CGM data processing via DRL pipeline
- Results visualization (metrics, table, chart)
"""

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta
from typing import Optional, List
import io
from app.engine.models import RawReading
from app.engine.pipeline import run_pipeline, EmptyWindowError
from app.engine.csv_loader import load_cgm_csv

app = FastAPI(title="Diabetes Reliability Layer", version="1.0.0")

# Simple in-memory cache for the last uploaded CSV (single-user, local use only)
LAST_UPLOAD = {
    "raw_readings": None,  # List[RawReading]
    "filename": None,      # str
    "min_ts": None,        # datetime
    "max_ts": None,        # datetime
}

# Setup templates
templates = Jinja2Templates(directory="app/templates")


def human_number(value, decimals: int = 3, small_threshold: float = 1e-4):
    """
    Format numeric values for display:
    - No scientific notation.
    - Tiny values near zero become "0".
    - Trim trailing zeros and decimal point.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        # Non-numeric or None: just return as-is
        return value
    
    # Treat very small values as 0
    if abs(v) < small_threshold:
        return "0"
    
    # Format with fixed decimals then trim
    text = f"{v:.{decimals}f}"
    # Strip trailing zeros and decimal point if not needed
    text = text.rstrip("0").rstrip(".")
    return text


# Register as a Jinja2 filter
templates.env.filters["human_number"] = human_number

# Mount static files (if needed)
# app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main upload and analysis page."""
    return templates.TemplateResponse(
        "upload_and_report.html",
            {
                "request": request,
                "metrics": None,
                "sensor_health": None,
                "bolus_risks": None,
                "readings": None,
                "chart_data": None,
                "view_mode": "patient",
                "view_scope": "full",
                "view_day": None,
                "view_start": None,
                "view_end": None,
                "view_month": None,
                "available_dates": [],
                "analysis_id": None,
                "analysis_result": None,
                "data_min_date": None,
                "data_max_date": None,
                "data_min_month": None,
                "data_max_month": None,
                "time_bounds": None,
                "config": {
                "max_delta_mgdl": 50.0,
                "max_delta_minutes": 5,
                "expected_interval_minutes": 5,
                "gap_threshold_minutes": 10,
                "trust_correction_threshold": 0.4,
                "time_in_range_low": 70,
                "time_in_range_high": 180,
                "low_threshold": 70,
                "very_low_threshold": 54,
                "high_threshold": 180,
                "very_high_threshold": 250,
            }
        }
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    file: Optional[UploadFile] = File(None),
    max_delta_mgdl: float = Form(50.0),
    max_delta_minutes: int = Form(5),
    expected_interval_minutes: int = Form(5),
    gap_threshold_minutes: int = Form(10),
    trust_correction_threshold: float = Form(0.4),
    time_in_range_low: float = Form(70.0),
    time_in_range_high: float = Form(180.0),
    low_threshold: float = Form(70.0),
    very_low_threshold: float = Form(54.0),
    high_threshold: float = Form(180.0),
    very_high_threshold: float = Form(250.0),
    view_scope: str = Form("full"),
    view_day: Optional[str] = Form(None),
    view_start: Optional[str] = Form(None),
    view_end: Optional[str] = Form(None),
    view_month: Optional[str] = Form(None),
    view_mode: str = Form("patient")
):
    """
    Process uploaded CSV file through DRL pipeline.
    
    Expected CSV format:
    - timestamp: ISO8601 format string
    - glucose: float or int (mg/dL)
    """
    try:
        # Validate and normalize glucose targets
        target_warning = None
        try:
            # Clamp to valid range
            time_in_range_low = max(40.0, min(400.0, float(time_in_range_low)))
            time_in_range_high = max(40.0, min(400.0, float(time_in_range_high)))
            low_threshold = max(40.0, min(400.0, float(low_threshold)))
            very_low_threshold = max(40.0, min(400.0, float(very_low_threshold)))
            high_threshold = max(40.0, min(400.0, float(high_threshold)))
            very_high_threshold = max(40.0, min(400.0, float(very_high_threshold)))
            
            # Validate relationships
            if time_in_range_low >= time_in_range_high:
                target_warning = "Time-in-range low must be less than high. Using defaults."
                time_in_range_low, time_in_range_high = 70.0, 180.0
            if very_low_threshold >= low_threshold:
                target_warning = "Very low threshold must be less than low threshold. Using defaults."
                very_low_threshold, low_threshold = 54.0, 70.0
            if high_threshold >= very_high_threshold:
                target_warning = "High threshold must be less than very high threshold. Using defaults."
                high_threshold, very_high_threshold = 180.0, 250.0
            
            # Build glucose target config
            glucose_target_config = {
                "time_in_range_low": time_in_range_low,
                "time_in_range_high": time_in_range_high,
                "low_threshold": low_threshold,
                "very_low_threshold": very_low_threshold,
                "high_threshold": high_threshold,
                "very_high_threshold": very_high_threshold,
            }
        except (ValueError, TypeError):
            target_warning = "Some target values were invalid. Default targets (70–180 mg/dL) were used."
            glucose_target_config = {
                "time_in_range_low": 70.0,
                "time_in_range_high": 180.0,
                "low_threshold": 70.0,
                "very_low_threshold": 54.0,
                "high_threshold": 180.0,
                "very_high_threshold": 250.0,
            }
        
        # Check if file was uploaded or use cached data
        raw_readings = []
        data_min_date = None
        data_max_date = None
        data_min_month = None
        data_max_month = None
        available_dates = []
        uploaded_filename = None
        
        if file and file.filename:
            # New file uploaded - read and cache it
            contents = await file.read()
            uploaded_filename = file.filename
            
            # Load using CSV loader
            try:
                from app.engine.csv_loader import load_raw_readings_from_upload
                raw_readings = load_raw_readings_from_upload(contents)
                
                # Cache the data
                if raw_readings:
                    timestamps = [r.timestamp for r in raw_readings]
                    LAST_UPLOAD["raw_readings"] = raw_readings
                    LAST_UPLOAD["filename"] = uploaded_filename
                    LAST_UPLOAD["min_ts"] = min(timestamps)
                    LAST_UPLOAD["max_ts"] = max(timestamps)
            except ValueError as e:
                # CSV loading failed - don't cache, show error
                return templates.TemplateResponse(
                    "upload_and_report.html",
                    {
                        "request": request,
                        "error": str(e),
                        "metrics": None,
                        "sensor_health": None,
                        "bolus_risks": None,
                        "failure_prediction": None,
                        "insights": None,
                        "readings": None,
                        "chart_data": None,
                        "glucose_targets": None,
                        "target_warning": None,
                        "view_scope": view_scope,
                        "view_day": view_day,
                        "view_start": view_start,
                        "view_end": view_end,
                        "view_month": view_month,
                        "view_mode": view_mode,
                        "data_min_date": None,
                        "data_max_date": None,
                        "data_min_month": None,
                        "data_max_month": None,
                        "time_bounds": None,
                        "available_dates": [],
                        "config": {
                            "max_delta_mgdl": max_delta_mgdl,
                            "max_delta_minutes": max_delta_minutes,
                            "expected_interval_minutes": expected_interval_minutes,
                            "gap_threshold_minutes": gap_threshold_minutes,
                            "trust_correction_threshold": trust_correction_threshold,
                            "time_in_range_low": glucose_target_config["time_in_range_low"],
                            "time_in_range_high": glucose_target_config["time_in_range_high"],
                            "low_threshold": glucose_target_config["low_threshold"],
                            "very_low_threshold": glucose_target_config["very_low_threshold"],
                            "high_threshold": glucose_target_config["high_threshold"],
                            "very_high_threshold": glucose_target_config["very_high_threshold"],
                        }
                    }
                )
        else:
            # No file uploaded - try to use cached data
            if LAST_UPLOAD["raw_readings"]:
                raw_readings = LAST_UPLOAD["raw_readings"]
                uploaded_filename = LAST_UPLOAD["filename"] or "cached_file.csv"
            else:
                # No file and no cache - show error
                return templates.TemplateResponse(
                    "upload_and_report.html",
                    {
                        "request": request,
                        "error": "Please upload a CSV file to analyze. No previous file is available.",
                        "metrics": None,
                        "sensor_health": None,
                        "bolus_risks": None,
                        "failure_prediction": None,
                        "insights": None,
                        "readings": None,
                        "chart_data": None,
                        "glucose_targets": None,
                        "target_warning": None,
                        "view_scope": view_scope,
                        "view_day": view_day,
                        "view_start": view_start,
                        "view_end": view_end,
                        "view_month": view_month,
                        "view_mode": view_mode,
                        "data_min_date": None,
                        "data_max_date": None,
                        "data_min_month": None,
                        "data_max_month": None,
                        "time_bounds": None,
                        "available_dates": [],
                        "config": {
                            "max_delta_mgdl": max_delta_mgdl,
                            "max_delta_minutes": max_delta_minutes,
                            "expected_interval_minutes": expected_interval_minutes,
                            "gap_threshold_minutes": gap_threshold_minutes,
                            "trust_correction_threshold": trust_correction_threshold,
                            "time_in_range_low": glucose_target_config["time_in_range_low"],
                            "time_in_range_high": glucose_target_config["time_in_range_high"],
                            "low_threshold": glucose_target_config["low_threshold"],
                            "very_low_threshold": glucose_target_config["very_low_threshold"],
                            "high_threshold": glucose_target_config["high_threshold"],
                            "very_high_threshold": glucose_target_config["very_high_threshold"],
                        }
                    }
                )
        
        # Compute date constraints from raw_readings
        if raw_readings:
            timestamps = [r.timestamp for r in raw_readings]
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            
            data_min_date = min_ts.date().isoformat()
            data_max_date = max_ts.date().isoformat()
            data_min_month = min_ts.strftime("%Y-%m")
            data_max_month = max_ts.strftime("%Y-%m")
            
            # Create time_bounds dict for template
            time_bounds = {
                "min_date": data_min_date,
                "max_date": data_max_date,
                "min_month": data_min_month,
                "max_month": data_max_month,
            }
            
            # Extract available dates from raw_readings (before filtering)
            available_dates = sorted(list(set(
                r.timestamp.date().isoformat() for r in raw_readings
            )))
        else:
            time_bounds = None
            available_dates = []
        
        # Check if we got any readings
        if not raw_readings:
            return templates.TemplateResponse(
                    "upload_and_report.html",
                    {
                        "request": request,
                        "error": "Uploaded file parsed successfully but contained no usable glucose rows. This file may be a Dexcom Event Log without glucose values.",
                        "metrics": None,
                        "sensor_health": None,
                        "bolus_risks": None,
                        "failure_prediction": None,
                        "insights": None,
                        "readings": None,
                        "chart_data": None,
                        "glucose_targets": None,
                        "target_warning": None,
                        "view_scope": view_scope,
                        "view_day": view_day,
                        "view_start": view_start,
                        "view_end": view_end,
                        "view_month": view_month,
                        "view_mode": view_mode,
                        "data_min_date": data_min_date,
                        "data_max_date": data_max_date,
                        "data_min_month": data_min_month,
                        "data_max_month": data_max_month,
                        "time_bounds": time_bounds,
                        "available_dates": available_dates,
                        "config": {
                            "max_delta_mgdl": max_delta_mgdl,
                            "max_delta_minutes": max_delta_minutes,
                            "expected_interval_minutes": expected_interval_minutes,
                            "gap_threshold_minutes": gap_threshold_minutes,
                            "trust_correction_threshold": trust_correction_threshold,
                            "time_in_range_low": 70,
                            "time_in_range_high": 180,
                            "low_threshold": 70,
                            "very_low_threshold": 54,
                            "high_threshold": 180,
                            "very_high_threshold": 250,
                        }
                    }
                )
        
        # Additional check (redundant but safe)
        if not raw_readings:
            return templates.TemplateResponse(
                "upload_and_report.html",
                {
                    "request": request,
                    "error": "The file was read successfully but no glucose data points were found.",
                    "metrics": None,
                    "sensor_health": None,
                    "bolus_risks": None,
                    "failure_prediction": None,
                    "insights": None,
                    "readings": None,
                    "chart_data": None,
                    "glucose_targets": None,
                    "target_warning": None,
                    "view_scope": view_scope,
                    "view_day": view_day,
                    "view_start": view_start,
                    "view_end": view_end,
                    "view_month": view_month,
                    "view_mode": view_mode,
                        "data_min_date": data_min_date,
                        "data_max_date": data_max_date,
                        "data_min_month": data_min_month,
                        "data_max_month": data_max_month,
                        "time_bounds": time_bounds,
                        "available_dates": available_dates,
                    "analysis_id": None,
                    "config": {
                        "max_delta_mgdl": max_delta_mgdl,
                        "max_delta_minutes": max_delta_minutes,
                        "expected_interval_minutes": expected_interval_minutes,
                        "gap_threshold_minutes": gap_threshold_minutes,
                        "trust_correction_threshold": trust_correction_threshold,
                        "time_in_range_low": 70,
                        "time_in_range_high": 180,
                        "low_threshold": 70,
                        "very_low_threshold": 54,
                        "high_threshold": 180,
                        "very_high_threshold": 250,
                    }
                }
            )
        
        # Note: Time window filtering is now handled in run_pipeline via view_scope parameters
        # Keep start_time/end_time for backward compatibility if needed
        start_time = None
        end_time = None
        
        # Run pipeline
        processed_readings, results = run_pipeline(
            raw_readings,
            max_delta_mgdl=max_delta_mgdl,
            max_delta_minutes=max_delta_minutes,
            expected_interval_minutes=expected_interval_minutes,
            gap_threshold_minutes=gap_threshold_minutes,
            trust_correction_threshold=trust_correction_threshold,
            glucose_targets=glucose_target_config,
            start_time=start_time,
            end_time=end_time,
            view_scope=view_scope,
            view_day=view_day,
            view_start=view_start,
            view_end=view_end,
            view_month=view_month
        )
        
        # Check if pipeline returned no data
        if not results.get("has_data", True) and results.get("error") == "NO_DATA_IN_RANGE":
            return templates.TemplateResponse(
                "upload_and_report.html",
                {
                    "request": request,
                    "range_error": "No CGM readings were found in the selected time window. Try a different day, range, or month.",
                    "error": None,
                    "analysis": None,
                    "view_scope": view_scope,
                    "view_day": view_day,
                    "view_start": view_start,
                    "view_end": view_end,
                    "view_month": view_month,
                    "data_min_date": data_min_date,
                    "data_max_date": data_max_date,
                    "data_min_month": data_min_month,
                    "data_max_month": data_max_month,
                    "metrics": None,
                    "sensor_health": None,
                    "bolus_risks": None,
                    "failure_prediction": None,
                    "insights": None,
                    "readings": None,
                    "chart_data": None,
                    "glucose_targets": glucose_target_config,
                    "target_warning": target_warning,
                    "view_scope": view_scope,
                    "view_day": view_day,
                    "view_start": view_start,
                    "view_end": view_end,
                    "view_month": view_month,
                    "view_mode": view_mode,
                    "available_dates": available_dates,
                    "analysis_id": None,
                    "config": {
                        "max_delta_mgdl": max_delta_mgdl,
                        "max_delta_minutes": max_delta_minutes,
                        "expected_interval_minutes": expected_interval_minutes,
                        "gap_threshold_minutes": gap_threshold_minutes,
                        "trust_correction_threshold": trust_correction_threshold,
                        "time_in_range_low": glucose_target_config["time_in_range_low"],
                        "time_in_range_high": glucose_target_config["time_in_range_high"],
                        "low_threshold": glucose_target_config["low_threshold"],
                        "very_low_threshold": glucose_target_config["very_low_threshold"],
                        "high_threshold": glucose_target_config["high_threshold"],
                        "very_high_threshold": glucose_target_config["very_high_threshold"],
                    }
                }
            )
        
        # Extract results
        basic_metrics = results.get("basic_metrics", {})
        sensor_health = results.get("sensor_health", {})
        bolus_risks = results.get("bolus_risks", {})
        failure_prediction = results.get("failure_prediction", {})
        insights_raw = results.get("insights", [])
        
        # Convert insights to dictionaries for template
        insights = [
            {
                "title": insight.title,
                "severity": insight.severity,
                "message": insight.message,
                "details": insight.details,
            }
            for insight in insights_raw
        ]
        
        # Prepare data for template
        readings_data = [
            {
                "timestamp": r.timestamp.isoformat(),
                "raw_glucose": round(r.raw_glucose, 2),
                "corrected_glucose": round(r.corrected_glucose, 2),
                "trust_score": round(r.trust_score, 3),
                "is_anomaly": r.is_anomaly,
                "anomaly_reasons": ", ".join(r.anomaly_reasons) if r.anomaly_reasons else "",
                "is_filled": r.is_filled,
            }
            for r in processed_readings
        ]
        
        # Prepare chart data
        chart_data = {
            "timestamps": [r.timestamp.isoformat() for r in processed_readings],
            "raw_glucose": [round(r.raw_glucose, 2) for r in processed_readings],
            "corrected_glucose": [round(r.corrected_glucose, 2) for r in processed_readings],
            "trust_scores": [round(r.trust_score, 3) for r in processed_readings],
        }
        
        # Build a compact summary for the "Saved Analyses" dashboard
        # Extract time-in-range data from insights
        time_in_range_pct = None
        above_range_pct = None
        below_range_pct = None
        for insight in insights:
            if insight.get("title") == "Glucose Range Summary":
                details = insight.get("details", {})
                time_in_range_pct = details.get("pct_in_range")
                above_range_pct = details.get("pct_above_high")
                below_range_pct = details.get("pct_below_low")
                break
        
        # Extract SFRI and risk category from failure_prediction
        sfr_index = None
        risk_category = None
        if failure_prediction and failure_prediction.get("sfri"):
            sfr_index = failure_prediction["sfri"].get("sfri_score")
            risk_category = failure_prediction["sfri"].get("risk_category")
        
        # Extract summary information from results
        summary = results.get("summary", {})
        
        # Build the full report dict for storage and rendering
        full_report = {
            "file_info": {
                "source": "uploaded_csv",
                "file_name": file.filename,
                "view_scope": view_scope,
                "view_day": view_day,
                "view_start": view_start,
                "view_end": view_end,
                "view_month": view_month,
            },
            "summary": summary,
            "targets": glucose_target_config,
            "metrics": basic_metrics,
            "sensor_health": sensor_health,
            "bolus_risks": bolus_risks,
            "failure_prediction": failure_prediction,
            "insights": insights,
            "readings": readings_data,
            "chart_data": chart_data,
            "glucose_targets": glucose_target_config,
            "target_warning": target_warning,
        }
        
        # Build summary row for the table
        summary_row = {
            "sensor_grade": sensor_health.get("health_grade") if sensor_health else None,
            "anomaly_ratio_pct": round((sensor_health.get("anomaly_ratio") or 0) * 100, 1) if sensor_health else None,
            "avg_trust_score": sensor_health.get("average_trust_score") if sensor_health else None,
            "tir_pct": round(time_in_range_pct, 1) if time_in_range_pct is not None else None,
            "above_range_pct": round(above_range_pct, 1) if above_range_pct is not None else None,
            "below_range_pct": round(below_range_pct, 1) if below_range_pct is not None else None,
            "sfri": round(sfr_index, 3) if sfr_index is not None else None,
            "risk_category": risk_category.upper() if risk_category else None,
        }
        
        return templates.TemplateResponse(
            "upload_and_report.html",
            {
                "request": request,
                "metrics": basic_metrics,
                "sensor_health": sensor_health,
                "bolus_risks": bolus_risks,
                "failure_prediction": failure_prediction,
                "insights": insights,
                "readings": readings_data,
                "chart_data": chart_data,
                "glucose_targets": glucose_target_config,
                "target_warning": target_warning,
                "analysis_result": full_report,
                "view_scope": view_scope,
                "view_day": view_day,
                "view_start": view_start,
                "view_end": view_end,
                "view_month": view_month,
                "view_mode": view_mode,
                "available_dates": available_dates,
                "range_error": None,
                "data_min_date": data_min_date,
                "data_max_date": data_max_date,
                "data_min_month": data_min_month,
                "data_max_month": data_max_month,
                "config": {
                    "max_delta_mgdl": max_delta_mgdl,
                    "max_delta_minutes": max_delta_minutes,
                    "expected_interval_minutes": expected_interval_minutes,
                    "gap_threshold_minutes": gap_threshold_minutes,
                    "trust_correction_threshold": trust_correction_threshold,
                    "time_in_range_low": glucose_target_config["time_in_range_low"],
                    "time_in_range_high": glucose_target_config["time_in_range_high"],
                    "low_threshold": glucose_target_config["low_threshold"],
                    "very_low_threshold": glucose_target_config["very_low_threshold"],
                    "high_threshold": glucose_target_config["high_threshold"],
                    "very_high_threshold": glucose_target_config["very_high_threshold"],
                }
            }
        )
    
    except Exception as e:
        # Safely get view_mode from form if available, otherwise default to "patient"
        try:
            form = await request.form()
            view_mode = form.get("view_mode", "patient")
        except Exception:
            view_mode = "patient"
        
        return templates.TemplateResponse(
            "upload_and_report.html",
            {
                "request": request,
                "error": f"Error processing file: {str(e)}",
                "metrics": None,
                "sensor_health": None,
                "bolus_risks": None,
                "failure_prediction": None,
                "insights": None,
                "readings": None,
                "chart_data": None,
                "glucose_targets": None,
                "target_warning": None,
                "view_mode": view_mode,
                "view_scope": "full",
                "view_day": None,
                "view_start": None,
                "view_end": None,
                "view_month": None,
                "available_dates": [],
                "data_min_date": None,
                "data_max_date": None,
                "data_min_month": None,
                "data_max_month": None,
                "config": {
                    "max_delta_mgdl": max_delta_mgdl,
                    "max_delta_minutes": max_delta_minutes,
                    "expected_interval_minutes": expected_interval_minutes,
                    "gap_threshold_minutes": gap_threshold_minutes,
                    "trust_correction_threshold": trust_correction_threshold,
                    "time_in_range_low": 70,
                    "time_in_range_high": 180,
                    "low_threshold": 70,
                    "very_low_threshold": 54,
                    "high_threshold": 180,
                    "very_high_threshold": 250,
                }
            }
        )


@app.post("/report/json")
async def download_json(
    file: UploadFile = File(...),
    max_delta_mgdl: float = Form(50.0),
    max_delta_minutes: int = Form(5),
    expected_interval_minutes: int = Form(5),
    gap_threshold_minutes: int = Form(10),
    trust_correction_threshold: float = Form(0.4),
    time_in_range_low: float = Form(70.0),
    time_in_range_high: float = Form(180.0),
    low_threshold: float = Form(70.0),
    very_low_threshold: float = Form(54.0),
    high_threshold: float = Form(180.0),
    very_high_threshold: float = Form(250.0),
    view_scope: str = Form("full"),
    view_day: Optional[str] = Form(None),
    view_start: Optional[str] = Form(None),
    view_end: Optional[str] = Form(None),
    view_month: Optional[str] = Form(None),
    view_mode: str = Form("patient")
):
    """
    Process uploaded CSV file through DRL pipeline and return JSON summary.
    
    Accepts the same parameters as the HTML analysis endpoint.
    Returns a clean JSON object with all key metrics and insights.
    """
    try:
        # Validate and normalize glucose targets (same logic as /analyze)
        try:
            # Clamp to valid range
            time_in_range_low = max(40.0, min(400.0, float(time_in_range_low)))
            time_in_range_high = max(40.0, min(400.0, float(time_in_range_high)))
            low_threshold = max(40.0, min(400.0, float(low_threshold)))
            very_low_threshold = max(40.0, min(400.0, float(very_low_threshold)))
            high_threshold = max(40.0, min(400.0, float(high_threshold)))
            very_high_threshold = max(40.0, min(400.0, float(very_high_threshold)))
            
            # Validate relationships
            if time_in_range_low >= time_in_range_high:
                time_in_range_low, time_in_range_high = 70.0, 180.0
            if very_low_threshold >= low_threshold:
                very_low_threshold, low_threshold = 54.0, 70.0
            if high_threshold >= very_high_threshold:
                high_threshold, very_high_threshold = 180.0, 250.0
            
            # Build glucose target config
            glucose_target_config = {
                "time_in_range_low": time_in_range_low,
                "time_in_range_high": time_in_range_high,
                "low_threshold": low_threshold,
                "very_low_threshold": very_low_threshold,
                "high_threshold": high_threshold,
                "very_high_threshold": very_high_threshold,
            }
        except (ValueError, TypeError):
            glucose_target_config = {
                "time_in_range_low": 70.0,
                "time_in_range_high": 180.0,
                "low_threshold": 70.0,
                "very_low_threshold": 54.0,
                "high_threshold": 180.0,
                "very_high_threshold": 250.0,
            }
        
        # Read CSV file
        contents = await file.read()
        
        # Load using CSV loader
        try:
            from app.engine.csv_loader import load_raw_readings_from_upload
            raw_readings = load_raw_readings_from_upload(contents)
            
            if not raw_readings:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Uploaded file parsed successfully but contained no usable glucose rows. This file may be a Dexcom Event Log without glucose values."
                    }
                )
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content={"error": str(e)}
            )
        
        # Extract available dates from raw_readings (before filtering)
        available_dates = sorted(list(set(
            r.timestamp.date().isoformat() for r in raw_readings
        )))
        
        # Note: Time window filtering is now handled in run_pipeline via view_scope parameters
        start_time = None
        end_time = None
        
        # Run pipeline
        processed_readings, results = run_pipeline(
            raw_readings,
            max_delta_mgdl=max_delta_mgdl,
            max_delta_minutes=max_delta_minutes,
            expected_interval_minutes=expected_interval_minutes,
            gap_threshold_minutes=gap_threshold_minutes,
            trust_correction_threshold=trust_correction_threshold,
            glucose_targets=glucose_target_config,
            start_time=start_time,
            end_time=end_time,
            view_scope=view_scope,
            view_day=view_day,
            view_start=view_start,
            view_end=view_end,
            view_month=view_month
        )
        
        # Check if pipeline returned no data
        if not results.get("has_data", True) and results.get("error") == "NO_DATA_IN_RANGE":
            return JSONResponse(
                status_code=400,
                content={"error": "No CGM readings were found in the selected time window. Try a different day, range, or month."}
            )
        
        # Extract results
        basic_metrics = results.get("basic_metrics", {})
        sensor_health = results.get("sensor_health", {})
        bolus_risks = results.get("bolus_risks", {})
        failure_prediction = results.get("failure_prediction", {})
        insights_raw = results.get("insights", [])
        
        # Build JSON summary
        summary = {
            "file_info": {
                "source": "uploaded_csv",
                "view_scope": view_scope,
                "view_day": view_day,
                "view_start": view_start,
                "view_end": view_end,
                "view_month": view_month,
                "available_dates": available_dates,
            },
            "targets": {
                "time_in_range_low": glucose_target_config["time_in_range_low"],
                "time_in_range_high": glucose_target_config["time_in_range_high"],
                "low_threshold": glucose_target_config["low_threshold"],
                "very_low_threshold": glucose_target_config["very_low_threshold"],
                "high_threshold": glucose_target_config["high_threshold"],
                "very_high_threshold": glucose_target_config["very_high_threshold"],
            },
            "metrics": basic_metrics,
            "sensor_health": sensor_health,
            "bolus_risks": {
                "risk_summary": bolus_risks.get("risk_summary"),
                "total_high_risk": bolus_risks.get("total_high_risk", 0),
                "high_risk_readings_count": len(bolus_risks.get("high_risk_readings", [])),
                "risk_zones": [
                    {
                        "start": zone.get("start"),
                        "end": zone.get("end"),
                        "risk_level": zone.get("risk_level"),
                    }
                    for zone in bolus_risks.get("risk_zones", [])
                ] if bolus_risks.get("risk_zones") else [],
            },
            "failure_prediction": {
                "sfri": failure_prediction.get("sfri"),
                "ttf": failure_prediction.get("ttf"),
                "trust_trend": failure_prediction.get("trust_trend"),
                "drift": failure_prediction.get("drift"),
                "compression_risk": failure_prediction.get("compression_risk"),
                "instability": failure_prediction.get("instability"),
                "not_enough_data": failure_prediction.get("not_enough_data", False),
            },
            "insights": [
                {
                    "title": insight.title,
                    "severity": insight.severity,
                    "message": insight.message,
                    "details": insight.details,
                }
                for insight in insights_raw
            ],
        }
        
        return JSONResponse(content=summary)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing file: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

