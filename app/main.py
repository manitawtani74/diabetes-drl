"""
FastAPI Application for Diabetes Reliability Layer

Main web application that provides:
- CSV file upload interface
- CGM data processing via DRL pipeline
- Results visualization (metrics, table, chart)
"""

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta
from typing import Optional, List
import io
import uuid
# SAFE IMPORT FOR WEASYPRINT
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except Exception:
    weasyprint = None
    WEASYPRINT_AVAILABLE = False
    print("⚠️  WeasyPrint not available — PDF export disabled.")

from app.engine.models import RawReading
from app.engine.pipeline import run_pipeline, EmptyWindowError
from app.engine.csv_loader import load_cgm_csv

# In-memory storage for PDF generation (simple dict, cleared per request)
_analysis_cache = {}


def pdf_error_html():
    return HTMLResponse(
        "<h2 style='color:red;'>PDF export is not available on this system.</h2>"
        "<p>This feature requires native libraries (GTK/Pango/Cairo) that are missing.</p>"
        "<p>Your report is still viewable normally in the browser.</p>"
    )

app = FastAPI(title="Diabetes Reliability Layer", version="1.0.0")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

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
                "view_scope": "all",
                "view_date": None,
                "available_dates": [],
                "analysis_id": None,
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
    view_scope: str = Form("all"),
    view_date: Optional[str] = Form(None),
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
        
        # Read CSV file
        contents = await file.read()
        
        # Decode to text (handle UTF-8, replace errors)
        try:
            raw_text = contents.decode("utf-8")
        except UnicodeDecodeError:
            # Try with error replacement
            raw_text = contents.decode("utf-8", errors="replace")
        
        # Handle BOM if present
        if raw_text.startswith('\ufeff'):
            raw_text = raw_text[1:]
        
        # Load using CSV loader (new function that handles bytes directly)
        try:
            from app.engine.csv_loader import load_raw_readings_from_upload
            raw_readings = load_raw_readings_from_upload(contents)
            
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
        except ValueError as e:
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
                    "view_date": view_date,
                    "view_mode": view_mode,
                    "available_dates": [],
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
        
        # Extract available dates from raw_readings (before filtering)
        available_dates = sorted(list(set(
            r.timestamp.date().isoformat() for r in raw_readings
        )))
        
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
                    "view_date": view_date,
                    "view_mode": view_mode,
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
        
        # Extract available dates from raw_readings (before filtering)
        available_dates = sorted(list(set(
            r.timestamp.date().isoformat() for r in raw_readings
        )))
        
        # Determine time window for filtering
        start_time = None
        end_time = None
        if view_scope == "day" and view_date:
            try:
                # Parse date and create time window (00:00 to 00:00 next day)
                selected_date = datetime.strptime(view_date, "%Y-%m-%d")
                start_time = selected_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = start_time + timedelta(days=1)
            except ValueError:
                # Invalid date format, use all data
                pass
        
        # Run pipeline
        try:
            processed_readings, results = run_pipeline(
                raw_readings,
                max_delta_mgdl=max_delta_mgdl,
                max_delta_minutes=max_delta_minutes,
                expected_interval_minutes=expected_interval_minutes,
                gap_threshold_minutes=gap_threshold_minutes,
                trust_correction_threshold=trust_correction_threshold,
                glucose_targets=glucose_target_config,
                start_time=start_time,
                end_time=end_time
            )
        except EmptyWindowError as e:
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
                    "glucose_targets": glucose_target_config,
                    "target_warning": target_warning,
                    "view_scope": view_scope,
                    "view_date": view_date,
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
        
        # Generate unique analysis ID for PDF export
        analysis_id = str(uuid.uuid4())
        
        # Extract results
        basic_metrics = results.get("basic_metrics", {})
        sensor_health = results.get("sensor_health", {})
        bolus_risks = results.get("bolus_risks", {})
        failure_prediction = results.get("failure_prediction", {})
        insights_raw = results.get("insights", [])
        
        # Store analysis context for PDF export
        _analysis_cache[analysis_id] = {
            "glucose_targets": glucose_target_config,
            "view_scope": view_scope,
            "view_date": view_date,
            "view_mode": view_mode,
            "basic_metrics": basic_metrics,
            "sensor_health": sensor_health,
            "bolus_risks": bolus_risks,
            "failure_prediction": failure_prediction,
            "insights": insights_raw,
            "processed_readings": processed_readings,
        }
        
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
                "view_scope": view_scope,
                "view_date": view_date,
                "view_mode": view_mode,
                "available_dates": available_dates,
                "analysis_id": analysis_id,
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
                "view_scope": "all",
                "view_date": None,
                "available_dates": [],
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


@app.get("/report/pdf/{analysis_id}")
async def download_pdf(analysis_id: str):
    """
    Generate and download a PDF summary report for a given analysis.
    
    Args:
        analysis_id: Unique identifier for the analysis (from previous /analyze call)
        
    Returns:
        PDF file as Response
    """
    # Guard against missing WeasyPrint
    if not WEASYPRINT_AVAILABLE:
        return pdf_error_html()
    
    # Look up stored context
    if analysis_id not in _analysis_cache:
        return Response(
            content="Analysis not found. Please run an analysis first.",
            status_code=404,
            media_type="text/plain"
        )
    
    context = _analysis_cache[analysis_id]
    
    # Prepare context for PDF template
    pdf_context = {
        "glucose_targets": context["glucose_targets"],
        "view_scope": context.get("view_scope", "all"),
        "view_date": context.get("view_date"),
        "view_mode": context.get("view_mode", "patient"),
        "basic_metrics": context["basic_metrics"],
        "sensor_health": context["sensor_health"],
        "bolus_risks": context["bolus_risks"],
        "failure_prediction": context["failure_prediction"],
        "insights": [
            {
                "title": insight.title,
                "severity": insight.severity,
                "message": insight.message,
                "details": insight.details,
            }
            for insight in context["insights"]
        ],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Render PDF template to HTML
    html_content = templates.get_template("pdf_report.html").render(pdf_context)
    
    # Generate PDF using WeasyPrint
    try:
        html = weasyprint.HTML(string=html_content)
        
        # Convert to PDF with error handling
        try:
            pdf_bytes = html.write_pdf()
        except Exception as e:
            print("PDF generation error:", e)
            return pdf_error_html()
        
        # Clean up cache entry (optional, or keep for a while)
        # del _analysis_cache[analysis_id]
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="drlayer_report_{analysis_id[:8]}.pdf"'
            }
        )
    except Exception as e:
        print("PDF generation error:", e)
        return pdf_error_html()


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
    view_scope: str = Form("all"),
    view_date: Optional[str] = Form(None),
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
        
        # Determine time window for filtering
        start_time = None
        end_time = None
        if view_scope == "day" and view_date:
            try:
                # Parse date and create time window (00:00 to 00:00 next day)
                selected_date = datetime.strptime(view_date, "%Y-%m-%d")
                start_time = selected_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = start_time + timedelta(days=1)
            except ValueError:
                # Invalid date format, use all data
                pass
        
        # Run pipeline
        try:
            processed_readings, results = run_pipeline(
                raw_readings,
                max_delta_mgdl=max_delta_mgdl,
                max_delta_minutes=max_delta_minutes,
                expected_interval_minutes=expected_interval_minutes,
                gap_threshold_minutes=gap_threshold_minutes,
                trust_correction_threshold=trust_correction_threshold,
                glucose_targets=glucose_target_config,
                start_time=start_time,
                end_time=end_time
            )
        except EmptyWindowError as e:
            return JSONResponse(
                status_code=400,
                content={"error": str(e)}
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
                "view_date": view_date,
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

