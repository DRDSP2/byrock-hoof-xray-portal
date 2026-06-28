# Byrock Hoof X-Ray Portal — Stateless Compute Microservice
# FastAPI + OpenCV. No persistence. Receives landmarks, returns measurements.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import math
import os
from typing import Optional, List, Dict

# ───────────────────────────── Environment Config ─────────────────────────────
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:4179")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", FRONTEND_ORIGIN).split(",")
ENV = os.environ.get("ENV", "development")

app = FastAPI(title="Byrock Compute Service", version="2.1.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "ALLOWALL"
    response.headers["Content-Security-Policy"] = f"frame-ancestors 'self' {FRONTEND_ORIGIN}"
    return response

# ───────────────────────────── Pydantic Models ─────────────────────────────
class LandmarkIn(BaseModel):
    name: str
    x: float = Field(..., ge=0, le=100)
    y: float = Field(..., ge=0, le=100)

class AnalyzeRequest(BaseModel):
    landmarks: List[LandmarkIn] = Field(..., min_length=1)
    pixel_spacing_x: Optional[float] = None
    pixel_spacing_y: Optional[float] = None

class MeasurementOut(BaseModel):
    metric: str
    value: float
    unit: str
    severity: str
    deviation_z: float

class AnalysisOut(BaseModel):
    overall_severity: str
    score: float
    findings: Dict[str, dict]
    recommendations: List[str]

class AnalyzeResponse(BaseModel):
    measurements: List[MeasurementOut]
    analysis: AnalysisOut

# ───────────────────────────── Population Norms ─────────────────────────────
POPULATION_NORMS = {
    "p3_rotation_deg":      {"mean": 3.0,  "sd": 2.0,  "unit": "°",  "weight": 1.0},
    "hoof_pastern_axis_deg":{"mean": 0.0,  "sd": 2.5,  "unit": "°",  "weight": 0.9},
    "palmar_angle_deg":     {"mean": 5.0,  "sd": 2.0,  "unit": "°",  "weight": 0.8},
    "sole_depth_mm":        {"mean": 18.0, "sd": 4.0,  "unit": "mm", "weight": 1.0},
    "founder_distance_mm":  {"mean": 12.0, "sd": 3.0,  "unit": "mm", "weight": 0.9},
    "toe_angle_deg":        {"mean": 50.0, "sd": 5.0,  "unit": "°",  "weight": 0.6},
    "heel_angle_deg":       {"mean": 55.0, "sd": 5.0,  "unit": "°",  "weight": 0.6},
    "capsule_deviation_deg":{"mean": 0.0,  "sd": 3.0,  "unit": "°",  "weight": 0.7},
    "laminar_width_mm":     {"mean": 11.0, "sd": 1.5,  "unit": "mm", "weight": 0.8},
}

def get_severity_from_z(z: float) -> str:
    abs_z = abs(z)
    if abs_z < 1.0: return "normal"
    elif abs_z < 2.0: return "mild"
    elif abs_z < 3.0: return "moderate"
    return "severe"

def compute_severity(value: float, metric: str) -> tuple[str, float]:
    norm = POPULATION_NORMS[metric]
    z = (value - norm["mean"]) / norm["sd"]
    return get_severity_from_z(z), z

# ───────────────────────────── Geometry Engine ─────────────────────────────
def angle_between_points(p1, p2, p3=None) -> float:
    if p3 is None:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(-dy, dx))
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))

def perpendicular_distance(point, line_p1, line_p2) -> float:
    p = np.array(point)
    a = np.array(line_p1)
    b = np.array(line_p2)
    ab = b - a
    ab_len = np.linalg.norm(ab)
    if ab_len == 0: return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, np.dot(p - a, ab) / (ab_len ** 2)))
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))

def pixel_to_mm(px: float, spacing_x: Optional[float], spacing_y: Optional[float]) -> float:
    spacing = spacing_x or spacing_y or 0.1
    return px * spacing

def compute_all_measurements(landmarks: List[Dict], spacing_x: Optional[float] = None, spacing_y: Optional[float] = None) -> List[Dict]:
    pts = {lm["name"]: (lm["x"], lm["y"]) for lm in landmarks}
    results = []
    
    def add(metric: str, value: float, unit: str):
        sev, z = compute_severity(value, metric)
        results.append({"metric": metric, "value": round(value, 2), "unit": unit, "severity": sev, "deviation_z": round(z, 2)})
    
    # 1. Toe Angle
    if "coronary_band" in pts and "toe_tip" in pts and "toe_ground" in pts and "heel_ground" in pts:
        wall_angle = angle_between_points(pts["coronary_band"], pts["toe_tip"])
        ground_angle = angle_between_points(pts["heel_ground"], pts["toe_ground"])
        rel = abs(wall_angle - ground_angle)
        if rel > 90: rel = 180 - rel
        add("toe_angle_deg", rel, "°")
    
    # 2. Heel Angle
    if "coronary_band" in pts and "heel_ground" in pts and "toe_ground" in pts:
        heel_angle = angle_between_points(pts["coronary_band"], pts["heel_ground"], pts["toe_ground"])
        add("heel_angle_deg", heel_angle, "°")
    
    # 3. P3 Rotation
    if "coronary_band" in pts and "toe_tip" in pts and "extensor_process" in pts and "p3_tip" in pts:
        wall_vec = np.array([pts["toe_tip"][0] - pts["coronary_band"][0], pts["toe_tip"][1] - pts["coronary_band"][1]])
        p3_vec = np.array([pts["p3_tip"][0] - pts["extensor_process"][0], pts["p3_tip"][1] - pts["extensor_process"][1]])
        cos_angle = np.dot(wall_vec, p3_vec) / (np.linalg.norm(wall_vec) * np.linalg.norm(p3_vec) + 1e-9)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        rotation = math.degrees(math.acos(cos_angle))
        add("p3_rotation_deg", rotation, "°")
    
    # 4. Hoof-Pastern Axis
    if "p2_pastern_top" in pts and "p2_pastern_bottom" in pts and "coronary_band" in pts and "toe_tip" in pts:
        pastern_vec = np.array([pts["p2_pastern_bottom"][0] - pts["p2_pastern_top"][0], pts["p2_pastern_bottom"][1] - pts["p2_pastern_top"][1]])
        hoof_vec = np.array([pts["toe_tip"][0] - pts["coronary_band"][0], pts["toe_tip"][1] - pts["coronary_band"][1]])
        cos_angle = np.dot(pastern_vec, hoof_vec) / (np.linalg.norm(pastern_vec) * np.linalg.norm(hoof_vec) + 1e-9)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        hpa = math.degrees(math.acos(cos_angle))
        add("hoof_pastern_axis_deg", hpa, "°")
    
    # 5. Palmar Angle
    if "p3_heel" in pts and "p3_tip" in pts and "toe_ground" in pts and "heel_ground" in pts:
        palmar_angle = angle_between_points(pts["p3_heel"], pts["p3_tip"], pts["toe_ground"])
        add("palmar_angle_deg", palmar_angle, "°")
    
    # 6. Sole Depth
    if "p3_tip" in pts and "toe_ground" in pts and "heel_ground" in pts:
        px_dist = perpendicular_distance(pts["p3_tip"], pts["toe_ground"], pts["heel_ground"])
        mm = pixel_to_mm(px_dist, spacing_x, spacing_y)
        add("sole_depth_mm", mm, "mm")
    
    # 7. Founder Distance
    if "coronary_band" in pts and "extensor_process" in pts:
        px_dist = math.dist(pts["coronary_band"], pts["extensor_process"])
        mm = pixel_to_mm(px_dist, spacing_x, spacing_y)
        add("founder_distance_mm", mm, "mm")
    
    # 8. Capsule Deviation
    if "extensor_process" in pts and "p3_tip" in pts and "toe_ground" in pts and "heel_ground" in pts:
        p3_dorsal_angle = angle_between_points(pts["extensor_process"], pts["p3_tip"], pts["toe_ground"])
        toe_val = next((r["value"] for r in results if r["metric"] == "toe_angle_deg"), None)
        if toe_val is not None:
            add("capsule_deviation_deg", abs(toe_val - p3_dorsal_angle), "°")
    
    # 9. Laminar Width
    if "coronary_band" in pts and "toe_tip" in pts and "extensor_process" in pts and "p3_tip" in pts:
        mid_wall = ((pts["coronary_band"][0] + pts["toe_tip"][0]) / 2, (pts["coronary_band"][1] + pts["toe_tip"][1]) / 2)
        px_dist = perpendicular_distance(mid_wall, pts["extensor_process"], pts["p3_tip"])
        mm = pixel_to_mm(px_dist, spacing_x, spacing_y)
        add("laminar_width_mm", mm, "mm")
    
    return results

def run_analysis(measurements: List[Dict]) -> Dict:
    if not measurements:
        return {"overall_severity": "normal", "score": 0.0, "findings": {}, "recommendations": ["No measurements available."], "measurements": []}
    
    max_score = 0.0
    weighted_score = 0.0
    findings = {}
    severity_rank = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}
    max_severity = "normal"
    
    for m in measurements:
        metric = m["metric"]
        norm = POPULATION_NORMS.get(metric)
        if not norm: continue
        weight = norm["weight"]
        z = abs(m["deviation_z"])
        sev = m["severity"]
        if severity_rank[sev] > severity_rank[max_severity]: max_severity = sev
        contrib = min(33.0, z * 10.0) * weight
        weighted_score += contrib
        max_score += 33.0 * weight
        findings[metric] = {"value": m["value"], "unit": m["unit"], "severity": sev, "z": m["deviation_z"], "normal_mean": norm["mean"], "normal_sd": norm["sd"]}
    
    overall_score = min(100.0, (weighted_score / max(max_score, 1e-9)) * 100) if max_score > 0 else 0.0
    
    recommendations = []
    rot = next((m for m in measurements if m["metric"] == "p3_rotation_deg"), None)
    sole = next((m for m in measurements if m["metric"] == "sole_depth_mm"), None)
    pal = next((m for m in measurements if m["metric"] == "palmar_angle_deg"), None)
    founder = next((m for m in measurements if m["metric"] == "founder_distance_mm"), None)
    
    if max_severity == "normal":
        recommendations.append("All parameters within normal range. Continue routine monitoring and standard farriery.")
    else:
        if rot and rot["severity"] in ("moderate", "severe"):
            recommendations.append(f"P3 rotation {rot['value']}° indicates significant displacement. Consider therapeutic shoeing to reduce breakover and support heel.")
        if sole and sole["severity"] in ("moderate", "severe"):
            recommendations.append(f"Sole depth {sole['value']}mm is critically reduced. Protect sole with pads/shoes; avoid hard ground. P3 penetration risk.")
        if pal and pal["severity"] in ("moderate", "severe"):
            recommendations.append(f"Palmar angle {pal['value']}° indicates abnormal loading. Heel support or wedge pads may be indicated.")
        if founder and founder["severity"] in ("moderate", "severe"):
            recommendations.append(f"Founder distance {founder['value']}mm suggests distal displacement (sinking). Deep digital flexor tenotomy may be needed in severe cases.")
        if max_severity == "mild":
            recommendations.append("Mild deviation detected. Monitor closely with serial radiographs in 4-6 weeks. Review diet and exercise.")
        if max_severity == "severe":
            recommendations.append("EMERGENCY: Severe laminitis pattern. Immediate veterinary intervention, NSAID therapy, cryotherapy, and strict stall rest required.")
    
    return {"overall_severity": max_severity, "score": round(overall_score, 1), "findings": findings, "recommendations": recommendations}

# ───────────────────────────── API Endpoints ─────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.1.0", "env": ENV, "timestamp": "2025-06-26T00:00:00Z"}

@app.get("/api/config")
async def get_config():
    return {"api_base": "/", "env": ENV}

@app.get("/api/norms")
async def get_norms():
    out = []
    for metric, n in POPULATION_NORMS.items():
        out.append({
            "metric": metric,
            "mean": n["mean"], "sd": n["sd"], "unit": n["unit"],
            "normal_range": [round(n["mean"] - n["sd"], 2), round(n["mean"] + n["sd"], 2)],
            "mild_range": [round(n["mean"] + n["sd"], 2), round(n["mean"] + 2*n["sd"], 2)],
            "moderate_range": [round(n["mean"] + 2*n["sd"], 2), round(n["mean"] + 3*n["sd"], 2)],
            "severe_range": [round(n["mean"] + 3*n["sd"], 2), round(n["mean"] + 5*n["sd"], 2)],
        })
    return out

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Stateless compute endpoint. Accepts landmarks, returns measurements and analysis."""
    landmarks = [{"name": lm.name, "x": lm.x, "y": lm.y} for lm in req.landmarks]
    measurements = compute_all_measurements(landmarks, req.pixel_spacing_x, req.pixel_spacing_y)
    analysis = run_analysis(measurements)
    return AnalyzeResponse(
        measurements=[MeasurementOut(**m) for m in measurements],
        analysis=AnalysisOut(**analysis),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
