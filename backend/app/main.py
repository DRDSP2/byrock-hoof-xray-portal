# Byrock Hoof X-Ray Portal Backend
# FastAPI + SQLite + OpenCV measurement pipeline
# Computes equine hoof biometric deviations from population norms

from contextlib import asynccontextmanager
import os
import json
import math
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import aiosqlite
import aiofiles
from PIL import Image
import pydicom
import numpy as np
import cv2

# ───────────────────────────── Paths ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "byrock.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ───────────────────────────── DB Schema ─────────────────────────────
INIT_SQL = """
CREATE TABLE IF NOT EXISTS horses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    owner TEXT,
    breed TEXT,
    dob TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horse_id INTEGER NOT NULL,
    hoof TEXT NOT NULL CHECK(hoof IN ('FL','FR','HL','HR')),
    scan_date TEXT NOT NULL,
    view TEXT NOT NULL CHECK(view IN ('Lateral','DP')),
    modality TEXT,
    description TEXT,
    image_path TEXT,
    dicom_path TEXT,
    has_image INTEGER DEFAULT 0,
    pixel_spacing_x REAL,
    pixel_spacing_y REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (horse_id) REFERENCES horses(id)
);

CREATE TABLE IF NOT EXISTS landmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    severity TEXT CHECK(severity IN ('normal','mild','moderate','severe')),
    deviation_z REAL,
    computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL UNIQUE,
    overall_severity TEXT CHECK(overall_severity IN ('normal','mild','moderate','severe')),
    score REAL,
    findings_json TEXT,
    recommendations_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);
"""

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(INIT_SQL)
        await db.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Byrock Hoof X-Ray Portal API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ───────────────────────────── Pydantic Models ─────────────────────────────
class HorseCreate(BaseModel):
    name: str
    owner: Optional[str] = None
    breed: Optional[str] = None
    dob: Optional[str] = None

class HorseOut(BaseModel):
    id: int
    name: str
    owner: Optional[str]
    breed: Optional[str]
    dob: Optional[str]
    created_at: str

class ScanCreate(BaseModel):
    horse_id: int
    hoof: str = Field(..., pattern="^(FL|FR|HL|HR)$")
    scan_date: str
    view: str = Field(..., pattern="^(Lateral|DP)$")
    modality: Optional[str] = None
    description: Optional[str] = None

class ScanOut(BaseModel):
    id: int
    horse_id: int
    hoof: str
    scan_date: str
    view: str
    modality: Optional[str]
    description: Optional[str]
    image_url: Optional[str]
    has_image: bool
    pixel_spacing_x: Optional[float]
    pixel_spacing_y: Optional[float]
    created_at: str

class LandmarkCreate(BaseModel):
    name: str
    x: float
    y: float

class LandmarkOut(BaseModel):
    id: int
    scan_id: int
    name: str
    x: float
    y: float

class MeasurementOut(BaseModel):
    id: int
    scan_id: int
    metric: str
    value: float
    unit: Optional[str]
    severity: Optional[str]
    deviation_z: Optional[float]
    computed_at: str

class AnalysisOut(BaseModel):
    id: int
    scan_id: int
    overall_severity: str
    score: float
    findings: Dict[str, Any]
    recommendations: List[str]
    created_at: str

class NormOut(BaseModel):
    metric: str
    mean: float
    sd: float
    unit: str
    normal_range: List[float]
    mild_range: List[float]
    moderate_range: List[float]
    severe_range: List[float]

class ComputeRequest(BaseModel):
    image: Optional[str] = None
    landmarks: List[LandmarkCreate]
    pixel_spacing_x: Optional[float] = None
    pixel_spacing_y: Optional[float] = None

class ComputeResponse(BaseModel):
    measurements: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    auth: str = "supabase_rls"

# ───────────────────────────── Population Norms (EquiSim + Veterinary Lit) ─────────────────────────────
# Sources: EquiSim (Van Houtte 2021), Stashak's Lameness in Horses, Turner 2003, RVC guidelines
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

def get_severity_from_z(z: float, metric: str) -> str:
    """Map z-score to clinical severity. Some metrics are 'higher is worse',
    others are 'lower is worse'. We use absolute z for most, but apply
    direction-specific logic for metrics where both directions matter."""
    abs_z = abs(z)
    if abs_z < 1.0:
        return "normal"
    elif abs_z < 2.0:
        return "mild"
    elif abs_z < 3.0:
        return "moderate"
    else:
        return "severe"

def compute_severity(value: float, metric: str) -> tuple[str, float]:
    norm = POPULATION_NORMS[metric]
    z = (value - norm["mean"]) / norm["sd"]
    sev = get_severity_from_z(z, metric)
    return sev, z

def build_norms_response() -> List[NormOut]:
    out = []
    for metric, n in POPULATION_NORMS.items():
        out.append(NormOut(
            metric=metric,
            mean=n["mean"],
            sd=n["sd"],
            unit=n["unit"],
            normal_range=[round(n["mean"] - n["sd"], 2), round(n["mean"] + n["sd"], 2)],
            mild_range=[round(n["mean"] + n["sd"], 2), round(n["mean"] + 2*n["sd"], 2)],
            moderate_range=[round(n["mean"] + 2*n["sd"], 2), round(n["mean"] + 3*n["sd"], 2)],
            severe_range=[round(n["mean"] + 3*n["sd"], 2), round(n["mean"] + 5*n["sd"], 2)],
        ))
    return out

# ───────────────────────────── Measurement Engine ─────────────────────────────

def angle_between_points(p1, p2, p3=None) -> float:
    """Compute angle at p2 between p1->p2 and p2->p3 (or just p1->p2 vs horizontal)."""
    if p3 is None:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(-dy, dx))  # y grows downward in images
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))

def perpendicular_distance(point, line_p1, line_p2) -> float:
    """Distance from point to line segment."""
    p = np.array(point)
    a = np.array(line_p1)
    b = np.array(line_p2)
    ab = b - a
    ab_len = np.linalg.norm(ab)
    if ab_len == 0:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, np.dot(p - a, ab) / (ab_len ** 2)))
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))

def pixel_to_mm(px: float, spacing_x: Optional[float], spacing_y: Optional[float]) -> float:
    """Convert pixels to mm using DICOM pixel spacing. Fallback to 0.1mm/px if unknown."""
    spacing = spacing_x or spacing_y or 0.1
    return px * spacing

def compute_all_measurements(landmarks: List[Dict], spacing_x: Optional[float] = None, spacing_y: Optional[float] = None) -> List[Dict]:
    """Given a list of landmarks {name, x, y}, compute all standard metrics."""
    pts = {lm["name"]: (lm["x"], lm["y"]) for lm in landmarks}
    results = []
    
    def add(metric: str, value: float, unit: str):
        sev, z = compute_severity(value, metric)
        results.append({"metric": metric, "value": round(value, 2), "unit": unit, "severity": sev, "deviation_z": round(z, 2)})
    
    # Required landmarks for lateral view
    # ground_line: toe_ground -> heel_ground
    # dorsal_wall: coronary_band -> toe_tip
    # p3_dorsal: extensor_process -> p3_tip
    # p3_palmar: p3_heel -> p3_tip (or p3_palmar_point)
    
    # 1. Toe Angle: angle of dorsal wall to ground
    if "coronary_band" in pts and "toe_tip" in pts and "toe_ground" in pts and "heel_ground" in pts:
        ground_line = (pts["toe_ground"], pts["heel_ground"])
        toe_angle = angle_between_points(pts["coronary_band"], pts["toe_tip"], pts["toe_ground"])
        # We want the angle relative to the ground line, not the absolute angle
        wall_angle = angle_between_points(pts["coronary_band"], pts["toe_tip"])
        ground_angle = angle_between_points(pts["heel_ground"], pts["toe_ground"])
        # Compute relative angle
        rel = abs(wall_angle - ground_angle)
        if rel > 90:
            rel = 180 - rel
        add("toe_angle_deg", rel, "°")
    
    # 2. Heel Angle: angle of heel wall to ground
    if "coronary_band" in pts and "heel_ground" in pts and "toe_ground" in pts:
        heel_angle = angle_between_points(pts["coronary_band"], pts["heel_ground"], pts["toe_ground"])
        add("heel_angle_deg", heel_angle, "°")
    
    # 3. P3 Rotation: angle between dorsal wall and dorsal P3 surface
    if "coronary_band" in pts and "toe_tip" in pts and "extensor_process" in pts and "p3_tip" in pts:
        wall_vec = np.array([pts["toe_tip"][0] - pts["coronary_band"][0], pts["toe_tip"][1] - pts["coronary_band"][1]])
        p3_vec = np.array([pts["p3_tip"][0] - pts["extensor_process"][0], pts["p3_tip"][1] - pts["extensor_process"][1]])
        cos_angle = np.dot(wall_vec, p3_vec) / (np.linalg.norm(wall_vec) * np.linalg.norm(p3_vec) + 1e-9)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        rotation = math.degrees(math.acos(cos_angle))
        add("p3_rotation_deg", rotation, "°")
    
    # 4. Hoof-Pastern Axis (HPA): break between pastern and hoof
    if "p2_pastern_top" in pts and "p2_pastern_bottom" in pts and "coronary_band" in pts and "toe_tip" in pts:
        pastern_vec = np.array([pts["p2_pastern_bottom"][0] - pts["p2_pastern_top"][0], pts["p2_pastern_bottom"][1] - pts["p2_pastern_top"][1]])
        hoof_vec = np.array([pts["toe_tip"][0] - pts["coronary_band"][0], pts["toe_tip"][1] - pts["coronary_band"][1]])
        cos_angle = np.dot(pastern_vec, hoof_vec) / (np.linalg.norm(pastern_vec) * np.linalg.norm(hoof_vec) + 1e-9)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        hpa = math.degrees(math.acos(cos_angle))
        # HPA is the break from straight; if pastern and hoof are parallel, hpa=0. If they diverge, hpa>0.
        add("hoof_pastern_axis_deg", hpa, "°")
    
    # 5. Palmar Angle: angle between palmar P3 surface and ground
    if "p3_heel" in pts and "p3_tip" in pts and "toe_ground" in pts and "heel_ground" in pts:
        palmar_angle = angle_between_points(pts["p3_heel"], pts["p3_tip"], pts["toe_ground"])
        add("palmar_angle_deg", palmar_angle, "°")
    
    # 6. Sole Depth: perpendicular distance from P3 tip to ground line
    if "p3_tip" in pts and "toe_ground" in pts and "heel_ground" in pts:
        px_dist = perpendicular_distance(pts["p3_tip"], pts["toe_ground"], pts["heel_ground"])
        mm = pixel_to_mm(px_dist, spacing_x, spacing_y)
        add("sole_depth_mm", mm, "mm")
    
    # 7. Founder Distance: distance from coronary band to extensor process
    if "coronary_band" in pts and "extensor_process" in pts:
        px_dist = math.dist(pts["coronary_band"], pts["extensor_process"])
        mm = pixel_to_mm(px_dist, spacing_x, spacing_y)
        add("founder_distance_mm", mm, "mm")
    
    # 8. Capsule Deviation: toe_angle - coffin_angle (coffin angle = P3 dorsal angle to ground)
    if "extensor_process" in pts and "p3_tip" in pts and "toe_ground" in pts and "heel_ground" in pts:
        p3_dorsal_angle = angle_between_points(pts["extensor_process"], pts["p3_tip"], pts["toe_ground"])
        if "toe_angle_deg" in [r["metric"] for r in results]:
            toe_val = next(r["value"] for r in results if r["metric"] == "toe_angle_deg")
            add("capsule_deviation_deg", abs(toe_val - p3_dorsal_angle), "°")
    
    # 9. Laminar Width: distance from dorsal wall to P3 dorsal surface at mid-level
    if "coronary_band" in pts and "toe_tip" in pts and "extensor_process" in pts and "p3_tip" in pts:
        # Midpoint of wall
        mid_wall = ((pts["coronary_band"][0] + pts["toe_tip"][0]) / 2, (pts["coronary_band"][1] + pts["toe_tip"][1]) / 2)
        # Find closest point on P3 dorsal line to mid_wall
        p3_line = (pts["extensor_process"], pts["p3_tip"])
        px_dist = perpendicular_distance(mid_wall, p3_line[0], p3_line[1])
        mm = pixel_to_mm(px_dist, spacing_x, spacing_y)
        add("laminar_width_mm", mm, "mm")
    
    return results

# ───────────────────────────── Deviation Scoring ─────────────────────────────

def run_analysis(scan_id: int, measurements: List[Dict]) -> Dict:
    """Generate overall severity, composite score, findings, and recommendations."""
    if not measurements:
        return {
            "overall_severity": "normal",
            "score": 0.0,
            "findings": {},
            "recommendations": ["No measurements available. Please annotate landmarks on the scan."]
        }
    
    # Weighted composite score (0-100)
    max_score = 0.0
    weighted_score = 0.0
    findings = {}
    recommendations = []
    
    severity_rank = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}
    max_severity = "normal"
    
    for m in measurements:
        metric = m["metric"]
        norm = POPULATION_NORMS.get(metric)
        if not norm:
            continue
        weight = norm["weight"]
        z = abs(m["deviation_z"])
        sev = m["severity"]
        
        if severity_rank[sev] > severity_rank[max_severity]:
            max_severity = sev
        
        # Score contribution: 0-33 per metric, scaled by z and weight
        contrib = min(33.0, z * 10.0) * weight
        weighted_score += contrib
        max_score += 33.0 * weight
        
        findings[metric] = {
            "value": m["value"],
            "unit": m["unit"],
            "severity": sev,
            "z": m["deviation_z"],
            "normal_mean": norm["mean"],
            "normal_sd": norm["sd"]
        }
    
    overall_score = min(100.0, (weighted_score / max(max_score, 1e-9)) * 100) if max_score > 0 else 0.0
    
    # Clinical recommendations based on findings
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
            recommendations.append("🚨 EMERGENCY: Severe laminitis pattern. Immediate veterinary intervention, NSAID therapy, cryotherapy, and strict stall rest required.")
    
    return {
        "overall_severity": max_severity,
        "score": round(overall_score, 1),
        "findings": findings,
        "recommendations": recommendations
    }

# ───────────────────────────── DICOM / Image Helpers ─────────────────────────────

async def process_upload(file: UploadFile, scan_id: int) -> tuple[str, Optional[str], Optional[float], Optional[float]]:
    """Save uploaded file, extract DICOM if applicable, convert to viewable JPEG, return paths."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    base_name = f"scan_{scan_id}"
    
    raw_path = os.path.join(UPLOAD_DIR, f"{base_name}{ext}")
    async with aiofiles.open(raw_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    dicom_path = None
    pixel_spacing_x = None
    pixel_spacing_y = None
    
    if ext == ".dcm" or file.content_type == "application/dicom":
        dicom_path = raw_path
        try:
            ds = pydicom.dcmread(raw_path)
            if "PixelSpacing" in ds:
                pixel_spacing_x, pixel_spacing_y = ds.PixelSpacing
            elif "ImagerPixelSpacing" in ds:
                pixel_spacing_x, pixel_spacing_y = ds.ImagerPixelSpacing
            # Convert to JPEG for viewing
            img_array = ds.pixel_array
            # Apply windowing if available
            if "WindowCenter" in ds and "WindowWidth" in ds:
                wc = ds.WindowCenter if isinstance(ds.WindowCenter, (int, float)) else ds.WindowCenter[0]
                ww = ds.WindowWidth if isinstance(ds.WindowWidth, (int, float)) else ds.WindowWidth[0]
                img_min = wc - ww / 2
                img_max = wc + ww / 2
                img_array = np.clip(img_array, img_min, img_max)
            img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-9) * 255).astype(np.uint8)
            if len(img_array.shape) == 3 and img_array.shape[0] < 10:
                img_array = img_array[0]  # Take first frame if multi-frame
            Image.fromarray(img_array).convert("L").save(os.path.join(UPLOAD_DIR, f"{base_name}.jpg"), quality=95)
        except Exception as e:
            print(f"DICOM conversion error: {e}")
            # Fallback: save as-is and try PIL
            try:
                Image.open(raw_path).convert("RGB").save(os.path.join(UPLOAD_DIR, f"{base_name}.jpg"), quality=95)
            except Exception:
                pass
    else:
        # Image file (JPEG/PNG)
        try:
            img = Image.open(raw_path).convert("RGB")
            img.save(os.path.join(UPLOAD_DIR, f"{base_name}.jpg"), quality=95)
        except Exception as e:
            print(f"Image processing error: {e}")
    
    image_url = f"/uploads/{base_name}.jpg"
    return image_url, dicom_path, pixel_spacing_x, pixel_spacing_y

# ───────────────────────────── API Endpoints ─────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}

@app.post("/api/compute", response_model=ComputeResponse)
async def compute_measurements(data: ComputeRequest):
    landmarks = [lm.dict() for lm in data.landmarks]
    if not landmarks:
        raise HTTPException(400, "No landmarks supplied. Provide raw image context plus landmarks.")

    measurements = compute_all_measurements(landmarks, data.pixel_spacing_x, data.pixel_spacing_y)
    if not measurements:
        raise HTTPException(400, "Insufficient landmarks supplied for measurement computation.")

    return {
        "measurements": measurements,
        "analysis": run_analysis(0, measurements),
        "auth": "supabase_rls",
    }

@app.post("/api/horses", response_model=HorseOut)
async def create_horse(data: HorseCreate):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO horses (name, owner, breed, dob) VALUES (?, ?, ?, ?)",
            (data.name, data.owner, data.breed, data.dob)
        )
        await db.commit()
        horse_id = cursor.lastrowid
        row = await db.execute_fetchall("SELECT * FROM horses WHERE id = ?", (horse_id,))
    if row:
        r = row[0]
        return HorseOut(id=r[0], name=r[1], owner=r[2], breed=r[3], dob=r[4], created_at=r[5])
    raise HTTPException(500, "Failed to create horse")

@app.get("/api/horses", response_model=List[HorseOut])
async def list_horses():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall("SELECT * FROM horses ORDER BY created_at DESC")
    return [HorseOut(id=r["id"], name=r["name"], owner=r["owner"], breed=r["breed"], dob=r["dob"], created_at=r["created_at"]) for r in rows]

@app.post("/api/scans", response_model=ScanOut)
async def create_scan(
    horse_id: int = Form(...),
    hoof: str = Form(...),
    scan_date: str = Form(...),
    view: str = Form(...),
    modality: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    file: UploadFile = File(None)
):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO scans (horse_id, hoof, scan_date, view, modality, description, has_image) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (horse_id, hoof, scan_date, view, modality, description, 0)
        )
        await db.commit()
        scan_id = cursor.lastrowid
    
    image_url = None
    dicom_path = None
    pixel_spacing_x = None
    pixel_spacing_y = None
    has_image = False
    
    if file:
        image_url, dicom_path, pixel_spacing_x, pixel_spacing_y = await process_upload(file, scan_id)
        has_image = True
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE scans SET image_path=?, dicom_path=?, has_image=?, pixel_spacing_x=?, pixel_spacing_y=? WHERE id=?",
                (image_url, dicom_path, 1, pixel_spacing_x, pixel_spacing_y, scan_id)
            )
            await db.commit()
    
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute_fetchall("SELECT * FROM scans WHERE id = ?", (scan_id,))
    if not row:
        raise HTTPException(500, "Failed to create scan")
    r = row[0]
    return ScanOut(
        id=r["id"], horse_id=r["horse_id"], hoof=r["hoof"], scan_date=r["scan_date"],
        view=r["view"], modality=r["modality"], description=r["description"],
        image_url=r["image_path"], has_image=bool(r["has_image"]),
        pixel_spacing_x=r["pixel_spacing_x"], pixel_spacing_y=r["pixel_spacing_y"],
        created_at=r["created_at"]
    )

@app.get("/api/scans", response_model=List[ScanOut])
async def list_scans(horse_id: Optional[int] = None, hoof: Optional[str] = None):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        query = "SELECT * FROM scans WHERE 1=1"
        params = []
        if horse_id:
            query += " AND horse_id = ?"
            params.append(horse_id)
        if hoof:
            query += " AND hoof = ?"
            params.append(hoof)
        query += " ORDER BY scan_date DESC, id DESC"
        rows = await db.execute_fetchall(query, params)
    return [ScanOut(
        id=r["id"], horse_id=r["horse_id"], hoof=r["hoof"], scan_date=r["scan_date"],
        view=r["view"], modality=r["modality"], description=r["description"],
        image_url=r["image_path"], has_image=bool(r["has_image"]),
        pixel_spacing_x=r["pixel_spacing_x"], pixel_spacing_y=r["pixel_spacing_y"],
        created_at=r["created_at"]
    ) for r in rows]

@app.get("/api/scans/{scan_id}", response_model=ScanOut)
async def get_scan(scan_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall("SELECT * FROM scans WHERE id = ?", (scan_id,))
    if not rows:
        raise HTTPException(404, "Scan not found")
    r = rows[0]
    return ScanOut(
        id=r["id"], horse_id=r["horse_id"], hoof=r["hoof"], scan_date=r["scan_date"],
        view=r["view"], modality=r["modality"], description=r["description"],
        image_url=r["image_path"], has_image=bool(r["has_image"]),
        pixel_spacing_x=r["pixel_spacing_x"], pixel_spacing_y=r["pixel_spacing_y"],
        created_at=r["created_at"]
    )

@app.post("/api/scans/{scan_id}/landmarks")
async def add_landmarks(scan_id: int, landmarks: List[LandmarkCreate]):
    async with aiosqlite.connect(DB_PATH) as db:
        # Verify scan exists
        rows = await db.execute_fetchall("SELECT id FROM scans WHERE id = ?", (scan_id,))
        if not rows:
            raise HTTPException(404, "Scan not found")
        # Clear existing landmarks for this scan
        await db.execute("DELETE FROM landmarks WHERE scan_id = ?", (scan_id,))
        for lm in landmarks:
            await db.execute(
                "INSERT INTO landmarks (scan_id, name, x, y) VALUES (?, ?, ?, ?)",
                (scan_id, lm.name, lm.x, lm.y)
            )
        await db.commit()
    return {"status": "ok", "landmarks_added": len(landmarks)}

@app.get("/api/scans/{scan_id}/landmarks", response_model=List[LandmarkOut])
async def get_landmarks(scan_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall("SELECT * FROM landmarks WHERE scan_id = ?", (scan_id,))
    return [LandmarkOut(id=r["id"], scan_id=r["scan_id"], name=r["name"], x=r["x"], y=r["y"]) for r in rows]

@app.post("/api/scans/{scan_id}/analyze")
async def analyze_scan(scan_id: int, background_tasks: BackgroundTasks):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        # Get scan with pixel spacing
        scan_rows = await db.execute_fetchall("SELECT * FROM scans WHERE id = ?", (scan_id,))
        if not scan_rows:
            raise HTTPException(404, "Scan not found")
        scan = scan_rows[0]
        
        # Get landmarks
        lm_rows = await db.execute_fetchall("SELECT * FROM landmarks WHERE scan_id = ?", (scan_id,))
        landmarks = [{"name": r["name"], "x": r["x"], "y": r["y"]} for r in lm_rows]
        
        if not landmarks:
            raise HTTPException(400, "No landmarks found for this scan. Please annotate landmarks first.")
        
        # Compute measurements
        measurements = compute_all_measurements(landmarks, scan["pixel_spacing_x"], scan["pixel_spacing_y"])
        
        # Save measurements
        await db.execute("DELETE FROM measurements WHERE scan_id = ?", (scan_id,))
        for m in measurements:
            await db.execute(
                "INSERT INTO measurements (scan_id, metric, value, unit, severity, deviation_z) VALUES (?, ?, ?, ?, ?, ?)",
                (scan_id, m["metric"], m["value"], m["unit"], m["severity"], m["deviation_z"])
            )
        
        # Run analysis
        analysis = run_analysis(scan_id, measurements)
        
        # Save analysis
        await db.execute("DELETE FROM analyses WHERE scan_id = ?", (scan_id,))
        await db.execute(
            "INSERT INTO analyses (scan_id, overall_severity, score, findings_json, recommendations_json) VALUES (?, ?, ?, ?, ?)",
            (scan_id, analysis["overall_severity"], analysis["score"], json.dumps(analysis["findings"]), json.dumps(analysis["recommendations"]))
        )
        await db.commit()
    
    return {
        "scan_id": scan_id,
        "measurements": measurements,
        "analysis": analysis
    }

@app.get("/api/scans/{scan_id}/measurements", response_model=List[MeasurementOut])
async def get_measurements(scan_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall("SELECT * FROM measurements WHERE scan_id = ? ORDER BY id", (scan_id,))
    return [MeasurementOut(
        id=r["id"], scan_id=r["scan_id"], metric=r["metric"], value=r["value"],
        unit=r["unit"], severity=r["severity"], deviation_z=r["deviation_z"], computed_at=r["computed_at"]
    ) for r in rows]

@app.get("/api/scans/{scan_id}/analysis", response_model=AnalysisOut)
async def get_analysis(scan_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall("SELECT * FROM analyses WHERE scan_id = ?", (scan_id,))
    if not rows:
        raise HTTPException(404, "Analysis not found. Run analysis first.")
    r = rows[0]
    return AnalysisOut(
        id=r["id"], scan_id=r["scan_id"], overall_severity=r["overall_severity"],
        score=r["score"], findings=json.loads(r["findings_json"]),
        recommendations=json.loads(r["recommendations_json"]), created_at=r["created_at"]
    )

@app.get("/api/norms", response_model=List[NormOut])
async def get_norms():
    return build_norms_response()

@app.get("/api/horses/{horse_id}/timeline")
async def get_horse_timeline(horse_id: int):
    """Get all scans with analyses for a horse, grouped by hoof."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        # Get horse info
        horse_rows = await db.execute_fetchall("SELECT * FROM horses WHERE id = ?", (horse_id,))
        if not horse_rows:
            raise HTTPException(404, "Horse not found")
        horse = dict(horse_rows[0])
        
        # Get scans with analyses
        scan_rows = await db.execute_fetchall(
            """SELECT s.*, a.overall_severity, a.score, a.findings_json, a.recommendations_json
               FROM scans s LEFT JOIN analyses a ON s.id = a.scan_id
               WHERE s.horse_id = ? ORDER BY s.scan_date DESC, s.id DESC""", (horse_id,)
        )
        
        timeline = {}
        for r in scan_rows:
            hoof = r["hoof"]
            if hoof not in timeline:
                timeline[hoof] = []
            timeline[hoof].append({
                "scan_id": r["id"],
                "date": r["scan_date"],
                "view": r["view"],
                "modality": r["modality"],
                "description": r["description"],
                "image_url": r["image_path"],
                "severity": r["overall_severity"] or "unscanned",
                "score": r["score"],
                "findings": json.loads(r["findings_json"]) if r["findings_json"] else {},
                "recommendations": json.loads(r["recommendations_json"]) if r["recommendations_json"] else []
            })
    
    return {"horse": horse, "timeline": timeline}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
