# Byrock Hoof X-Ray Portal — Analytical Edition

An open-source equine hoof X-ray analytical platform that assesses morphological deviation from population norms to support lameness recognition. Built with a FastAPI backend (Python/OpenCV) and a medical-grade HTML5 frontend.

## 🚀 What's New (v2.0)

- **Backend API** (`FastAPI` + `SQLite`) for DICOM and image ingestion
- **Landmark Annotation Tool** — click anatomical points on the radiograph
- **Real-time Measurement Computation** — P3 rotation, HPA, palmar angle, sole depth, founder distance, laminar width, capsule deviation
- **EquiSim-derived Population Norms** — statistical deviation scoring (z-scores) with severity classification
- **Longitudinal Tracking** — per-horse scan timeline with trend analysis
- **Automated Clinical Recommendations** — severity-based veterinary guidance
- **Docker Compose** stack for one-command deployment

## 📁 Architecture

```
byrock-hoof-xray-portal/
├── backend/
│   ├── app/main.py          # FastAPI application
│   ├── requirements.txt
│   ├── Dockerfile
│   └── data/                # SQLite DB + uploads
├── frontend/
│   ├── index.html           # Upgraded analytical SPA
│   ├── legacy.html          # Original static visual aid
│   └── assets/              # Reference & trial images
├── docker-compose.yml
└── README.md
```

## 🐎 Quick Start

### Rotation Detection

The frontend includes a browser-side P3 rotation overlay for laminitis review:

- Click `↻` in the scan viewer to calculate rotation from placed landmarks, or attempt deterministic edge-based landmark detection when landmarks are absent.
- Click `◎` to load the synthetic 5.2° demo case for vet-facing validation without patient images.
- Default alert thresholds are 5° for coronal/DP views and 3° for sagittal/lateral views. Draft breeds use 7° coronal and 5° sagittal defaults.
- These thresholds are configurable in `frontend/image-processor.js` and require veterinary validation before clinical use.
- Production page unload clears patient/scan keys from browser storage as a privacy safeguard; this is not a HIPAA compliance claim.

```bash
npm test
npm run build
npm run deploy:ipfs
```

`deploy:ipfs` uses the local IPFS CLI and reports the generated CID. If the CLI or pinning credentials are missing, configure those first and rerun the command.

### Option A: Docker (Recommended)

```bash
# Start the full stack
docker-compose up --build

# Frontend: http://localhost:4179
# Backend API: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### Option B: Local Development

```bash
# 1. Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# 2. Frontend (static file server)
cd frontend
python -m http.server 4179
# Open http://localhost:4179
```

## 🔬 Measurement Pipeline

### Lateral View Landmarks

1. **Coronary Band** — top of hoof capsule
2. **Toe Tip** — distal point of hoof / P3 tip
3. **Extensor Process** — dorsal proximal P3
4. **P3 Tip** — distal tip of P3 (coffin bone)
5. **P3 Heel** — palmar aspect of P3 at heel
6. **Toe Ground** — ground contact under toe
7. **Heel Ground** — ground contact under heel
8. **Pastern Top** — proximal P2
9. **Pastern Bottom** — distal P2 (above coronary band)

### Computed Biometrics

| Metric | Normal Mean | SD | Clinical Significance |
|--------|-------------|----|-----------------------|
| P3 Rotation | 3° | 2° | Dorsal wall / P3 divergence |
| Hoof-Pastern Axis | 0° | 2.5° | Break in alignment |
| Palmar Angle | 5° | 2° | Load distribution |
| Sole Depth | 18 mm | 4 mm | P3 penetration risk |
| Founder Distance | 12 mm | 3 mm | Distal displacement (sinking) |
| Capsule Deviation | 0° | 3° | Coffin vs toe angle mismatch |
| Laminar Width | 11 mm | 1.5 mm | Laminar separation |

### Severity Scoring

- **Z-score < 1.0**: Normal
- **1.0 ≤ Z < 2.0**: Mild — monitor, dietary review
- **2.0 ≤ Z < 3.0**: Moderate — therapeutic shoeing, NSAID
- **Z ≥ 3.0**: Severe — emergency intervention, possible tenotomy

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/horses` | Create horse |
| GET | `/api/horses` | List horses |
| POST | `/api/scans` | Upload scan (JPEG/PNG/DICOM) |
| GET | `/api/scans` | List scans |
| POST | `/api/scans/{id}/landmarks` | Save landmarks |
| POST | `/api/scans/{id}/analyze` | Run analysis |
| GET | `/api/scans/{id}/measurements` | Get measurements |
| GET | `/api/scans/{id}/analysis` | Get analysis report |
| GET | `/api/norms` | Population norms |
| GET | `/api/horses/{id}/timeline` | Full timeline |

## 🧬 EquiSim Integration

This platform incorporates statistical shape modeling principles from [EquiSim](https://github.com/jvhoutte/equisim) (Van Houtte et al., Frontiers in Veterinary Science, 2021). While EquiSim is a 3D CT articulation model, we extracted the relevant biometric population distributions (toe angle, heel angle, palmar angle, capsule deviation, sole depth relationships) and adapted them for 2D radiograph deviation analysis.

## 📝 License

Open source. Built for veterinary research and equine welfare.

## ⚠️ Disclaimer

This tool is for **research and educational support** only. It does not replace veterinary diagnosis or clinical judgment. Always consult a qualified equine veterinarian for lameness assessment and treatment decisions.
