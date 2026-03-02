# LoocieAI_V2_Master

V2 Master Base Model (FastAPI skeleton) for Loocie’s SmartPackage Main Base Model (master).

## What this is
A clean, minimal FastAPI service that proves:
- the repo boots
- the API responds
- the structure is ready to expand (routers, services, models, etc.)

## Run (Mac)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Endpoints:
	•	GET /  → status payload
	•	GET /health → {“healthy”: true}
	•	Docs: http://127.0.0.1:8000/docs
Notes:
	•	.env is local-only (not committed)
	•	.venv/ is local-only (not committed)
	•	404 for favicon/apple-touch icons in logs is normal
Repo Structure:
 LoocieAI_V2_Master/
 app/
__init__.py
 main.py
 requirements.txt
 README.md
 .gitignore