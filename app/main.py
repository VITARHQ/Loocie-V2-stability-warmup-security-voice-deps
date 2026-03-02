from fastapi import FastAPI
from dotenv import load_dotenv

from app.api.router import api_router  # central router

load_dotenv()

app = FastAPI(title="LoocieAI V2 Master")

# Attach the central API router (routes live in app/api/routes/*)
app.include_router(api_router)