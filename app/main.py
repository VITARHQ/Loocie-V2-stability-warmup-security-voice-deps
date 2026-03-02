from fastapi import FastAPI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="LoocieAI V2 Master")

@app.get("/")
def root():
    return {"status": "ok", "app": "LoocieAI_V2_Master"}

@app.get("/health")
def health():
    return {"healthy": True}