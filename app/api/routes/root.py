from fastapi import APIRouter
from fastapi.responses import FileResponse
import pathlib

router = APIRouter()

@router.get("/")
def root():
    ui_path = pathlib.Path("loocie_ui.html")
    if ui_path.exists():
        return FileResponse(ui_path)
    return {"message": "Loocie AI V2 Master - API Running"}
