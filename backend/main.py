import asyncio
import os
import shutil
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import uuid
from pydantic import BaseModel
import processing
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def cleanup_task():
    """Background task to delete old files."""
    while True:
        now = time.time()
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                # 5 minutes = 300 seconds
                if os.stat(file_path).st_mtime < now - 300:
                    try:
                        os.remove(file_path)
                        print(f"Deleted old file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_task())

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    
    contents = await file.read()
    
    # Process background
    bg_removed_bytes = processing.remove_bg(contents)
    
    # Extract base contours
    contours, shape = processing.extract_contours(bg_removed_bytes)
    
    image_id = str(uuid.uuid4())
    
    # Save the base data
    data_path = os.path.join(UPLOAD_DIR, f"{image_id}.pkl")
    with open(data_path, "wb") as f:
        pickle.dump({
            "bytes": bg_removed_bytes,
            "contours": contours,
            "width": shape[1],
            "height": shape[0]
        }, f)
        
    import base64
    b64_img = base64.b64encode(bg_removed_bytes).decode('utf-8')
    
    return {
        "id": image_id,
        "image": f"data:image/png;base64,{b64_img}",
        "width": shape[1],
        "height": shape[0]
    }

class ContourRequest(BaseModel):
    offset: float
    thickness: float
    corner_type: int # 1=round, 2=miter, 3=bevel
    smoothness: bool

@app.post("/api/contour/{image_id}")
async def get_contour(image_id: str, req: ContourRequest):
    data_path = os.path.join(UPLOAD_DIR, f"{image_id}.pkl")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Image not found or expired")
        
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        
    poly = processing.create_offset_contour(
        data["contours"], 
        req.offset, 
        req.corner_type, 
        req.smoothness
    )
    
    svg_path = processing.polygon_to_svg_path(poly, data["height"])
    
    # Save current poly for export
    with open(os.path.join(UPLOAD_DIR, f"{image_id}_poly.pkl"), "wb") as f:
        pickle.dump({
            "poly": poly,
            "thickness": req.thickness
        }, f)
        
    return {"path": svg_path}

@app.post("/api/export/{image_id}")
async def prepare_exports(image_id: str):
    data_path = os.path.join(UPLOAD_DIR, f"{image_id}.pkl")
    poly_path = os.path.join(UPLOAD_DIR, f"{image_id}_poly.pkl")
    
    if not os.path.exists(data_path) or not os.path.exists(poly_path):
        raise HTTPException(status_code=404, detail="Image not found or expired")
        
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(poly_path, "rb") as f:
        poly_data = pickle.load(f)
        
    exports = processing.generate_exports(
        data["bytes"], 
        poly_data["poly"], 
        poly_data["thickness"], 
        data["width"], 
        data["height"]
    )
    
    # Save exports
    for ext, content in exports.items():
        with open(os.path.join(UPLOAD_DIR, f"{image_id}.{ext}"), "wb") as f:
            f.write(content)
            
    return {
        "svg": f"/api/download/{image_id}/svg",
        "pdf": f"/api/download/{image_id}/pdf",
        "dxf": f"/api/download/{image_id}/dxf",
        "png": f"/api/download/{image_id}/png"
    }

@app.get("/api/download/{image_id}/{format}")
async def download(image_id: str, format: str):
    if format not in ["svg", "pdf", "dxf", "png"]:
        raise HTTPException(status_code=400, detail="Invalid format")
        
    file_path = os.path.join(UPLOAD_DIR, f"{image_id}.{format}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    media_types = {
        "svg": "image/svg+xml",
        "pdf": "application/pdf",
        "dxf": "application/dxf",
        "png": "image/png"
    }
    
    with open(file_path, "rb") as f:
        content = f.read()
        
    return Response(
        content=content, 
        media_type=media_types[format], 
        headers={"Content-Disposition": f"attachment; filename=contour_{image_id}.{format}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
