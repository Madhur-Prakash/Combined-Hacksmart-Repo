from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEOS_DIR = Path("videos")

def get_video_stream(file_path: Path, start: int = 0, end: int = None):
    file_size = file_path.stat().st_size
    
    if end is None:
        end = file_size - 1
    
    with open(file_path, "rb") as video:
        video.seek(start)
        data = video.read(end - start + 1)
        yield data

@app.get("/videos")
async def list_videos():
    videos = [f.name for f in VIDEOS_DIR.glob("*") if f.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']]
    return {"videos": videos}

@app.get("/stream/{filename}")
async def stream_video(filename: str, request: Request):
    file_path = VIDEOS_DIR / filename
    print("file_path:", file_path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    
    if range_header:
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if range_match[1] else file_size - 1
        
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Content-Type": "video/mp4",
        }
        
        return StreamingResponse(
            get_video_stream(file_path, start, end),
            status_code=206,
            headers=headers
        )
    
    return StreamingResponse(
        get_video_stream(file_path),
        media_type="video/mp4"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)