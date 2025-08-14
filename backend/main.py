# main.py
import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Update import to match your actual filename
from chains import VideoGenerator, GenerationRequest, GenerationResponse

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_file = os.getenv("LOG_FILE", "video_generation.log")
logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title=os.getenv("APP_TITLE", "AI Video Generation API"),
    description=os.getenv("APP_DESCRIPTION", "Generate videos from text prompts using Stability AI"),
    version=os.getenv("APP_VERSION", "1.0.0"),
    debug=os.getenv("DEBUG", "false").lower() == "true"
)

# CORS configuration
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
cors_credentials = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware (optional)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

# Output directory
output_dir = os.getenv("OUTPUT_DIR", "./outputs")
os.makedirs(output_dir, exist_ok=True)

# Initialize video generator
try:
    video_generator = VideoGenerator.from_environment(output_dir=output_dir)
    logger.info("Video generator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize video generator: {e}")
    video_generator = None

# Health check
@app.get("/")
async def root():
    api_key_status = "configured" if video_generator and video_generator.api_key else "missing"
    return {
        "message": "AI Video Generation API is running",
        "status": "healthy" if video_generator else "degraded",
        "api_key_status": api_key_status,
        "model": "Stability AI Image-to-Video",
        "output_dir": output_dir,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "generate_video": "/generate-video",
            "list_videos": "/videos", 
            "download": "/download/{filename}",
            "validate_prompt": "/validate-prompt",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Video generation endpoint
@app.post("/generate-video", response_model=GenerationResponse)
async def generate_video(request: GenerationRequest):
    if video_generator is None:
        raise HTTPException(
            status_code=503, 
            detail="Video generator not initialized. Check Stability AI API key."
        )
    
    # Basic prompt validation
    prompt = request.prompt.strip()
    if len(prompt) < 10:
        raise HTTPException(
            status_code=400, 
            detail="Prompt must be at least 10 characters long"
        )
    
    try:
        logger.info(f"Received video generation request: {prompt[:100]}...")
        result = await video_generator.generate_video(request)
        
        if result.success:
            logger.info(f"Video generated successfully: {result.filename}")
        else:
            logger.error(f"Video generation failed: {result.message}")
            
        return result
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate video: {str(e)}"
        )

# Download endpoint
@app.get("/download/{filename}")
async def download_video(filename: str):
    # Security check for path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Additional security: only allow certain file extensions
    allowed_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    if not filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Check if it's actually in the output directory (additional security)
    if not os.path.abspath(file_path).startswith(os.path.abspath(output_dir)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    return FileResponse(
        file_path, 
        media_type="video/mp4", 
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Prompt validation endpoint
@app.post("/validate-prompt")
async def validate_prompt(request: dict):
    prompt = request.get("prompt", "")
    issues = []
    
    # Length validation
    if len(prompt.strip()) < 10:
        issues.append("Prompt must be at least 10 characters long")
    
    max_length = int(os.getenv("MAX_PROMPT_LENGTH", "500"))
    if len(prompt) > max_length:
        issues.append(f"Prompt is too long (max {max_length} characters)")
    
    # Content validation (basic)
    forbidden_words = os.getenv("FORBIDDEN_WORDS", "violence,explicit,harmful").split(",")
    found_forbidden = []
    for word in forbidden_words:
        if word.strip().lower() in prompt.lower():
            found_forbidden.append(word.strip())
    
    if found_forbidden:
        issues.append(f"Prompt contains inappropriate content: {', '.join(found_forbidden)}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "character_count": len(prompt),
        "word_count": len(prompt.split()),
        "max_length": max_length,
        "forbidden_words_found": found_forbidden
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    status = "healthy" if video_generator else "unhealthy"
    api_key_configured = bool(video_generator and video_generator.api_key)
    
    # Check disk space
    import shutil
    disk_usage = shutil.disk_usage(output_dir)
    free_space_gb = disk_usage.free / (1024**3)
    
    return {
        "status": status,
        "video_generator": "healthy" if video_generator else "unhealthy",
        "api_key_configured": api_key_configured,
        "output_dir_exists": os.path.exists(output_dir),
        "free_disk_space_gb": round(free_space_gb, 2),
        "timestamp": datetime.now().isoformat()
    }

# List generated videos endpoint
@app.get("/videos")
async def list_videos():
    try:
        video_files = []
        total_size = 0
        
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    file_path = os.path.join(output_dir, filename)
                    if os.path.isfile(file_path):  # Additional check
                        file_stats = os.stat(file_path)
                        file_size = file_stats.st_size
                        total_size += file_size
                        
                        video_files.append({
                            "filename": filename,
                            "size_bytes": file_size,
                            "size_mb": round(file_size / (1024*1024), 2),
                            "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                            "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                            "download_url": f"/download/{filename}"
                        })
        
        return {
            "videos": sorted(video_files, key=lambda x: x["created_at"], reverse=True),
            "total_count": len(video_files),
            "total_size_mb": round(total_size / (1024*1024), 2),
            "output_directory": output_dir
        }
        
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail="Failed to list videos")

# Delete video endpoint (optional)
@app.delete("/videos/{filename}")
async def delete_video(filename: str):
    # Security checks
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Additional security check
    if not os.path.abspath(file_path).startswith(os.path.abspath(output_dir)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    try:
        os.remove(file_path)
        logger.info(f"Deleted video file: {filename}")
        return {"message": f"Video {filename} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting video {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete video")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"API Key configured: {'Yes' if video_generator else 'No'}")
    
    uvicorn.run("main:app", host=host, port=port, reload=reload, log_level="info")