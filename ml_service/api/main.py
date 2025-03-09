from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import logging
import traceback
import os
import psutil
import torch
from typing import Optional
from model.generator import MusicGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Harmony ML Service API",
    description="API for generating music accompaniments using ML models",
    version="0.1.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the generator
try:
    generator = MusicGenerator()
    logger.info("MusicGenerator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MusicGenerator: {str(e)}")
    logger.error(traceback.format_exc())
    raise

def get_system_info():
    """Get current system resource information for debugging"""
    memory = psutil.virtual_memory()
    return {
        "available_memory_gb": round(memory.available / (1024**3), 2),
        "used_memory_percent": memory.percent,
        "cpu_percent": psutil.cpu_percent(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

def cleanup_temp_files(temp_dir: Path):
    """Clean up temporary files after response has been sent"""
    try:
        if temp_dir.exists():
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
            logger.info(f"Successfully cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory {temp_dir}: {str(e)}")

@app.get("/health")
async def health_check():
    """API health check endpoint that returns system information"""
    try:
        sys_info = get_system_info()
        return {
            "status": "ok",
            "system_info": sys_info
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/generate")
async def generate_music(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    semantic_steps: int = Form(50),  # Required parameter
    duration: Optional[int] = Form(10),  # Default is 20 seconds
    time_steps_factor: Optional[int] = Form(40),
    temperature: Optional[float] = Form(0.85),
    prompt: Optional[str] = Form("Add a bass line, harmony, and drums, and remove the piano melody to leave space for the soloist."),
    save_for_eval: Optional[bool] = Form(False)
):
    """
    Generate music based on an input audio file and various parameters.
    
    Parameters:
    - audio_file: Input audio file to transform. Should be a mono or stereo WAV file.
    
    - semantic_steps: Number of semantic steps for token generation. Controls the complexity
      and quality of the semantic understanding of the input audio. Higher values (6-10) may
      produce better quality output but will increase processing time. Default is 6, range 1-10 recommended.
    
    - duration: Target duration of the output audio in seconds. Default is 20 seconds.
      IMPORTANT: Values over 35-40 seconds may cause generation to fail due to memory constraints.
      For optimal results, stay within 10-35 seconds.
    
    - time_steps_factor: Multiplier that controls temporal resolution in the coarse stage.
      Higher values create more detailed time steps but require more memory. 
      Default is 6. Works together with duration to determine max_time_steps.
    
    - temperature: Controls randomness of generation. Values closer to 0 produce more predictable output,
      while higher values (up to 1.5) introduce more creativity and variation.
      Recommended range: 0.5-1.0. Default is 0.85.
    
    - prompt: Text description to guide the audio generation. Describes the style, instruments,
      or qualities desired in the generated audio. Default is "Diverse kinds of instrument and richness".
    
    - save_for_eval: Whether to save the input/output files and metadata for later evaluation.
      Useful for debugging and quality assessment. Default is False.
    
    Returns:
    - FileResponse: The generated audio file in WAV format
    """
    request_id = str(uuid.uuid4())
    temp_dir = Path("temp") / request_id
    logger.info(f"Starting audio generation with request ID: {request_id}")
    logger.info(f"Parameters: semantic_steps={semantic_steps}, duration={duration}, time_steps_factor={time_steps_factor}, temperature={temperature}")
    
    # Log system resources before processing
    sys_info = get_system_info()
    logger.info(f"System resources before processing: {sys_info}")
    
    # Validate parameters
    if semantic_steps <= 0:
        raise HTTPException(status_code=400, detail="semantic_steps must be greater than 0")
    
    if semantic_steps > 10:
        logger.warning(f"semantic_steps set to {semantic_steps} which is high. This may significantly increase processing time.")
    
    if duration is not None:
        if duration <= 0:
            raise HTTPException(status_code=400, detail="duration must be greater than 0")
        if duration > 40:
            logger.warning(f"Duration set to {duration} which may cause generation failure due to memory constraints. Recommended max is 35-40.")
    
    if time_steps_factor is not None and time_steps_factor <= 0:
        raise HTTPException(status_code=400, detail="time_steps_factor must be greater than 0")
    
    if duration is not None and time_steps_factor is not None:
        total_steps = duration * time_steps_factor
        if total_steps > 240:  # This is a guess at a safe upper limit
            logger.warning(f"Total time steps (duration * time_steps_factor = {total_steps}) is very high and may cause memory issues.")
    
    if temperature is not None:
        if temperature <= 0:
            raise HTTPException(status_code=400, detail="temperature must be greater than 0")
        if temperature < 0.1 or temperature > 2.0:
            logger.warning(f"Temperature of {temperature} is outside recommended range (0.1-2.0)")
    
    if not prompt:
        logger.warning("Empty prompt provided. Using default prompt.")
        prompt = "Diverse kinds of instrument and richness"
    
    try:
        # Create temporary directories
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        logger.info(f"Created temporary directories: {input_dir}, {output_dir}")
        
        # Save uploaded file
        input_path = input_dir / audio_file.filename
        with input_path.open("wb") as input_file:
            shutil.copyfileobj(audio_file.file, input_file)
        
        logger.info(f"Saved uploaded file to {input_path}")
        
        # Process the audio
        logger.info("Starting audio processing...")
        output_path = generator.process_audio(
            audio_path=input_path,
            output_dir=output_dir,
            request_id=request_id,
            semantic_steps=semantic_steps,
            duration=duration,
            time_steps_factor=time_steps_factor,
            temperature=temperature,
            prompt=prompt,
            save_for_eval=save_for_eval,
        )
        logger.info(f"Successfully processed audio, output saved to {output_path}")
        
        # Add cleanup task to background tasks
        background_tasks.add_task(cleanup_temp_files, temp_dir)
        
        # Return the generated file
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={output_path.name}"}
        )
            
    except FileNotFoundError as e:
        error_msg = f"File not found: {str(e)}"
        logger.error(error_msg)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=404, detail=error_msg)
        
    except PermissionError as e:
        error_msg = f"Permission error: {str(e)}"
        logger.error(error_msg)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=403, detail=error_msg)
    
    except torch.cuda.OutOfMemoryError as e:
        # Specific handling for CUDA out of memory errors
        error_msg = "GPU memory exceeded. Try reducing duration or semantic_steps parameters."
        logger.error(f"CUDA out of memory: {str(e)}")
        logger.error(traceback.format_exc())
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=error_msg)
    
    except torch.cuda.CudaError as e:
        # Specific handling for other CUDA errors
        error_msg = "GPU error occurred. Please try again with different parameters."
        logger.error(f"CUDA error: {str(e)}")
        logger.error(traceback.format_exc())
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=error_msg)
    
    except RuntimeError as e:
        # This might catch memory issues even on CPU
        if "out of memory" in str(e).lower():
            sys_info = get_system_info()
            error_msg = "Memory limit exceeded. Try reducing duration or semantic_steps parameters."
            logger.error(f"Memory error: {str(e)}")
            logger.error(f"System resources at error: {sys_info}")
            logger.error(traceback.format_exc())
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise HTTPException(status_code=500, detail=error_msg)
        else:
            # General runtime error
            error_msg = f"Runtime error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise HTTPException(status_code=500, detail=error_msg)
        
    except Exception as e:
        # Log the full exception with traceback
        error_msg = f"Error during audio generation: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Log system info at time of error
        sys_info = get_system_info()
        logger.error(f"System resources at error: {sys_info}")
        
        # Clean up if anything goes wrong
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        # Return a more user-friendly error message
        raise HTTPException(status_code=500, detail=error_msg)