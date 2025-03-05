from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid
import logging
import traceback
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

app = FastAPI()
generator = MusicGenerator()

def cleanup_temp_files(temp_dir: Path):
    """Clean up temporary files after response has been sent"""
    try:
        if temp_dir.exists():
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
            logger.info(f"Successfully cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory {temp_dir}: {str(e)}")

@app.post("/generate")
async def generate_music(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    semantic_steps: int = Form(6),  # Required parameter
    duration: Optional[int] = Form(40),  # Default is 20 seconds
    time_steps_factor: Optional[int] = Form(6),
    temperature: Optional[float] = Form(0.85),
    prompt: Optional[str] = Form("Diverse kinds of instrument and richness"),
    save_for_eval: Optional[bool] = Form(False)
):
    """
    Generate music based on an input audio file and various parameters.
    
    Parameters:
    - audio_file: Input audio file to transform
    - semantic_steps: Number of semantic steps for token generation (higher = more processing time but potentially better quality)
    - duration: Target duration of the output audio in seconds (20-35 seconds recommended, values >40 may cause issues)
    - time_steps_factor: Controls how detailed the time steps are in the model (higher = more granular)
    - temperature: Controls randomness of generation (higher = more creative/random, lower = more deterministic)
    - prompt: Text description to guide the audio generation
    - save_for_eval: Whether to save the input/output files for later evaluation
    
    Returns:
    - FileResponse: The generated audio file
    """
    request_id = str(uuid.uuid4())
    temp_dir = Path("temp") / request_id
    logger.info(f"Starting audio generation with request ID: {request_id}")
    logger.info(f"Parameters: semantic_steps={semantic_steps}, duration={duration}, time_steps_factor={time_steps_factor}, temperature={temperature}")
    
    # Validate parameters
    if semantic_steps <= 0:
        raise HTTPException(status_code=400, detail="semantic_steps must be greater than 0")
    
    if duration is not None and duration > 40:
        logger.warning(f"Duration set to {duration} which may cause generation failure. Recommended max is 40.")
    
    if temperature is not None and (temperature < 0.1 or temperature > 2.0):
        logger.warning(f"Temperature of {temperature} is outside recommended range (0.1-2.0)")
    
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
        
    except Exception as e:
        # Log the full exception with traceback
        error_msg = f"Error during audio generation: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Clean up if anything goes wrong
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        # Return a more user-friendly error message
        raise HTTPException(status_code=500, detail=error_msg)