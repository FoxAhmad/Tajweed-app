# File: segment_audio.py (Add to your Flask backend)
import os
import subprocess
import logging
import tempfile
import uuid
import shutil
from flask import jsonify

# Configure logging
logger = logging.getLogger(__name__)

# Set a constant for the input directory
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
logger.info(INPUT_DIR)
os.makedirs(INPUT_DIR, exist_ok=True)

def run_segmentation_pipeline(audio_file_path, letter_name):
    """
    Runs the Docker segmentation pipeline on the provided audio file.
    
    Args:
        audio_file_path (str): Path to the input audio file
        letter_name (str): Name of the letter (e.g., 'alif', 'ba', etc.)
        
    Returns:
        dict: Dictionary with segmentation results or error
    """
    try:
        # Create directory for segments output
        segments_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'segments')
        os.makedirs(segments_dir, exist_ok=True)
        
        # If the audio_file_path is not already in the input directory, copy it there
        if not audio_file_path.startswith(INPUT_DIR):
            # # Generate a filename for the input file
            input_filename = f"input.wav"
            input_file_path = os.path.join(INPUT_DIR, input_filename)
            
            # # Copy the audio file to the input directory
            # shutil.copy(audio_file_path, input_file_path)
        else:
            # The file is already in the input directory
            input_file_path = audio_file_path
            input_filename = os.path.basename(input_file_path)
        
        logger.info(f"Starting segmentation pipeline for letter: {letter_name}")
        logger.info(f"Input file path: {input_file_path}")
        
        # Run the Docker command
        cmd = [
            "docker", "run",
            "-v", f"{INPUT_DIR}:/app/input",
            "-v", f"{segments_dir}:/app/segments",
            "-e", f"LETTER={letter_name}",
            "xxmoeedxx/pipeline:latest"
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Run the Docker command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        # Check if the command was successful
        if process.returncode != 0:
            logger.error(f"Docker command failed with code {process.returncode}: {stderr}")
            return {
                "success": False,
                "error": f"Segmentation failed: {stderr}"
            }
        
        logger.info("Segmentation completed successfully")
        
        # Get list of segmented files
        segmented_files = []
        for filename in os.listdir(segments_dir):
            if filename.endswith('.wav'):
                segment_path = os.path.join(segments_dir, filename)
                # Get phoneme name from filename (assuming format like 'alif_a.wav')
                parts = filename.split('_')
                if len(parts) > 1:
                    phoneme = parts[1].split('.')[0]  # Extract 'a' from 'alif_a.wav'
                    segmented_files.append({
                        "phoneme": phoneme,
                        "path": segment_path,
                        "filename": filename
                    })
        
        logger.info(f"Found {len(segmented_files)} segmented files")
        
        # Create result dictionary
        result = {
            "success": True,
            "letter": letter_name,
            "input_file": input_file_path,
            "segments": segmented_files,
            "segments_dir": segments_dir
        }
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in segmentation pipeline: {str(e)}")
        return {
            "success": False,
            "error": f"Segmentation failed: {str(e)}"
        }
    
    # Note: We're not cleaning up the temp directory to keep the segments
    # In a production environment, you might want to implement a cleanup strategy