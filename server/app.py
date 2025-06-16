from flask import Flask, request, jsonify 
from flask_cors import CORS 
import os
import torch
import torch.nn as nn
import numpy as np
import tempfile
import logging
import time
from datetime import datetime 
import wave
import io
import struct
import subprocess
import shutil
from transformers import AutoProcessor, AutoModelForAudioClassification, WhisperProcessor, WhisperModel 
from werkzeug.utils import secure_filename
import librosa
import soundfile as sf

# Create an input directory for storing uploaded audio files
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
os.makedirs(INPUT_DIR, exist_ok=True)

from segment_audio import run_segmentation_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store models and processors
models = {}
processors = {}

# Configuration constants
MAX_AUDIO_LENGTH = 30
WHISPER_SAMPLE_RATE = 16000
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.ogg', '.webm']

# CNN Model Class Definition
class SpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 2)  # Binary classification
        )
    
    def forward(self, x):
        return self.cnn(x)
    
    def predict_tensor(self, input_tensor):
        """Predict from preprocessed tensor"""
        self.eval()
        with torch.no_grad():
            logits = self(input_tensor)
            probs = torch.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted].item()
        return predicted, confidence
    
    @classmethod
    def load_model(cls, weight_path, map_location=None):
        """Load model from weights file"""
        if map_location is None:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = cls()
        model.load_state_dict(torch.load(weight_path, map_location=map_location))
        model.to(map_location)
        return model

# Define the mapping between phonemes and models - now includes CNN
PHONEME_MODEL_MAPPING = {
    'ee': {
        'whisper': 'ahmad1703/whisper_ee', 
        'wave2vec': 'xxmoeedxx/wav2vec2_ee',
        'cnn': 'ee_cnn_spectrogram_model.pth'
    },
    'so': {
        'whisper': 'ahmad1703/whisper_so',
        'wave2vec': 'xxmoeedxx/wav2vec2_so',
        'cnn': 'so_cnn_spectrogram_model.pth'
    },
    'si': {
        'whisper': 'ahmad1703/whisper_si',
        'wave2vec': 'xxmoeedxx/wav2vec2_si',
        'cnn': 'si_cnn_spectrogram_model.pth'
    },
    'aa': {
        'whisper': 'ahmad1703/whisper_aa',
        'wave2vec': 'xxmoeedxx/wav2vec2_aa',
        'cnn': 'aa_cnn_spectrogram_model.pth'
    },
    'n': {
        'whisper': 'ahmad1703/whisper_n',
        'wave2vec': 'xxmoeedxx/wav2vec2_n',
        'cnn': 'n_cnn_spectrogram_model.pth'
    },
    'd': {
        'whisper': 'ahmad1703/whisper_d',
        'wave2vec': 'xxmoeedxx/wav2vec2_d',
        'cnn': 'd_cnn_spectrogram_model.pth'
    }
}

# Update the SUPPORTED_PHONEMES constant
SUPPORTED_PHONEMES = list(PHONEME_MODEL_MAPPING.keys())

# WhisperClassifier class definition (unchanged)
class WhisperClassifier(torch.nn.Module):
    def __init__(self, model_name="openai/whisper-small"):
        super(WhisperClassifier, self).__init__()
        self.whisper = WhisperModel.from_pretrained(model_name)
        self.fc = torch.nn.Linear(self.whisper.config.d_model, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_features):
        hidden_states = self.whisper.encoder(input_features).last_hidden_state
        pooled_output = hidden_states.mean(dim=1)
        logits = self.fc(pooled_output)
        return self.sigmoid(logits).squeeze(1)

def load_model(model_name, model_id=None, max_retries=2):
    """
    Enhanced model loading function that supports your specific directory structure
    """
    # Define default model IDs if none provided
    default_model_ids = {
        "wave2vec": "xxmoeedxx/wav2vec2_si",
        "whisper": "ahmad1703/whisper_ee",
        "cnn": "ee_cnn_spectrogram_model.pth"
    }
    
    # Use model_id if provided, otherwise use default
    actual_model_id = model_id if model_id else default_model_ids.get(model_name)
    cache_key = f"{model_name}_{actual_model_id}"
    
    # Check if model is already loaded
    if cache_key in models and models[cache_key] is not None:
        logger.info(f"Using cached model: {cache_key}")
        return models[cache_key], processors.get(cache_key)
    
    # Initialize placeholders in the cache
    models[cache_key] = None
    processors[cache_key] = None
    
    # Retry logic
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Loading model (attempt {attempt+1}/{max_retries+1}): {actual_model_id}")
            
            if model_name == "cnn":
                logger.info(f"Loading CNN model: {actual_model_id}")
                
                # Updated model paths to match your directory structure
                model_paths = [
                    actual_model_id,  # Direct path
                    os.path.join("cnn_models", actual_model_id),  # Your cnn_models directory
                    os.path.join("server", "cnn_models", actual_model_id),  # Your server/cnn_models
                    os.path.join("models", "cnn", actual_model_id),  # Standard models/cnn
                    os.path.join("models", actual_model_id),  # Standard models directory
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_models", actual_model_id),  # Relative to script
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", "cnn_models", actual_model_id),  # Your specific structure
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cnn", actual_model_id),
                    os.path.join("..", "cnn_models", actual_model_id),  # If running from subdirectory
                    actual_model_id.replace('.pth', '') + '.pth',  # Ensure .pth extension
                ]
                
                model_path = None
                for path in model_paths:
                    logger.debug(f"Checking model path: {path}")
                    if os.path.exists(path):
                        model_path = path
                        logger.info(f"✓ Found CNN model at: {model_path}")
                        break
                
                if not model_path:
                    logger.error(f"CNN model file not found: {actual_model_id}")
                    logger.info(f"Searched in paths: {model_paths}")
                    
                    # Try to list what's actually in the cnn_models directory
                    for search_dir in ["cnn_models", "server/cnn_models", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_models")]:
                        if os.path.exists(search_dir):
                            try:
                                files = os.listdir(search_dir)
                                logger.info(f"Files in {search_dir}: {files}")
                            except Exception as e:
                                logger.warning(f"Could not list files in {search_dir}: {e}")
                    
                    models.pop(cache_key, None)
                    processors.pop(cache_key, None)
                    return None, None
                
                # Load CNN model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                try:
                    models[cache_key] = SpectrogramCNN.load_model(model_path, map_location=device)
                    models[cache_key].eval()
                    
                    # CNN models don't need a separate processor
                    processors[cache_key] = None
                    
                    logger.info(f"✓ CNN model {actual_model_id} loaded successfully from {model_path} on {device}")
                    return models[cache_key], processors[cache_key]
                    
                except Exception as e:
                    logger.error(f"Error loading CNN model from {model_path}: {e}")
                    # Check if it's a file permission or corruption issue
                    try:
                        file_size = os.path.getsize(model_path)
                        logger.info(f"Model file size: {file_size} bytes")
                        if file_size < 1024:  # Less than 1KB is suspicious
                            logger.warning(f"Model file seems too small: {file_size} bytes")
                    except:
                        pass
                    raise
                
            elif model_name == "wave2vec":
                logger.info(f"Loading Wave2Vec model: {actual_model_id}")
                
                # Load processor
                processors[cache_key] = AutoProcessor.from_pretrained(
                    actual_model_id,
                    use_auth_token=os.environ.get("HF_TOKEN"),
                    cache_dir=os.environ.get("MODEL_CACHE_DIR")
                )
                
                # Load model
                models[cache_key] = AutoModelForAudioClassification.from_pretrained(
                    actual_model_id,
                    use_auth_token=os.environ.get("HF_TOKEN"),
                    cache_dir=os.environ.get("MODEL_CACHE_DIR")
                )
                
                # Move model to GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                models[cache_key] = models[cache_key].to(device)
                models[cache_key].eval()
                
                logger.info(f"✓ Wave2Vec model {actual_model_id} loaded successfully on {device}")
                return models[cache_key], processors[cache_key]
                
            elif model_name == "whisper":
                logger.info(f"Loading Whisper classifier model: {actual_model_id}")
                
                # Load processor
                processors[cache_key] = WhisperProcessor.from_pretrained(
                    "openai/whisper-small",
                    use_auth_token=os.environ.get("HF_TOKEN"),
                    cache_dir=os.environ.get("MODEL_CACHE_DIR")
                )
                
                # Load model
                models[cache_key] = WhisperClassifier()
                
                # Load weights
                try:
                    hf_url = f"https://huggingface.co/{actual_model_id}/resolve/main/pytorch_model.bin"
                    logger.info(f"Attempting to download model weights from: {hf_url}")
                    
                    models[cache_key].load_state_dict(torch.hub.load_state_dict_from_url(
                        hf_url,
                        map_location=torch.device('cpu'),
                        progress=True
                    ))
                    logger.info(f"✓ Loaded Whisper model weights from Hugging Face: {actual_model_id}")
                    
                except Exception as e:
                    logger.warning(f"Could not load from HF directly: {e}")
                    # Fallback to local path if available
                    model_filename = actual_model_id.split('/')[-1]
                    local_paths = [
                        os.environ.get(f"WHISPER_MODEL_PATH_{actual_model_id.replace('/', '_')}", ""),
                        os.path.join("models", "whisper", f"{model_filename}.pth"),
                        os.path.join("whisper_models", f"{model_filename}.pth"),
                        os.path.join("models", f"whis_{model_filename.split('_')[-1]}.pth")
                    ]
                    
                    loaded = False
                    for local_path in local_paths:
                        if local_path and os.path.exists(local_path):
                            models[cache_key].load_state_dict(torch.load(local_path, map_location=torch.device('cpu')))
                            logger.info(f"✓ Loaded Whisper model weights from local path: {local_path}")
                            loaded = True
                            break
                    
                    if not loaded:
                        logger.error(f"Could not find model weights for {actual_model_id}")
                        models.pop(cache_key, None)
                        processors.pop(cache_key, None)
                        return None, None
                
                # Move model to GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                models[cache_key] = models[cache_key].to(device)
                models[cache_key].eval()
                
                logger.info(f"✓ Whisper model {actual_model_id} loaded successfully on {device}")
                return models[cache_key], processors[cache_key]
            
            else:
                logger.error(f"Unknown model type: {model_name}")
                models.pop(cache_key, None)
                processors.pop(cache_key, None)
                return None, None
                
        except Exception as e:
            logger.error(f"Error loading {model_name} model (attempt {attempt+1}/{max_retries+1}): {e}")
            
            if attempt == max_retries:
                logger.error(f"Failed to load {model_name} model after {max_retries+1} attempts")
                models.pop(cache_key, None)
                processors.pop(cache_key, None)
                return None, None
            
            # Wait before retrying with exponential backoff
            wait_time = 2 ** attempt
            logger.info(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)



# Debug function to check your current setup
def debug_model_paths():
    """Debug function to check what model files are available"""
    logger.info("=== DEBUG: Checking model file locations ===")
    
    # Check your cnn_models directory
    cnn_dirs = ["cnn_models", "server/cnn_models", "./cnn_models", "../cnn_models"]
    
    for cnn_dir in cnn_dirs:
        if os.path.exists(cnn_dir):
            logger.info(f"✓ Found directory: {cnn_dir}")
            try:
                files = os.listdir(cnn_dir)
                logger.info(f"  Files: {files}")
                
                # Check each expected model file
                expected_models = ['aa_cnn_spectrogram_model.pth', 'ee_cnn_spectrogram_model.pth', 
                                 'n_cnn_spectrogram_model.pth', 'si_cnn_spectrogram_model.pth', 
                                 'so_cnn_spectrogram_model.pth']
                
                for model_file in expected_models:
                    model_path = os.path.join(cnn_dir, model_file)
                    if os.path.exists(model_path):
                        size = os.path.getsize(model_path)
                        logger.info(f"  ✓ {model_file} - {size} bytes")
                    else:
                        logger.warning(f"  ✗ {model_file} - NOT FOUND")
                        
            except Exception as e:
                logger.error(f"  Error reading {cnn_dir}: {e}")
        else:
            logger.info(f"✗ Directory not found: {cnn_dir}")
    
    # Check current working directory
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    
    # Check script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Script directory: {script_dir}")
    
    logger.info("=== END DEBUG ===")
    
    
def preprocess_audio_for_cnn(audio_data, sample_rate, n_mels=128, duration=1.0):
    """
    Preprocess audio data for CNN model input
    """
    try:
        # Ensure we have the right duration
        target_samples = int(sample_rate * duration)
        
        # Fix length to target duration
        if len(audio_data) != target_samples:
            audio_data = librosa.util.fix_length(audio_data, size=target_samples)
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sample_rate, 
            n_mels=n_mels,
            hop_length=512,
            win_length=2048
        )
        
        # Convert to dB scale
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_db_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        
        # Convert to tensor with correct dimensions: (batch, channels, height, width)
        input_tensor = torch.tensor(mel_db_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return input_tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing audio for CNN: {e}")
        raise

def read_audio_file_improved(file_or_bytes):
    """
    Improved audio file reader using librosa for better compatibility and quality.
    """
    try:
        # Get bytes from file object if needed
        if hasattr(file_or_bytes, 'read'):
            logger.info("Reading as file-like object")
            audio_bytes = file_or_bytes.read()
        else:
            audio_bytes = file_or_bytes
        
        # Write to temp file for librosa
        with tempfile.NamedTemporaryFile(suffix='.audio', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes)
        
        try:
            # Use librosa for robust audio loading with automatic resampling
            audio_data, sample_rate = librosa.load(
                temp_path,
                sr=WHISPER_SAMPLE_RATE,  # Resample to 16kHz
                mono=True,               # Convert to mono
                dtype=np.float32         # Use float32
            )
            
            logger.info(f"Successfully read audio with librosa: shape={audio_data.shape}, sample_rate={sample_rate}")
            
            # Ensure audio is not empty and has reasonable length
            if len(audio_data) == 0:
                raise ValueError("Audio file appears to be empty")
            
            # Normalize audio to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            return audio_data, sample_rate
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error reading audio file: {str(e)}")
        # Return silence as fallback
        logger.warning("Returning silence as fallback")
        sample_rate = WHISPER_SAMPLE_RATE
        duration = 1.0
        data = np.zeros(int(sample_rate * duration), dtype=np.float32)
        return data, sample_rate

def preprocess_audio_for_whisper(audio_data, sample_rate, allow_short=False):
    """
    Specialized preprocessing for Whisper models with support for short segments.
    """
    # Ensure audio is float32
    audio_data = audio_data.astype(np.float32)
    
    # Check for NaN or Inf values
    if np.isnan(audio_data).any() or np.isinf(audio_data).any():
        logger.warning("Found NaN or Inf values in audio data. Replacing with zeros.")
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure audio is in range [-1, 1]
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        logger.info(f"Audio amplitude too high ({max_val}), normalizing...")
        audio_data = audio_data / max_val
    
    # For segments, we allow much shorter audio
    if allow_short:
        min_samples = int(0.01 * sample_rate)  # 0.01 seconds minimum for segments
        logger.info(f"Using short segment mode, minimum samples: {min_samples}")
    else:
        min_samples = int(0.1 * sample_rate)  # 0.1 seconds minimum for full recordings
    
    # Ensure minimum length
    if len(audio_data) < min_samples:
        logger.info(f"Audio too short ({len(audio_data)} samples), padding to {min_samples} samples")
        pad_length = min_samples - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad_length), mode='constant', constant_values=0.0)
    
    # For very short segments, don't trim silence as it might remove the entire signal
    if not allow_short or len(audio_data) > int(0.2 * sample_rate):
        try:
            audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
            logger.info(f"Audio after trimming: {len(audio_data)} samples")
        except:
            logger.warning("Could not trim audio, using original")
    else:
        logger.info("Skipping silence trimming for short segment")
    
    # Ensure we still have enough audio after trimming
    if len(audio_data) < min_samples:
        audio_data = np.pad(audio_data, (0, min_samples - len(audio_data)), mode='constant', constant_values=0.0)
    
    return audio_data

def predict_audio(model, processor, audio_data, sample_rate, model_name, phoneme=None, is_segment=False):
    """
    Enhanced audio prediction function supporting CNN, Wave2Vec, and Whisper models.
    """
    try:
        start_time = time.time()
        logger.info(f"Processing audio with {model_name} model, phoneme: {phoneme}, is_segment: {is_segment}")
        
        # Check audio length
        audio_length = len(audio_data) / sample_rate
        logger.info(f"Audio length: {audio_length:.3f} seconds")
        
        # Different length requirements for segments vs full recordings
        if is_segment:
            max_length = 5.0  # Allow up to 5 seconds for segments
            min_length = 0.005  # Allow segments as short as 5 milliseconds
        else:
            max_length = MAX_AUDIO_LENGTH  # 30 seconds for full recordings
            min_length = 0.1  # 0.1 seconds for full recordings
        
        if audio_length > max_length:
            return {
                "error": f"Audio is too long. Maximum allowed length is {max_length} seconds.",
                "audio_length": audio_length
            }
        
        if audio_length < min_length:
            if is_segment:
                logger.warning(f"Very short segment ({audio_length:.3f}s), proceeding with analysis")
            else:
                return {
                    "error": f"Audio is too short. Minimum length is {min_length} seconds.",
                    "audio_length": audio_length
                }
        
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        if model_name == "cnn":
            logger.info("Processing audio with CNN model")
            
            try:
                # Preprocess audio for CNN
                input_tensor = preprocess_audio_for_cnn(audio_data, sample_rate)
                
                # Move to device
                input_tensor = input_tensor.to(device)
                
                # Make prediction
                prediction, confidence = model.predict_tensor(input_tensor)
                
                # Convert confidence to percentage
                confidence = confidence * 100
                
                logger.info(f"CNN prediction: {prediction}, confidence: {confidence:.2f}%")
                
            except Exception as e:
                logger.exception(f"Error during CNN inference: {str(e)}")
                prediction = 0
                confidence = 50.0
                
        elif model_name == "wave2vec":
            logger.info("Processing audio with Wave2Vec model")
            
            # Ensure correct sample rate for Wave2Vec
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            try:
                # For very short segments, ensure we have enough samples for Wave2Vec
                min_wave2vec_samples = 320  # About 0.02 seconds at 16kHz
                if len(audio_data) < min_wave2vec_samples:
                    logger.info(f"Padding very short segment from {len(audio_data)} to {min_wave2vec_samples} samples")
                    pad_length = min_wave2vec_samples - len(audio_data)
                    audio_data = np.pad(audio_data, (0, pad_length), mode='edge')
                
                # Process input
                inputs = processor(
                    audio_data, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt", 
                    padding=True
                )
                
                # Move to device
                for key in inputs:
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(device)
                
                # Predict
                with torch.no_grad():
                    outputs = model(inputs["input_values"])
                    logits = outputs.logits
                    
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.warning("Found NaN or Inf in Wave2Vec output")
                        prediction = 0
                        confidence = 50.0
                    else:
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        prediction = torch.argmax(probs, dim=-1).item()
                        confidence = probs[0][prediction].item() * 100
                        
                        if np.isnan(confidence) or np.isinf(confidence):
                            confidence = 50.0
                            
            except Exception as e:
                logger.exception(f"Error during Wave2Vec inference: {str(e)}")
                prediction = 0
                confidence = 50.0
                
        elif model_name == "whisper":
            logger.info("Processing audio with Whisper model")
            
            # Specialized preprocessing for Whisper with short segment support
            audio_data = preprocess_audio_for_whisper(audio_data, sample_rate, allow_short=is_segment)
            
            try:
                # Process input features for Whisper
                logger.info(f"Processing audio data shape: {audio_data.shape}")
                
                # Use the processor to create input features
                inputs = processor(
                    audio_data,
                    sampling_rate=WHISPER_SAMPLE_RATE,
                    return_tensors="pt"
                )
                
                # Get input features
                input_features = inputs.input_features
                logger.info(f"Input features shape: {input_features.shape}")
                
                # Move to device
                input_features = input_features.to(device)
                
                # Make prediction
                with torch.no_grad():
                    output = model(input_features)
                    
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        logger.warning("Found NaN or Inf in Whisper output")
                        prediction = 0
                        confidence = 50.0
                    else:
                        # Get probability (sigmoid output is between 0 and 1)
                        probability = output.item()
                        logger.info(f"Raw Whisper output: {probability}")
                        
                        # Convert to binary prediction
                        prediction = 1 if probability > 0.5 else 0
                        
                        # Calculate confidence
                        if prediction == 1:
                            confidence = probability * 100
                        else:
                            confidence = (1 - probability) * 100
                        
                        # Ensure confidence is valid
                        if np.isnan(confidence) or np.isinf(confidence):
                            confidence = 50.0
                        
                        logger.info(f"Whisper prediction: {prediction}, confidence: {confidence:.2f}%")
                        
            except Exception as e:
                logger.exception(f"Error during Whisper inference: {str(e)}")
                prediction = 0
                confidence = 50.0
        
        else:
            return {"error": f"Unknown model: {model_name}"}
        
        # Generate feedback
        feedback = get_detailed_feedback(phoneme, prediction, confidence)
        
        # Create result
        result = {
            "correct": bool(prediction),
            "confidence": round(float(confidence), 2),
            "recommendation": feedback,
            "processing_time": round(time.time() - start_time, 3),
            "model_used": model_name,
            "audio_length": round(audio_length, 3),
            "is_segment": is_segment
        }
        
        logger.info(f"Final result: {result}")
        return result
        
    except Exception as e:
        logger.exception("Error processing audio data")
        return {"error": str(e)}

def get_detailed_feedback(phoneme, prediction, confidence):
    """
    Generate detailed feedback based on prediction and confidence.
    """
    phoneme_display = phoneme if phoneme else "this sound"
    
    if prediction == 1:
        if confidence > 90:
            return f"Excellent! Your pronunciation of '{phoneme_display}' is very accurate."
        elif confidence > 75:
            return f"Good job! Your pronunciation of '{phoneme_display}' is correct, but could be slightly improved."
        else:
            return f"Your pronunciation of '{phoneme_display}' is acceptable, but needs more practice for clarity."
    else:
        if confidence > 90:
            return f"Your pronunciation of '{phoneme_display}' needs significant improvement. Please listen to the reference audio again."
        elif confidence > 75:
            return f"Your pronunciation of '{phoneme_display}' has some issues. Try focusing on the correct articulation point."
        else:
            return f"Your pronunciation of '{phoneme_display}' needs work, but is not far off. Keep practicing!"

def validate_request(request):
    """
    Validate the incoming request.
    """
    # Check if audio file is present
    if 'audio' not in request.files:
        return False, "No audio file provided"
    
    audio_file = request.files['audio']
    
    # Check if model is specified and valid
    model_name = request.form.get('model', 'wave2vec')
    if model_name not in ['whisper', 'wave2vec', 'cnn']:
        return False, f"Invalid model specified: {model_name}. Use 'whisper', 'wave2vec', or 'cnn'."
    
    # Check if phoneme is valid (if specified)
    phoneme = request.form.get('phoneme', '')
    if phoneme and phoneme not in SUPPORTED_PHONEMES:
        return False, f"Unsupported phoneme: {phoneme}. Please use one of: {', '.join(SUPPORTED_PHONEMES)}"
    
    return True, ""
@app.route('/segment-audio', methods=['POST'])
def segment_audio():
    """
    Endpoint to segment an audio file using the Docker pipeline
    """
    try:
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"
        logger.info(f"Request {request_id}: Received segment-audio request")
        
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        letter = request.form.get('letter', '')
        
        if not letter:
            return jsonify({"error": "Please specify a letter name"}), 400
        
        model_name = request.form.get('model', 'whisper')
        
        # Save audio file to temporary location
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        
        try:
            audio_file.save(temp_file_path)
            logger.info(f"Request {request_id}: Saved audio file to {temp_file_path}")
            
            # Run segmentation pipeline
            result = run_segmentation_pipeline(temp_file_path, letter)
            
            if not result["success"]:
                logger.error(f"Request {request_id}: Segmentation failed - {result.get('error', 'Unknown error')}")
                return jsonify({"error": result.get('error', 'Segmentation failed')}), 500
            
            # Process the segmented files
            segments_data = []
            i = 1
            logger.info(f"Request {request_id}: Processing {len(result.get('segments', []))} segments")
            logger.info(f"Request {request_id}: Segments from pipeline: {result.get('segments', [])}")
            
            for segment in result.get('segments', []):
                logger.info(f"Request {request_id}: Processing segment {i} - phoneme: {segment['phoneme']}")
                
                segment_id = f"input_{i}_{segment['phoneme']}"
                segment_path = segment['path']
                
                # Log the original segment path from Docker pipeline
                logger.info(f"Request {request_id}: Original segment path from Docker: {segment_path}")
                
                # Check if the source file exists
                if not os.path.exists(segment_path):
                    logger.error(f"Request {request_id}: Source segment file does not exist: {segment_path}")
                    # Try to find the file in common locations
                    possible_source_paths = [
                        os.path.join('segments', 'input', f"input_{i}_{segment['phoneme']}.wav"),
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'segments', 'input', f"input_{i}_{segment['phoneme']}.wav"),
                        os.path.join('segments', f"input_{i}_{segment['phoneme']}.wav"),
                    ]
                    
                    for alt_path in possible_source_paths:
                        logger.info(f"Request {request_id}: Trying alternative source path: {alt_path}")
                        if os.path.exists(alt_path):
                            segment_path = alt_path
                            logger.info(f"Request {request_id}: ✓ Found segment at: {segment_path}")
                            break
                    else:
                        logger.error(f"Request {request_id}: Could not find segment file anywhere")
                        continue
                
                # Create the web-accessible path matching the URL structure
                # URL will be /segments/input/input_X_phoneme.wav, so copy to static/segments/input/
                web_accessible_path = os.path.join('static', 'segments', 'input', f"{segment_id}.wav")
                web_accessible_full_path = os.path.join(app.root_path, web_accessible_path)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(web_accessible_full_path), exist_ok=True)
                logger.info(f"Request {request_id}: Created directory: {os.path.dirname(web_accessible_full_path)}")
                
                try:
                    # Copy the segment file to web-accessible location
                    logger.info(f"Request {request_id}: Copying from {segment_path} to {web_accessible_full_path}")
                    shutil.copy(segment_path, web_accessible_full_path)
                    
                    # Verify the copy was successful
                    if os.path.exists(web_accessible_full_path):
                        file_size = os.path.getsize(web_accessible_full_path)
                        logger.info(f"Request {request_id}: ✓ Successfully copied segment, size: {file_size} bytes")
                    else:
                        logger.error(f"Request {request_id}: ✗ Copy failed - destination file doesn't exist")
                        continue
                    
                    # Generate the URL that matches the file location
                    segment_url = f"/segments/input/{segment_id}.wav"
                    
                    segment_data = {
                        "phoneme": segment['phoneme'],
                        "url": segment_url,
                        "segment_id": segment_id,
                        "original_filename": segment.get('filename', f"{segment_id}.wav"),
                        "file_path": web_accessible_full_path,  # For debugging
                        "source_path": segment_path  # For debugging
                    }
                    
                    segments_data.append(segment_data)
                    logger.info(f"Request {request_id}: ✓ Added segment to response: {segment_data}")
                    
                except Exception as e:
                    logger.exception(f"Request {request_id}: Error copying segment file from {segment_path} to {web_accessible_full_path}: {str(e)}")
                    continue
                
                i += 1
            
            # Final verification - check all copied files exist
            logger.info(f"Request {request_id}: Final verification of copied segments:")
            for segment_data in segments_data:
                if os.path.exists(segment_data['file_path']):
                    logger.info(f"Request {request_id}: ✓ {segment_data['segment_id']} exists at {segment_data['file_path']}")
                else:
                    logger.error(f"Request {request_id}: ✗ {segment_data['segment_id']} missing at {segment_data['file_path']}")
            
            response = {
                "success": True,
                "letter": letter,
                "model": model_name,
                "segments": segments_data,
                "total_segments": len(segments_data)
            }
           
            logger.info(f"Request {request_id}: Segmentation completed successfully with {len(segments_data)} segments")
            return jsonify(response)
            
        finally:
            try:
                os.unlink(temp_file_path)
                logger.info(f"Request {request_id}: Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Request {request_id}: Failed to clean up temp file: {str(e)}")
                
    except Exception as e:
        logger.exception(f"Request {request_id}: Unhandled error in segment-audio endpoint")
        return jsonify({"error": str(e)}), 500
# Add these simplified endpoints to your Flask app.py
@app.route('/analyze-all-segments', methods=['POST'])
def analyze_all_segments():
    """
    Updated endpoint to analyze all segments with CNN, Whisper, and Wave2Vec support
    """
    try:
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"
        logger.info(f"Request {request_id}: Received analyze-all-segments request")
        
        # Get model type from request - now includes CNN support
        model_name = request.form.get('model', 'cnn')  # Default to CNN for speed
        if model_name not in ['cnn', 'whisper', 'wave2vec']:
            return jsonify({"error": "Invalid model type. Use 'cnn', 'whisper', or 'wave2vec'"}), 400
        
        logger.info(f"Request {request_id}: Using {model_name.upper()} model for batch analysis")
        
        # Define segments directory - try both possible locations
        possible_dirs = [
            os.path.join("segments", "input"),
            os.path.join("server", "segments", "input"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "segments", "input"),
            os.path.join("static", "segments", "input")  # Add static directory
        ]
        
        segments_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                segments_dir = dir_path
                break
        
        if not segments_dir:
            logger.error(f"Request {request_id}: Segments directory not found in any expected location")
            logger.info(f"Request {request_id}: Searched in: {possible_dirs}")
            return jsonify({"error": "Segments directory not found"}), 404
        
        logger.info(f"Request {request_id}: Using segments directory: {segments_dir}")
        
        # Get all WAV files from segments directory
        try:
            segment_files = [f for f in os.listdir(segments_dir) if f.endswith('.wav')]
            segment_files.sort()  # Sort for consistent order
            logger.info(f"Request {request_id}: Found {len(segment_files)} segment files: {segment_files}")
        except Exception as e:
            logger.exception(f"Request {request_id}: Error reading segments directory")
            return jsonify({"error": f"Error reading segments directory: {str(e)}"}), 500
        
        if not segment_files:
            logger.warning(f"Request {request_id}: No segment files found")
            return jsonify({"error": "No segment files found in segments directory"}), 404
        
        # Process each segment file
        results = {}
        successful_analyses = 0
        failed_analyses = 0
        total_processing_time = 0
        
        for filename in segment_files:
            file_path = os.path.join(segments_dir, filename)
            logger.info(f"Request {request_id}: Processing {filename}")
            
            try:
                # Extract phoneme from filename (format: input_X_phoneme.wav)
                phoneme = extract_phoneme_from_filename(filename)
                logger.info(f"Request {request_id}: Extracted phoneme '{phoneme}' from {filename}")
                
                if not phoneme:
                    logger.warning(f"Request {request_id}: Could not extract phoneme from {filename}")
                    results[filename] = {"error": "Could not extract phoneme from filename"}
                    failed_analyses += 1
                    continue
                
                # Check if phoneme is supported
                if phoneme not in PHONEME_MODEL_MAPPING:
                    logger.warning(f"Request {request_id}: Unsupported phoneme '{phoneme}'")
                    results[phoneme] = {"error": f"Unsupported phoneme: {phoneme}"}
                    failed_analyses += 1
                    continue
                
                # Get model ID for this phoneme and model type
                model_id = PHONEME_MODEL_MAPPING[phoneme].get(model_name)
                if not model_id:
                    logger.warning(f"Request {request_id}: No {model_name} model available for phoneme '{phoneme}'")
                    results[phoneme] = {"error": f"No {model_name} model available for phoneme {phoneme}"}
                    failed_analyses += 1
                    continue
                
                logger.info(f"Request {request_id}: Using {model_name} model {model_id} for phoneme '{phoneme}'")
                
                # Load model
                start_load_time = time.time()
                model, processor = load_model(model_name, model_id)
                load_time = time.time() - start_load_time
                
                if model is None:
                    logger.error(f"Request {request_id}: Failed to load {model_name} model {model_id}")
                    results[phoneme] = {"error": f"Failed to load {model_name} model {model_id}"}
                    failed_analyses += 1
                    continue
                
                # For CNN models, processor might be None (which is expected)
                if model_name != 'cnn' and processor is None:
                    logger.error(f"Request {request_id}: Failed to load processor for {model_name} model {model_id}")
                    results[phoneme] = {"error": f"Failed to load processor for {model_name} model {model_id}"}
                    failed_analyses += 1
                    continue
                
                logger.info(f"Request {request_id}: Model loaded in {load_time:.3f}s")
                
                # Read and analyze audio
                start_analysis_time = time.time()
                
                with open(file_path, 'rb') as f:
                    audio_data, sample_rate = read_audio_file_improved(f)
                
                # Run prediction with segment flag
                result = predict_audio(model, processor, audio_data, sample_rate, model_name, phoneme, is_segment=True)
                
                analysis_time = time.time() - start_analysis_time
                total_processing_time += analysis_time
                
                # Check if analysis was successful
                if "error" in result:
                    logger.error(f"Request {request_id}: Analysis failed for '{phoneme}': {result['error']}")
                    results[phoneme] = result
                    failed_analyses += 1
                    continue
                
                # Add metadata
                result.update({
                    "phoneme": phoneme,
                    "filename": filename,
                    "model_name": model_name,
                    "model_id": model_id,
                    "model_load_time": load_time,
                    "analysis_time": analysis_time,
                    "file_path": file_path  # For debugging
                })
                
                results[phoneme] = result
                successful_analyses += 1
                
                logger.info(f"Request {request_id}: ✓ Analysis complete for '{phoneme}' - "
                          f"{'correct' if result.get('correct', False) else 'incorrect'} "
                          f"({result.get('confidence', 0):.1f}% confidence, {analysis_time:.3f}s)")
                
            except Exception as e:
                logger.exception(f"Request {request_id}: Error processing {filename}")
                phoneme_key = phoneme if 'phoneme' in locals() else filename
                results[phoneme_key] = {"error": str(e), "filename": filename}
                failed_analyses += 1
        
        # Calculate overall statistics
        overall_accuracy = 0
        if successful_analyses > 0:
            correct_count = sum(1 for r in results.values() if isinstance(r, dict) and r.get('correct', False))
            overall_accuracy = round((correct_count / successful_analyses) * 100, 1)
        
        avg_processing_time = total_processing_time / successful_analyses if successful_analyses > 0 else 0
        
        # Prepare response
        response = {
            "success": True,
            "request_id": request_id,
            "model_name": model_name,
            "model_type": model_name,
            "segments_directory": segments_dir,
            "total_segments": len(segment_files),
            "successful_analyses": successful_analyses,
            "failed_analyses": failed_analyses,
            "overall_accuracy": overall_accuracy,
            "total_processing_time": round(total_processing_time, 3),
            "avg_processing_time": round(avg_processing_time, 3),
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "performance_summary": {
                "model_used": f"{model_name.upper()}",
                "speed_rating": "Very Fast" if model_name == 'cnn' else "Medium" if model_name == 'wave2vec' else "Slower",
                "accuracy_rating": "Good" if model_name == 'cnn' else "High" if model_name == 'wave2vec' else "Very High",
                "best_for": "Quick feedback" if model_name == 'cnn' else "Phoneme precision" if model_name == 'wave2vec' else "Detailed analysis"
            }
        }
        
        logger.info(f"Request {request_id}: Completed batch analysis - "
                   f"{successful_analyses}/{len(segment_files)} successful, "
                   f"{overall_accuracy}% accuracy, "
                   f"{total_processing_time:.3f}s total time with {model_name.upper()}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.exception("Unhandled error in analyze-all-segments endpoint")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-segment', methods=['POST'])
def analyze_segment():
    """
    Updated endpoint to analyze a single segment with CNN, Whisper, and Wave2Vec support
    """
    try:
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"
        logger.info(f"Request {request_id}: Received analyze-segment request")
        
        # Get parameters
        segment_id = request.form.get('segment_id', '')
        model_name = request.form.get('model', 'cnn')  # Default to CNN for speed
        
        if not segment_id:
            return jsonify({"error": "No segment_id provided"}), 400
        
        if model_name not in ['cnn', 'whisper', 'wave2vec']:
            return jsonify({"error": "Invalid model type. Use 'cnn', 'whisper', or 'wave2vec'"}), 400
        
        logger.info(f"Request {request_id}: Analyzing segment '{segment_id}' with {model_name.upper()} model")
        
        # Find segments directory
        possible_dirs = [
            os.path.join("segments", "input"),
            os.path.join("server", "segments", "input"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "segments", "input"),
            os.path.join("static", "segments", "input")  # Add static directory
        ]
        
        segments_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                segments_dir = dir_path
                break
        
        if not segments_dir:
            logger.error(f"Request {request_id}: Segments directory not found")
            return jsonify({"error": "Segments directory not found"}), 404
        
        # Find the segment file
        segment_filename = f"{segment_id}.wav"
        segment_path = os.path.join(segments_dir, segment_filename)
        
        if not os.path.exists(segment_path):
            logger.error(f"Request {request_id}: Segment file not found: {segment_path}")
            
            # Try to find similar files for debugging
            try:
                available_files = [f for f in os.listdir(segments_dir) if f.endswith('.wav')]
                logger.info(f"Request {request_id}: Available segment files: {available_files}")
            except:
                pass
            
            return jsonify({"error": f"Segment file not found: {segment_filename}"}), 404
        
        # Extract phoneme from segment_id
        phoneme = extract_phoneme_from_filename(segment_filename)
        if not phoneme:
            logger.error(f"Request {request_id}: Could not extract phoneme from segment_id: {segment_id}")
            return jsonify({"error": f"Could not extract phoneme from segment_id: {segment_id}"}), 400
        
        logger.info(f"Request {request_id}: Extracted phoneme '{phoneme}' from segment '{segment_id}'")
        
        # Check if phoneme is supported
        if phoneme not in PHONEME_MODEL_MAPPING:
            logger.error(f"Request {request_id}: Unsupported phoneme '{phoneme}'")
            return jsonify({"error": f"Unsupported phoneme: {phoneme}"}), 400
        
        # Get model ID for this phoneme and model type
        model_id = PHONEME_MODEL_MAPPING[phoneme].get(model_name)
        if not model_id:
            logger.error(f"Request {request_id}: No {model_name} model available for phoneme '{phoneme}'")
            available_models = list(PHONEME_MODEL_MAPPING[phoneme].keys())
            return jsonify({
                "error": f"No {model_name} model available for phoneme {phoneme}",
                "available_models": available_models,
                "suggestion": f"Try using one of: {', '.join(available_models)}"
            }), 400
        
        logger.info(f"Request {request_id}: Using {model_name} model '{model_id}' for phoneme '{phoneme}'")
        
        # Load model
        start_load_time = time.time()
        model, processor = load_model(model_name, model_id)
        load_time = time.time() - start_load_time
        
        if model is None:
            logger.error(f"Request {request_id}: Failed to load {model_name} model {model_id}")
            return jsonify({"error": f"Failed to load {model_name} model {model_id}"}), 500
        
        # For CNN models, processor might be None (which is expected)
        if model_name != 'cnn' and processor is None:
            logger.error(f"Request {request_id}: Failed to load processor for {model_name} model {model_id}")
            return jsonify({"error": f"Failed to load processor for {model_name} model {model_id}"}), 500
        
        logger.info(f"Request {request_id}: Model loaded successfully in {load_time:.3f}s")
        
        # Read and analyze audio
        start_analysis_time = time.time()
        
        try:
            with open(segment_path, 'rb') as f:
                audio_data, sample_rate = read_audio_file_improved(f)
            
            logger.info(f"Request {request_id}: Audio loaded - {len(audio_data)} samples at {sample_rate}Hz")
            
            # Run prediction with segment flag
            result = predict_audio(model, processor, audio_data, sample_rate, model_name, phoneme, is_segment=True)
            
            analysis_time = time.time() - start_analysis_time
            
        except Exception as e:
            logger.exception(f"Request {request_id}: Error during audio analysis")
            return jsonify({"error": f"Audio analysis failed: {str(e)}"}), 500
        
        # Check if analysis was successful
        if "error" in result:
            logger.error(f"Request {request_id}: Analysis returned error: {result['error']}")
            return jsonify(result), 500
        
        # Add comprehensive metadata
        result.update({
            "request_id": request_id,
            "segment_id": segment_id,
            "phoneme": phoneme,
            "model_name": model_name,
            "model_id": model_id,
            "model_load_time": round(load_time, 3),
            "analysis_time": round(analysis_time, 3),
            "total_request_time": round(load_time + analysis_time, 3),
            "segment_file": segment_filename,
            "segments_directory": segments_dir,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "type": model_name,
                "speed_rating": "Very Fast" if model_name == 'cnn' else "Medium" if model_name == 'wave2vec' else "Slower",
                "accuracy_rating": "Good" if model_name == 'cnn' else "High" if model_name == 'wave2vec' else "Very High",
                "best_for": "Quick feedback" if model_name == 'cnn' else "Phoneme precision" if model_name == 'wave2vec' else "Detailed analysis"
            }
        })
        
        logger.info(f"Request {request_id}: ✓ Single segment analysis complete - "
                   f"Phoneme '{phoneme}' analyzed as {'correct' if result.get('correct', False) else 'incorrect'} "
                   f"({result.get('confidence', 0):.1f}% confidence) in {analysis_time:.3f}s using {model_name.upper()}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.exception("Unhandled error in analyze-segment endpoint")
        return jsonify({"error": str(e)}), 500


# Helper function to extract phoneme from filename (enhanced with better error handling)
def extract_phoneme_from_filename(filename):
    """
    Extract phoneme from segment filename with enhanced error handling
    Expected format: input_X_phoneme.wav
    """
    try:
        # Remove .wav extension
        base_name = filename.replace('.wav', '')
        
        # Split by underscore
        parts = base_name.split('_')
        
        # Expected format: ['input', 'X', 'phoneme', ...]
        if len(parts) >= 3 and parts[0] == 'input':
            # Join remaining parts (in case phoneme has underscores)
            phoneme = '_'.join(parts[2:])
            
            # Validate phoneme is not empty and contains valid characters
            if phoneme and phoneme.replace('_', '').isalnum():
                return phoneme
        
        # Alternative format handling: just phoneme.wav
        elif len(parts) == 1 and parts[0]:
            return parts[0]
        
        logger.warning(f"Could not extract phoneme from filename '{filename}' - unexpected format")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting phoneme from filename '{filename}': {e}")
        return None


# Additional helper endpoint to list available segments (useful for debugging)
@app.route('/list-segments', methods=['GET'])
def list_segments():
    """
    Enhanced endpoint to list all available segments with model compatibility info
    """
    try:
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"
        logger.info(f"Request {request_id}: Listing available segments")
        
        # Find segments directory
        possible_dirs = [
            os.path.join("segments", "input"),
            os.path.join("server", "segments", "input"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "segments", "input"),
            os.path.join("static", "segments", "input")
        ]
        
        segments_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                segments_dir = dir_path
                break
        
        if not segments_dir:
            return jsonify({"error": f"Segments directory not found. Searched: {possible_dirs}"}), 404
        
        # Get all files
        try:
            files = [f for f in os.listdir(segments_dir) if f.endswith('.wav')]
            files.sort()
        except Exception as e:
            return jsonify({"error": f"Error reading segments directory: {str(e)}"}), 500
        
        segments_info = []
        model_compatibility = {}
        
        for filename in files:
            # Extract info from filename
            base_name = filename.replace('.wav', '')
            parts = base_name.split('_')
            
            phoneme = extract_phoneme_from_filename(filename)
            file_path = os.path.join(segments_dir, filename)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Check model compatibility
            available_models = []
            if phoneme and phoneme in PHONEME_MODEL_MAPPING:
                available_models = list(PHONEME_MODEL_MAPPING[phoneme].keys())
                
                if phoneme not in model_compatibility:
                    model_compatibility[phoneme] = available_models
            
            segment_info = {
                "filename": filename,
                "segment_id": base_name,
                "file_path": file_path,
                "file_size_bytes": file_size,
                "file_size_kb": round(file_size / 1024, 2),
                "phoneme": phoneme,
                "available_models": available_models,
                "is_supported": len(available_models) > 0
            }
            
            if len(parts) >= 3:
                segment_info["sequence_number"] = parts[1] if parts[1].isdigit() else "unknown"
            else:
                segment_info["sequence_number"] = "unknown"
            
            segments_info.append(segment_info)
        
        # Calculate summary statistics
        total_segments = len(segments_info)
        supported_segments = len([s for s in segments_info if s['is_supported']])
        total_size_kb = sum(s['file_size_kb'] for s in segments_info)
        unique_phonemes = len(set(s['phoneme'] for s in segments_info if s['phoneme']))
        
        response = {
            "success": True,
            "request_id": request_id,
            "segments_dir": segments_dir,
            "total_segments": total_segments,
            "supported_segments": supported_segments,
            "unsupported_segments": total_segments - supported_segments,
            "unique_phonemes": unique_phonemes,
            "total_size_kb": round(total_size_kb, 2),
            "total_size_mb": round(total_size_kb / 1024, 2),
            "segments": segments_info,
            "model_compatibility": model_compatibility,
            "available_model_types": ["cnn", "whisper", "wave2vec"],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Request {request_id}: Listed {total_segments} segments "
                   f"({supported_segments} supported, {unique_phonemes} unique phonemes)")
        
        return jsonify(response)
        
    except Exception as e:
        logger.exception("Error in list-segments endpoint")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-tajweed', methods=['POST'])
def analyze_tajweed():
    """
    Main endpoint for analyzing tajweed pronunciation - now supports CNN models.
    """
    try:
        # Log request
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"
        logger.info(f"Request {request_id}: Received analyze-tajweed request")
        
        # Validate request
        is_valid, error_message = validate_request(request)
        if not is_valid:
            logger.warning(f"Request {request_id}: Validation failed - {error_message}")
            return jsonify({"error": error_message}), 400
        
        # Extract parameters
        audio_file = request.files['audio']
        model_name = request.form.get('model', 'wave2vec')
        phoneme = request.form.get('phoneme', '')
        model_id = request.form.get('model_id', '')
        
        logger.info(f"Request {request_id}: Processing with {model_name}, phoneme: '{phoneme}', model_id: '{model_id}'")
        
        try:
            # Read audio data with improved method
            audio_data, sample_rate = read_audio_file_improved(audio_file)
            logger.info(f"Audio loaded: {len(audio_data)} samples at {sample_rate}Hz")
        except Exception as e:
            logger.exception(f"Request {request_id}: Failed to read audio data")
            return jsonify({"error": f"Failed to read audio data: {str(e)}"}), 400
        
        # Determine the correct model ID based on phoneme if not provided
        if not model_id and phoneme and phoneme in PHONEME_MODEL_MAPPING:
            model_id = PHONEME_MODEL_MAPPING[phoneme].get(model_name)
            logger.info(f"Auto-selected model_id: {model_id} for phoneme: {phoneme}")
        
        # Load appropriate model
        model, processor = load_model(model_name, model_id)
        if model is None:
            logger.error(f"Request {request_id}: Failed to load {model_name} model (ID: {model_id})")
            return jsonify({"error": f"Failed to load {model_name} model"}), 500
        
        # Predict pronunciation
        result = predict_audio(model, processor, audio_data, sample_rate, model_name, phoneme)
        
        # Add request metadata
        result["request_id"] = request_id
        if phoneme:
            result["phoneme"] = phoneme
        result["timestamp"] = datetime.now().isoformat()
        result["model_id"] = model_id if model_id else f"default_{model_name}"
        
        # Log result summary
        if "error" in result:
            logger.warning(f"Request {request_id}: Error in processing - {result['error']}")
        else:
            logger.info(f"Request {request_id}: Analysis complete - "
                      f"Pronunciation {'correct' if result.get('correct', False) else 'incorrect'} "
                      f"with {result.get('confidence', 0):.2f}% confidence")

        return jsonify(result)

    except Exception as e:
        logger.exception("Unhandled error in analyze_tajweed endpoint")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": list(models.keys())
    })

@app.route('/models', methods=['GET'])
def list_models():
    """
    List available models and their status - now includes CNN models
    """
    loaded_models = {}
    for key in models:
        if '_' in key:
            model_type, model_id = key.split('_', 1)
            if model_type not in loaded_models:
                loaded_models[model_type] = []
            loaded_models[model_type].append(model_id)

    return jsonify({
        "available_models": {
            "wave2vec": "wave2vec" in loaded_models,
            "whisper": "whisper" in loaded_models,
            "cnn": "cnn" in loaded_models
        },
        "supported_phonemes": SUPPORTED_PHONEMES,
        "phoneme_model_mapping": PHONEME_MODEL_MAPPING,
        "loaded_models": loaded_models
    })

@app.route('/status', methods=['GET'])
def model_status():
    """
    Endpoint to check the loading status of all models - now includes CNN models
    """
    total_models = 0
    loaded_models = 0
    model_status = {}

    for phoneme, model_types in PHONEME_MODEL_MAPPING.items():
        for model_type, model_id in model_types.items():
            total_models += 1
            cache_key = f"{model_type}_{model_id}"

            is_loaded = cache_key in models and models[cache_key] is not None
            model_status[cache_key] = {
                "loaded": is_loaded,
                "phoneme": phoneme,
                "model_type": model_type,
                "model_id": model_id
            }

            if is_loaded:
                loaded_models += 1

    loading_percentage = (loaded_models / total_models * 100) if total_models > 0 else 0

    gpu_info = {}
    if torch.cuda.is_available():
        try:
            gpu_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
                "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
            }
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            gpu_info = {"error": str(e)}

    return jsonify({
        "status": "loading" if loading_percentage < 100 else "ready",
        "loading_progress": {
            "total_models": total_models,
            "loaded_models": loaded_models,
            "percentage": round(loading_percentage, 1)
        },
        "model_status": model_status,
        "gpu_info": gpu_info,
        "server_time": datetime.now().isoformat()
    })

def preload_all_models():
    """
    Preload all models when the server starts - now includes CNN models
    """
    logger.info("Preloading all models...")

    for phoneme, model_types in PHONEME_MODEL_MAPPING.items():
        for model_type, model_id in model_types.items():
            try:
                logger.info(f"Preloading {model_type} model for phoneme '{phoneme}': {model_id}")
                model, processor = load_model(model_type, model_id)
                if model:
                    logger.info(f"Successfully preloaded {model_type} model for phoneme '{phoneme}'")
                else:
                    logger.error(f"Failed to preload {model_type} model for phoneme '{phoneme}'")
            except Exception as e:
                logger.exception(f"Error preloading {model_type} model for phoneme '{phoneme}': {e}")


@app.route('/save-audio', methods=['POST'])
def save_audio():
    """
    Endpoint to save audio file to the input directory
    """
    try:
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"
        logger.info(f"Request {request_id}: Received save-audio request")
        
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        letter = request.form.get('letter', '')
        
        if not letter:
            return jsonify({"error": "Please specify a letter name"}), 400
        
        filename = "input.wav"
        filepath = os.path.join(INPUT_DIR, filename)
        
        # Save using improved audio handling
        try:
            # Read and process audio
            audio_data, sample_rate = read_audio_file_improved(audio_file)
            
            # Save as high-quality WAV
            sf.write(filepath, audio_data, sample_rate, subtype='PCM_16')
            
            logger.info(f"Request {request_id}: Saved processed audio to {filepath}")
            
            return jsonify({
                "success": True,
                "message": "Audio file saved successfully",
                "filename": filename,
                "filepath": filepath,
                "letter": letter
            })
            
        except Exception as e:
            logger.exception(f"Error processing and saving audio: {str(e)}")
            # Fallback to direct save
            audio_file.save(filepath)
            logger.info(f"Request {request_id}: Saved raw audio to {filepath}")
            
            return jsonify({
                "success": True,
                "message": "Audio file saved (raw format)",
                "filename": filename,
                "filepath": filepath,
                "letter": letter
            })
        
    except Exception as e:
        logger.exception(f"Error saving audio file: {str(e)}")
        return jsonify({"error": f"Failed to save audio file: {str(e)}"}), 500

# Keep all your existing endpoints (segment_audio, analyze_all_segments, etc.)
# [Include all your existing endpoint code here...]

if __name__ == '__main__':
    # Run debug check on startup
    debug_model_paths()
    
    # Preload all models at startup
    preload_all_models()

    # Start the server
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)





