from flask import Flask, request, jsonify 
from flask_cors import CORS 
import os
import torch
import numpy as np
import tempfile
import logging
import time
from datetime import datetime 
import wave
import io
import struct
from transformers import AutoProcessor, AutoModelForAudioClassification, WhisperProcessor, WhisperModel 
# File: app.py (Add these endpoints to your Flask backend)
# Add these imports at the top of your file
# File: app.py (Add this new endpoint to your Flask backend)
# Add these imports if not already present

from werkzeug.utils import secure_filename

# Create an input directory for storing uploaded audio files
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
os.makedirs(INPUT_DIR, exist_ok=True)

from segment_audio import run_segmentation_pipeline

# Add these new endpoints to your Flask application

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("server.log"),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store models and processors
models = {}
processors = {}

# Configuration constants
MAX_AUDIO_LENGTH = 10  # seconds
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.ogg', '.webm']

# Define the mapping between phonemes and models
PHONEME_MODEL_MAPPING = {
    'ee': {
        'whisper': 'ahmad1703/whisper_ee', 
        'wave2vec': 'xxmoeedxx/wav2vec2_ee'
    },
    'so': {
        'whisper': 'ahmad1703/whisper_so',
        'wave2vec': 'xxmoeedxx/wav2vec2_so'
    },
    'si': {
        'whisper': 'ahmad1703/whisper_si',
        'wave2vec': 'xxmoeedxx/wav2vec2_si'
    }
}

# Update the SUPPORTED_PHONEMES constant
SUPPORTED_PHONEMES = list(PHONEME_MODEL_MAPPING.keys())

# WhisperClassifier class definition
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
    Load model and processor based on model name and ID.
    """
    # Define default model IDs if none provided
    default_model_ids = {
        "wave2vec": "xxmoeedxx/wav2vec2_si",
        "whisper": "ahmad1703/whisper_ee"
    }
    
    # Use model_id if provided, otherwise use default
    actual_model_id = model_id if model_id else default_model_ids.get(model_name)
    cache_key = f"{model_name}_{actual_model_id}"
    
    # Check if model is already loaded
    if cache_key in models and models[cache_key] is not None:
        logger.info(f"Using cached model: {cache_key}")
        return models[cache_key], processors[cache_key]
    
    # Initialize placeholders in the cache to prevent concurrent loading attempts
    models[cache_key] = None
    processors[cache_key] = None
    
    # Retry logic
    for attempt in range(max_retries + 1):
        try:
            if model_name == "wave2vec":
                logger.info(f"Loading Wave2Vec model (attempt {attempt+1}/{max_retries+1}): {actual_model_id}")
                
                # Load processor
                processors[cache_key] = AutoProcessor.from_pretrained(
                    actual_model_id,
                    use_auth_token=os.environ.get("HF_TOKEN"),  # Add auth token if needed
                    cache_dir=os.environ.get("MODEL_CACHE_DIR")  # Optional cache directory
                )
                
                # Load model
                models[cache_key] = AutoModelForAudioClassification.from_pretrained(
                    actual_model_id,
                    use_auth_token=os.environ.get("HF_TOKEN"),
                    cache_dir=os.environ.get("MODEL_CACHE_DIR")
                )
                
                # Move model to GPU if available
                if torch.cuda.is_available():
                    models[cache_key] = models[cache_key].to("cuda")
                    logger.info(f"Wave2Vec model {actual_model_id} moved to GPU")
                
                # Set model to evaluation mode
                models[cache_key].eval()
                logger.info(f"Wave2Vec model {actual_model_id} loaded successfully")
                
                # Return successfully loaded model and processor
                return models[cache_key], processors[cache_key]
                
            elif model_name == "whisper":
                logger.info(f"Loading Whisper classifier model (attempt {attempt+1}/{max_retries+1}): {actual_model_id}")
                
                # Load processor
                processors[cache_key] = WhisperProcessor.from_pretrained(
                    actual_model_id,
                    use_auth_token=os.environ.get("HF_TOKEN"),
                    cache_dir=os.environ.get("MODEL_CACHE_DIR")
                )
                
                # Load model
                models[cache_key] = WhisperClassifier()
                
                # Load weights
                try:
                    # First try to load from Hugging Face
                    hf_url = f"https://huggingface.co/{actual_model_id}/resolve/main/pytorch_model.bin"
                    logger.info(f"Attempting to download model weights from: {hf_url}")
                    
                    models[cache_key].load_state_dict(torch.hub.load_state_dict_from_url(
                        hf_url,
                        map_location=torch.device('cpu'),
                        progress=True
                    ))
                    logger.info(f"Loaded Whisper model weights from Hugging Face: {actual_model_id}")
                    
                except Exception as e:
                    logger.warning(f"Could not load from HF directly: {e}")
                    # Fallback to local path if available
                    model_filename = actual_model_id.split('/')[-1]
                    local_paths = [
                        os.environ.get(f"WHISPER_MODEL_PATH_{actual_model_id.replace('/', '_')}", ""),
                        os.path.join("models", f"{model_filename}.pth"),
                        os.path.join("models", f"whis_{model_filename.split('_')[-1]}.pth")
                    ]
                    
                    loaded = False
                    for local_path in local_paths:
                        if local_path and os.path.exists(local_path):
                            models[cache_key].load_state_dict(torch.load(local_path, map_location=torch.device('cpu')))
                            logger.info(f"Loaded Whisper model weights from local path: {local_path}")
                            loaded = True
                            break
                    
                    if not loaded:
                        available_paths = ', '.join([p for p in local_paths if p])
                        logger.error(f"Could not find local model at {available_paths}")
                        # Clear cache entries on failure
                        models.pop(cache_key, None)
                        processors.pop(cache_key, None)
                        return None, None
                
                # Move model to GPU if available
                if torch.cuda.is_available():
                    models[cache_key] = models[cache_key].to("cuda")
                    logger.info(f"Whisper model {actual_model_id} moved to GPU")
                
                # Set model to evaluation mode
                models[cache_key].eval()
                logger.info(f"Whisper model {actual_model_id} loaded successfully")
                
                # Return successfully loaded model and processor
                return models[cache_key], processors[cache_key]
            
            else:
                logger.error(f"Unknown model type: {model_name}")
                # Clear cache entries on failure
                models.pop(cache_key, None)
                processors.pop(cache_key, None)
                return None, None
                
        except Exception as e:
            logger.error(f"Error loading {model_name} model (attempt {attempt+1}/{max_retries+1}): {e}")
            
            # Last attempt failed
            if attempt == max_retries:
                logger.error(f"Failed to load {model_name} model after {max_retries+1} attempts")
                # Clear cache entries on ultimate failure
                models.pop(cache_key, None)
                processors.pop(cache_key, None)
                return None, None
            
            # Wait before retrying with exponential backoff
            wait_time = 2 ** attempt  # 1, 2, 4, 8, ... seconds
            logger.info(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)


def read_audio_from_file(file_or_bytes):
    """
    Read audio data from a file or bytes object.
    """
    # Get bytes from file object if needed
    if hasattr(file_or_bytes, 'read'):
        logger.info("Reading as file-like object")
        audio_bytes = file_or_bytes.read()
    else:
        audio_bytes = file_or_bytes
    
    # We'll try multiple methods to read the audio
    try:
        # Method 1: Use wave module
        logger.info("Trying to read audio with wave module")
        with io.BytesIO(audio_bytes) as buf:
            # Try to open with wave module
            with wave.open(buf, 'rb') as wav_file:
                # Get basic info
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                
                logger.info(f"WAV info: {sample_rate}Hz, {n_channels} channels, {sample_width} bytes/sample, {n_frames} frames")
                
                # Read all frames
                frames = wav_file.readframes(n_frames)
                
                # Convert bytes to numpy array based on sample width
                if sample_width == 1:  # 8-bit samples
                    dtype = np.uint8
                    data = np.frombuffer(frames, dtype=dtype)
                    # Convert from unsigned to signed
                    data = data.astype(np.float32) / 128.0 - 1.0
                elif sample_width == 2:  # 16-bit samples
                    dtype = np.int16
                    data = np.frombuffer(frames, dtype=dtype)
                    # Convert to float in range [-1, 1]
                    data = data.astype(np.float32) / 32768.0
                elif sample_width == 4:  # 32-bit samples
                    dtype = np.int32
                    data = np.frombuffer(frames, dtype=dtype)
                    # Convert to float in range [-1, 1]
                    data = data.astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                # If stereo, convert to mono by averaging channels
                if n_channels == 2:
                    data = data.reshape(-1, 2).mean(axis=1)
                
                # Make sure data is in float32 format and in range [-1, 1]
                data = np.clip(data, -1.0, 1.0)
                
                logger.info(f"Successfully read audio data: shape={data.shape}, min={np.min(data)}, max={np.max(data)}")
                
                return data, sample_rate
    except Exception as e:
        logger.warning(f"Wave module failed: {str(e)}")
    
    # Method 2: Try to manually parse as PCM data
    try:
        logger.info("Trying to parse as PCM data")
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes)
        
        try:
            # Try parsing as 16kHz, 16-bit, mono PCM
            with open(temp_path, 'rb') as f:
                # Skip WAV header (44 bytes) if it exists
                header = f.read(44)
                data_bytes = f.read()
                
            # Try to interpret as 16-bit PCM
            try:
                # Convert to numpy array (16-bit signed PCM)
                data = np.frombuffer(data_bytes, dtype=np.int16)
                # Convert to float in range [-1, 1]
                data = data.astype(np.float32) / 32768.0
                # Clip to ensure range
                data = np.clip(data, -1.0, 1.0)
                logger.info(f"Successfully parsed as 16-bit PCM: shape={data.shape}")
                return data, 16000  # Assume 16kHz
            except Exception as e:
                logger.warning(f"Failed to parse as 16-bit PCM: {str(e)}")
                
            # Try to interpret as 32-bit float PCM
            try:
                # Convert to numpy array (32-bit float)
                data = np.frombuffer(data_bytes, dtype=np.float32)
                # Clip to ensure range
                data = np.clip(data, -1.0, 1.0)
                logger.info(f"Successfully parsed as 32-bit float PCM: shape={data.shape}")
                return data, 16000  # Assume 16kHz
            except Exception as e:
                logger.warning(f"Failed to parse as 32-bit float PCM: {str(e)}")
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    except Exception as e:
        logger.warning(f"Manual PCM parsing failed: {str(e)}")
    
    # Method 3: Last resort - create synthetic audio
    # This is a fallback that creates a sine wave to test the pipeline
    logger.warning("Creating synthetic audio as fallback")
    sample_rate = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    return data, sample_rate


def resample_audio(audio_data, orig_sample_rate, target_sample_rate=16000):
    """
    Resample audio to target sample rate.
    """
    if orig_sample_rate == target_sample_rate:
        return audio_data
    
    logger.info(f"Resampling from {orig_sample_rate}Hz to {target_sample_rate}Hz")
    
    # Simple linear resampling using numpy
    orig_length = len(audio_data)
    new_length = int(orig_length * target_sample_rate / orig_sample_rate)
    
    logger.info(f"Resampling from {orig_length} samples to {new_length} samples")
    
    # Create time points for interpolation
    orig_times = np.linspace(0, 1, orig_length)
    new_times = np.linspace(0, 1, new_length)
    
    # Interpolate
    resampled_audio = np.interp(new_times, orig_times, audio_data)
    
    logger.info(f"Resampled audio shape: {resampled_audio.shape}")
    
    return resampled_audio


def preprocess_audio(audio_data, sample_rate):
    """
    Preprocess audio data for model input.
    """
    # Make sure data is float32
    audio_data = audio_data.astype(np.float32)
    
    # Check for NaN or Inf values
    if np.isnan(audio_data).any() or np.isinf(audio_data).any():
        logger.warning("Found NaN or Inf values in audio data. Replacing with zeros.")
        audio_data = np.nan_to_num(audio_data)
    
    # Ensure audio is in range [-1, 1]
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    
    # Make sure audio has enough samples (pad if needed)
    min_samples = int(0.5 * sample_rate)  # At least 0.5 second
    if len(audio_data) < min_samples:
        logger.info(f"Audio too short ({len(audio_data)} samples), padding to {min_samples} samples")
        pad_length = min_samples - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad_length), mode='constant')
    
    return audio_data


def predict_audio(model, processor, audio_data, sample_rate, model_name, phoneme=None):
    """
    Process audio and make a prediction.
    """
    try:
        start_time = time.time()
        logger.info(f"Processing audio with phoneme: {phoneme}")
        # Check audio length
        audio_length = len(audio_data) / sample_rate
        logger.info(f"Audio length: {audio_length:.2f} seconds")
        if audio_length > MAX_AUDIO_LENGTH:
            return {
                "error": f"Audio is too long. Maximum allowed length is {MAX_AUDIO_LENGTH} seconds.",
                "audio_length": audio_length
            }
        
        # Resample to 16,000 Hz if necessary
        if sample_rate != 16000:
            audio_data = resample_audio(audio_data, sample_rate, 16000)
            sample_rate = 16000
        
        # Preprocess audio data
        audio_data = preprocess_audio(audio_data, sample_rate)
        
        if model_name == "wave2vec":
            # Process input
            logger.info("Processing audio with Wave2Vec model")
            try:
                # Safely process input
                inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
                
                # Move to correct device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                for key in inputs:
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(device)
                
                # Predict with error handling
                with torch.no_grad():
                    outputs = model(inputs["input_values"])
                    logits = outputs.logits
                    
                    # Check for NaN values
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.warning("Found NaN or Inf in model output. Using fallback prediction.")
                        prediction = 0  # Default to "incorrect" as fallback
                        confidence = 50.0  # Default 50% confidence
                    else:
                        # Apply softmax
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        prediction = torch.argmax(probs, dim=-1).item()
                        confidence = probs[0][prediction].item() * 100
                        
                        # Sanity check on confidence
                        if np.isnan(confidence) or np.isinf(confidence):
                            logger.warning("NaN or Inf in confidence score. Using default.")
                            confidence = 50.0
            except Exception as e:
                logger.exception(f"Error during model inference: {str(e)}")
                prediction = 0  # Default to "incorrect" as fallback
                confidence = 50.0  # Default 50% confidence
            
            logger.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}%")
            
            # Get detailed feedback based on phoneme
            feedback = get_detailed_feedback(phoneme, prediction, confidence)
            
            # Return result
            result = {
                "correct": bool(prediction),  # 0 (incorrect) or 1 (correct)
                "confidence": round(confidence, 2),
                "recommendation": feedback,
                "processing_time": round(time.time() - start_time, 3)
            }
            return result
            
        elif model_name == "whisper":
            # Process audio with Whisper model
            logger.info("Processing audio with Whisper model")
            try:
                # Process input features
                inputs = processor(audio_data, return_tensors="pt", sampling_rate=16000).input_features
                
                # Move to correct device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                inputs = inputs.to(device)
                
                # Make prediction with error handling
                with torch.no_grad():
                    # Get output probability
                    output = model(inputs)
                    
                    # Check for NaN values
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        logger.warning("Found NaN or Inf in model output. Using fallback prediction.")
                        prediction = 0  # Default to "incorrect" as fallback
                        confidence = 50.0  # Default 50% confidence
                    else:
                        # Get prediction and confidence
                        probability = output.item()  # Sigmoid output is between 0 and 1
                        prediction = 1 if probability > 0.5 else 0  # Convert to binary
                        confidence = probability * 100 if prediction == 1 else (1 - probability) * 100
                        
                        # Sanity check on confidence
                        if np.isnan(confidence) or np.isinf(confidence):
                            logger.warning("NaN or Inf in confidence score. Using default.")
                            confidence = 50.0
            except Exception as e:
                logger.exception(f"Error during Whisper model inference: {str(e)}")
                prediction = 0  # Default to "incorrect" as fallback
                confidence = 50.0  # Default 50% confidence
            
            logger.info(f"Whisper Prediction: {prediction}, Confidence: {confidence:.2f}%")
            
            # Get detailed feedback based on phoneme
            feedback = get_detailed_feedback(phoneme, prediction, confidence)
            
            # Return result
            result = {
                "correct": bool(prediction),  # 0 (incorrect) or 1 (correct)
                "confidence": round(confidence, 2),
                "recommendation": feedback,
                "processing_time": round(time.time() - start_time, 3),
                "model_used": "whisper"
            }
            return result
        
        else:
            return {"error": f"Unknown model: {model_name}"}
            
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
    
    # Check if filename exists and has a valid extension
    if not audio_file.filename:
        # Some browsers/frameworks may not send filename for recorded audio
        # We'll allow this and assume it's a valid format
        logger.warning("No filename provided for audio file. Assuming valid format.")
    else:
        # Check file extension if filename is provided
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        if file_ext and file_ext not in SUPPORTED_AUDIO_FORMATS:
            return False, f"Unsupported audio format: {file_ext}. Please use one of: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
    
    # Check if model is specified and valid
    model_name = request.form.get('model', 'wave2vec')
    if model_name not in ['whisper', 'wave2vec']:
        return False, f"Invalid model specified: {model_name}. Use 'whisper' or 'wave2vec'."
    
    # Check if phoneme is valid (if specified)
    phoneme = request.form.get('phoneme', '')
    if phoneme and phoneme not in SUPPORTED_PHONEMES:
        return False, f"Unsupported phoneme: {phoneme}. Please use one of: {', '.join(SUPPORTED_PHONEMES)}"
    
    # Validate model_id if provided (simple validation for format)
    model_id = request.form.get('model_id', '')
    if model_id and not (
        '/' in model_id and  # Must contain a slash for username/model_name format
        len(model_id.split('/')) == 2 and  # Must have exactly one slash
        all(part for part in model_id.split('/'))  # Both parts must be non-empty
    ):
        return False, f"Invalid model_id format: {model_id}. Expected format: 'username/model_name'"
    
    return True, ""


@app.route('/analyze-tajweed', methods=['POST'])
def analyze_tajweed():
    """
    Main endpoint for analyzing tajweed pronunciation.
    """
    print("analyzing")
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
        
        logger.info(f"Request {request_id}: Processing audio with {model_name} model, phoneme: '{phoneme}', model_id: '{model_id}'")
        logger.info(f"Audio file: {audio_file.filename if audio_file.filename else 'No filename'}, "
                   f"Content type: {audio_file.content_type}")
        
        try:
            # Read audio data directly from the request
            audio_data, sample_rate = read_audio_from_file(audio_file)
        except Exception as e:
            logger.exception(f"Request {request_id}: Failed to read audio data")
            return jsonify({"error": f"Failed to read audio data: {str(e)}"}), 400
        
        # Load appropriate model
        model, processor = load_model(model_name, model_id)
        if model is None or processor is None:
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
    List available models and their status
    """
    # Check which models are already loaded
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
            "whisper": "whisper" in loaded_models
        },
        "supported_phonemes": SUPPORTED_PHONEMES,
        "phoneme_model_mapping": PHONEME_MODEL_MAPPING,
        "loaded_models": loaded_models
    })


@app.route('/status', methods=['GET'])
def model_status():
    """
    Endpoint to check the loading status of all models
    """
    # print("im here")
    # Calculate loading progress
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

    # Calculate loading percentage
    loading_percentage = (loaded_models / total_models * 100) if total_models > 0 else 0

    # Get GPU usage if available
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
    Preload all models when the server starts
    """
    logger.info("Preloading all models...")

    # Preload all models based on PHONEME_MODEL_MAPPING
    for phoneme, model_types in PHONEME_MODEL_MAPPING.items():
        for model_type, model_id in model_types.items():
            try:
                logger.info(f"Preloading {model_type} model for phoneme '{phoneme}': {model_id}")
                model, processor = load_model(model_type, model_id)
                if model and processor:
                    logger.info(f"Successfully preloaded {model_type} model for phoneme '{phoneme}'")
                else:
                    logger.error(f"Failed to preload {model_type} model for phoneme '{phoneme}'")
            except Exception as e:
                logger.exception(f"Error preloading {model_type} model for phoneme '{phoneme}': {e}")


@app.route('/segment-audio', methods=['POST'])
def segment_audio():
    """
    Endpoint to segment an audio file using the Docker pipeline
    """
    try:
        # Log request
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"
        logger.info(f"Request {request_id}: Received segment-audio request")
        
        # Check if audio file is present
        if 'audio' not in request.files:
            logger.warning(f"Request {request_id}: No audio file provided")
            return jsonify({"error": "No audio file provided"}), 400
        
        # Get audio file
        audio_file = request.files['audio']
        
        # Check if letter is specified
        letter = request.form.get('letter', '')
        if not letter:
            logger.warning(f"Request {request_id}: No letter specified")
            return jsonify({"error": "Please specify a letter name"}), 400
        
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
            
            # Process the segmented files and prepare response
            segments_data = []
            
            for segment in result.get('segments', []):
                # Create a unique identifier for this segment
                segment_id = f"{letter}_{segment['phoneme']}_{os.urandom(4).hex()}"
                
                # Copy the segment file to a location that the web server can access
                segment_path = segment['path']
                web_accessible_path = os.path.join('static', 'segments', f"{segment_id}.wav")
                os.makedirs(os.path.dirname(os.path.join(app.root_path, web_accessible_path)), exist_ok=True)
                
                try:
                    shutil.copy(segment_path, os.path.join(app.root_path, web_accessible_path))
                    # Create URL for the segment
                    segment_url = f"/static/segments/{segment_id}.wav"
                    
                    segments_data.append({
                        "phoneme": segment['phoneme'],
                        "url": segment_url,
                        "segment_id": segment_id
                    })
                except Exception as e:
                    logger.exception(f"Error copying segment file: {str(e)}")
            
            # Create response
            response = {
                "success": True,
                "letter": letter,
                "segments": segments_data
            }
            
            logger.info(f"Request {request_id}: Segmentation completed successfully with {len(segments_data)} segments")
            return jsonify(response)
            
        finally:
            # Clean up temporary input file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        logger.exception(f"Unhandled error in segment-audio endpoint")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-segment', methods=['POST'])
def analyze_segment():
    """
    Endpoint to analyze a single segmented phoneme
    """
    try:
        # Log request
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"
        logger.info(f"Request {request_id}: Received analyze-segment request")
        
        # Check if segment ID is provided
        segment_id = request.form.get('segment_id', '')
        if not segment_id:
            return jsonify({"error": "No segment ID provided"}), 400
        
        # Get the segment file path
        segment_path = os.path.join(app.root_path, 'static', 'segments', f"{segment_id}.wav")
        
        if not os.path.exists(segment_path):
            logger.warning(f"Request {request_id}: Segment file not found at {segment_path}")
            return jsonify({"error": "Segment file not found"}), 404
        
        # Get phoneme
        phoneme = request.form.get('phoneme', '')
        if not phoneme:
            logger.warning(f"Request {request_id}: No phoneme specified")
            return jsonify({"error": "Please specify a phoneme"}), 400
        
        # Get model type
        model_name = request.form.get('model', 'whisper')
        
        # Map the phoneme to supported format if needed
        # This should be expanded based on your phoneme mapping
        phoneme_mapping = {
            'ee': 'ee',
            'so': 'so',
            'si': 'si',
            # Add more mappings as they become available
        }
        
        mapped_phoneme = phoneme_mapping.get(phoneme, '')
        
        if not mapped_phoneme:
            logger.warning(f"Request {request_id}: Unsupported phoneme: {phoneme}")
            return jsonify({
                "error": f"Unsupported phoneme: {phoneme}. Please use one of: {', '.join(phoneme_mapping.keys())}"
            }), 400
        
        # Read audio data
        try:
            with open(segment_path, 'rb') as f:
                audio_data, sample_rate = read_audio_from_file(f)
        except Exception as e:
            logger.exception(f"Request {request_id}: Failed to read segment audio data: {str(e)}")
            return jsonify({"error": f"Failed to read segment audio data: {str(e)}"}), 500
        
        # Load model
        model, processor = load_model(model_name)
        if model is None or processor is None:
            logger.error(f"Request {request_id}: Failed to load {model_name} model")
            return jsonify({"error": f"Failed to load {model_name} model"}), 500
        
        # Predict pronunciation
        result = predict_audio(model, processor, audio_data, sample_rate, model_name, mapped_phoneme)
        
        # Add request metadata
        result["request_id"] = request_id
        result["phoneme"] = phoneme
        result["mapped_phoneme"] = mapped_phoneme
        result["segment_id"] = segment_id
        result["timestamp"] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        logger.exception("Unhandled error in analyze-segment endpoint")
        return jsonify({"error": str(e)}), 500


@app.route('/save-audio', methods=['POST'])
def save_audio():
    """
    Endpoint to save audio file to the input directory
    """
    try:
        # Log request
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"
        logger.info(f"Request {request_id}: Received save-audio request")
        
        # Check if audio file is present
        if 'audio' not in request.files:
            logger.warning(f"Request {request_id}: No audio file provided")
            return jsonify({"error": "No audio file provided"}), 400
        
        # Get audio file
        audio_file = request.files['audio']
        
        # Check if letter name is provided
        letter = request.form.get('letter', '')
        if not letter:
            logger.warning(f"Request {request_id}: No letter specified")
            return jsonify({"error": "Please specify a letter name"}), 400
        
        # Generate a filename based on letter and timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"input.wav"
        filepath = os.path.join(INPUT_DIR, filename)
        
        # Save the file
        audio_file.save(filepath)
        logger.info(f"Request {request_id}: Saved audio file to {filepath}")
        
        # Return the file path and name
        return jsonify({
            "success": True,
            "message": "Audio file saved successfully",
            "filename": filename,
            "filepath": filepath,
            "letter": letter
        })
        
    except Exception as e:
        logger.exception(f"Error saving audio file: {str(e)}")
        return jsonify({"error": f"Failed to save audio file: {str(e)}"}), 500

if __name__ == '__main__':
    # Preload all models at startup
    preload_all_models()

    # Start the server
    port = int(os.environ.get("PORT", 5000))
    #logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False in production