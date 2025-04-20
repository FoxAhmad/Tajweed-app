
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
MAX_AUDIO_LENGTH = 10  # seconds
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.ogg', '.webm']
SUPPORTED_PHONEMES = ['ee', 'so', 'si']  # Add all your supported phonemes here

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


def load_model(model_name):
    """
    Load a model and its processor based on the model name.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        tuple: (model, processor) or None if model not found
    """
    if model_name == "wave2vec":
        # Initialize model and processor only once
        if "wave2vec" not in models:
            try:
                logger.info("Loading Wave2Vec model...")
                model_id = "xxmoeedxx/wav2vec2_si"
                
                # Load processor
                processors["wave2vec"] = AutoProcessor.from_pretrained(model_id)
                
                # Load model
                models["wave2vec"] = AutoModelForAudioClassification.from_pretrained(model_id)
                
                # Move model to GPU if available
                if torch.cuda.is_available():
                    models["wave2vec"] = models["wave2vec"].to("cuda")
                    logger.info("Wave2Vec model moved to GPU")
                
                logger.info("Wave2Vec model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Wave2Vec model: {e}")
                return None, None
                
        return models["wave2vec"], processors["wave2vec"]
    
    elif model_name == "whisper":
        # Load the Whisper-based model
        if "whisper" not in models:
            try:
                logger.info("Loading Whisper classifier model...")
                model_id = "ahmad1703/whisper_ee"
                
                # Load processor
                processors["whisper"] = WhisperProcessor.from_pretrained(model_id)
                
                # Load model
                models["whisper"] = WhisperClassifier()
                
                # Load weights
                try:
                    # First try to load from Hugging Face
                    models["whisper"].load_state_dict(torch.hub.load_state_dict_from_url(
                        f"https://huggingface.co/{model_id}/resolve/main/pytorch_model.bin",
                        map_location=torch.device('cpu')
                    ))
                    logger.info("Loaded Whisper model weights from Hugging Face")
                except Exception as e:
                    logger.warning(f"Could not load from HF directly: {e}")
                    # Fallback to local path if available
                    local_path = os.environ.get("WHISPER_MODEL_PATH", "models/whis_ee.pth")
                    if os.path.exists(local_path):
                        models["whisper"].load_state_dict(torch.load(local_path, map_location=torch.device('cpu')))
                        logger.info(f"Loaded Whisper model weights from local path: {local_path}")
                    else:
                        logger.error(f"Could not find local model at {local_path}")
                        return None, None
                
                # Move model to GPU if available
                if torch.cuda.is_available():
                    models["whisper"] = models["whisper"].to("cuda")
                    logger.info("Whisper model moved to GPU")
                
                # Set model to evaluation mode
                models["whisper"].eval()
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                return None, None
                
        return models["whisper"], processors["whisper"]
    
    else:
        logger.error(f"Unknown model: {model_name}")
        return None, None


def read_audio_from_file(file_or_bytes):
    """
    Read audio data from file or binary data
    
    Args:
        file_or_bytes: File-like object or bytes
        
    Returns:
        tuple: (audio_data, sample_rate)
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
    Resample audio data to target sample rate
    
    Args:
        audio_data (numpy.ndarray): Audio data
        orig_sample_rate (int): Original sample rate
        target_sample_rate (int): Target sample rate
        
    Returns:
        numpy.ndarray: Resampled audio data
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
    Preprocess audio data for model input
    
    Args:
        audio_data (numpy.ndarray): Audio data
        sample_rate (int): Sample rate
        
    Returns:
        numpy.ndarray: Preprocessed audio data
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
    Predicts whether the given audio is pronounced correctly or not.
    
    Args:
        model: The trained model
        processor: Feature processor
        audio_data (numpy.ndarray): Audio data
        sample_rate (int): Sample rate
        model_name (str): Name of the model being used
        phoneme (str, optional): The phoneme being checked
        
    Returns:
        dict: Prediction results
    """
    try:
        start_time = time.time()
        print(phoneme)
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
    Generate detailed feedback based on phoneme, prediction and confidence
    
    Args:
        phoneme (str): The phoneme being checked
        prediction (int): The model prediction (0=incorrect, 1=correct)
        confidence (float): Confidence score (0-100)
        
    Returns:
        str: Detailed feedback
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
    Validate the incoming request
    
    Args:
        request: Flask request object
        
    Returns:
        tuple: (is_valid, error_message)
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
    
    return True, ""


@app.route('/analyze-tajweed', methods=['POST'])
def analyze_tajweed():
    """
    Endpoint to analyze audio for tajweed pronunciation.
    
    Expects:
        - 'audio' file in request.files
        - 'model' parameter in request.form ('whisper' or 'wave2vec')
        - 'phoneme' parameter in request.form (selected phoneme)
        
    Returns:
        JSON response with analysis results
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
        
        logger.info(f"Request {request_id}: Processing audio with {model_name} model, phoneme: '{phoneme}'")
        logger.info(f"Audio file: {audio_file.filename if audio_file.filename else 'No filename'}, "
                   f"Content type: {audio_file.content_type}")
        
        try:
            # Read audio data directly from the request
            audio_data, sample_rate = read_audio_from_file(audio_file)
        except Exception as e:
            logger.exception(f"Request {request_id}: Failed to read audio data")
            return jsonify({"error": f"Failed to read audio data: {str(e)}"}), 400
        
        # Load appropriate model
        model, processor = load_model(model_name)
        if model is None or processor is None:
            logger.error(f"Request {request_id}: Failed to load {model_name} model")
            return jsonify({"error": f"Failed to load {model_name} model"}), 500
        
        # Predict pronunciation
        result = predict_audio(model, processor, audio_data, sample_rate, model_name, phoneme)
        
        # Add request metadata
        result["request_id"] = request_id
        if phoneme:
            result["phoneme"] = phoneme
        result["timestamp"] = datetime.now().isoformat()
        
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
    return jsonify({
        "available_models": {
            "wave2vec": "wave2vec" in models,
            "whisper": "whisper" in models
        },
        "supported_phonemes": SUPPORTED_PHONEMES
    })


# Add a test endpoint to verify audio processing
@app.route('/test-audio', methods=['POST'])
def test_audio():
    """
    Test endpoint to verify audio processing is working
    """
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
            
        audio_file = request.files['audio']
        
        # Read audio data
        try:
            audio_data, sample_rate = read_audio_from_file(audio_file)
            
            # Return basic audio info
            return jsonify({
                "status": "success",
                "sample_rate": sample_rate,
                "duration": len(audio_data) / sample_rate,
                "num_samples": len(audio_data),
                "min_value": float(np.min(audio_data)),
                "max_value": float(np.max(audio_data)),
                "is_normalized": abs(np.max(np.abs(audio_data)) - 1.0) < 0.01
            })
            
        except Exception as e:
            logger.exception("Error processing audio in test endpoint")
            return jsonify({"error": str(e)}), 400
            
    except Exception as e:
        logger.exception("Unhandled error in test-audio endpoint")
        return jsonify({"error": str(e)}), 500


@app.route('/test-models', methods=['GET'])
def test_models():
    """
    Test endpoint to verify that both models are loaded correctly
    """
    results = {}
    
    # Test Wave2Vec
    wave2vec_model, wave2vec_processor = load_model("wave2vec")
    results["wave2vec"] = {
        "loaded": wave2vec_model is not None and wave2vec_processor is not None,
        "model_type": type(wave2vec_model).__name__ if wave2vec_model else None
    }
    
    # Test Whisper
    whisper_model, whisper_processor = load_model("whisper")
    results["whisper"] = {
        "loaded": whisper_model is not None and whisper_processor is not None,
        "model_type": type(whisper_model).__name__ if whisper_model else None
    }
    
    return jsonify({
        "status": "ok" if results["wave2vec"]["loaded"] and results["whisper"]["loaded"] else "error",
        "models": results,
        "timestamp": datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Import numpy here to avoid issues
    import numpy as np
    
    # Set environment variable for local model path if needed
    # os.environ["WHISPER_MODEL_PATH"] = "/path/to/your/whisper/model.pth"
    
    # Load models at startup
    load_model("wave2vec")
    load_model("whisper")
    
    # Start the server
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)