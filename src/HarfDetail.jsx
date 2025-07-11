// File: src/HarfDetail.jsx (Updated with CNN support)
import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Play, MicIcon, ArrowLeft, Volume2, Award, RotateCcw, Layers } from "lucide-react";
import "./HarfDetail.css";
import AudioService from "./services/AudioService";

// API Configuration
const API_BASE_URL = 'http://localhost:5000';
const audioService = new AudioService(API_BASE_URL);

// The phonetic mapping of Arabic letters (unchanged)
const phoneticMap = {
  'ا': { name: 'alif', phonemes: ['a', 'l', 'i', 'f'] },
  'ب': { name: 'ba', phonemes: ['b', 'aa'] },
  'ت': { name: 'ta', phonemes: ['t', 'aa'] },
  'ث': { name: 'sa', phonemes: ['s', 'aa'] },
  'ج': { name: 'jeem', phonemes: ['j', 'ee', 'm'] },
  'ح': { name: 'hha', phonemes: ['hh', 'aa'] },
  'خ': { name: 'kha', phonemes: ['kh', 'aa'] },
  'د': { name: 'daal', phonemes: ['d', 'aa', 'l'] },
  'ذ': { name: 'zhaal', phonemes: ['zh', 'aa', 'l'] },
  'ر': { name: 'raa', phonemes: ['r', 'aa'] },
  'ز': { name: 'zaa', phonemes: ['z', 'aa'] },
  'س': { name: 'seen', phonemes: ['si', 'ee', 'n'] },
  'ش': { name: 'sheen', phonemes: ['sh', 'ee', 'n'] },
  'ص': { name: 'saad', phonemes: ['so', 'aa', 'd'] },
  'ض': { name: 'daad', phonemes: ['du', 'aa', 'd'] },
  'ط': { name: 'tua', phonemes: ['tu', 'aa'] },
  'ظ': { name: 'zua', phonemes: ['zu', 'aa'] },
  'ع': { name: 'aain', phonemes: ['aa', 'ee', 'n'] },
  'غ': { name: 'ghain', phonemes: ['gh', 'aa', 'ee', 'n'] },
  'ف': { name: 'faa', phonemes: ['f', 'aa'] },
  'ق': { name: 'qaaf', phonemes: ['qa', 'aa', 'f'] },
  'ك': { name: 'kaaf', phonemes: ['ka', 'aa', 'f'] },
  'ل': { name: 'laam', phonemes: ['l', 'aa', 'm'] },
  'م': { name: 'meem', phonemes: ['m', 'ee', 'm'] },
  'ن': { name: 'noon', phonemes: ['n', 'oo', 'n'] },
  'ه': { name: 'haa', phonemes: ['h', 'aa'] },
  'و': { name: 'wao', phonemes: ['wa', 'aa', 'o'] },
  'ی': { name: 'yaa', phonemes: ['y', 'aa'] }
};

// Updated map to include CNN models for supported phonemes
const supportedPhonemeMap = {
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
};

const haroof = [
  "ا", "ب", "ت", "ث", "ج", "ح", "خ",
  "د", "ذ", "ر", "ز", "س", "ش", "ص",
  "ض", "ط", "ظ", "ع", "غ", "ف", "ق",
  "ك", "ل", "م", "ن", "و", "ه", "ی"
];

const HarfDetail = () => {
  const { harfId } = useParams();
  const navigate = useNavigate();
  const harf = haroof[parseInt(harfId)] || "ا";  // Default to alif if invalid
  const harfInfo = phoneticMap[harf] || { name: 'unknown', phonemes: [] };
  
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [results, setResults] = useState({});
  const [isProcessing, setIsProcessing] = useState(false);
  const [overallScore, setOverallScore] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [segments, setSegments] = useState([]);
  const [segmentedAudio, setSegmentedAudio] = useState(null);
  const [usePipeline, setUsePipeline] = useState(true);
  const [selectedModel, setSelectedModel] = useState('cnn'); // Default to CNN for faster feedback
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Get available models for current harf's phonemes
  const getAvailableModels = () => {
    const availableModels = new Set();
    
    harfInfo.phonemes.forEach(phoneme => {
      if (supportedPhonemeMap[phoneme]) {
        Object.keys(supportedPhonemeMap[phoneme]).forEach(model => {
          availableModels.add(model);
        });
      }
    });
    
    return Array.from(availableModels);
  };

  const availableModels = getAvailableModels();

  // Ensure selected model is available for this harf
  useEffect(() => {
    if (!availableModels.includes(selectedModel) && availableModels.length > 0) {
      setSelectedModel(availableModels[0]);
    }
  }, [harfId, availableModels, selectedModel]);

  // Get model display name and description
  const getModelInfo = (modelType) => {
    const modelInfo = {
      'cnn': {
        name: 'CNN',
        description: 'Fast spectrogram analysis',
        speed: 'Very Fast',
        accuracy: 'Good'
      },
      'wave2vec': {
        name: 'Wave2Vec',
        description: 'Precision phoneme analysis',
        speed: 'Medium',
        accuracy: 'High'
      },
      'whisper': {
        name: 'Whisper',
        description: 'Contextual analysis',
        speed: 'Slower',
        accuracy: 'Very High'
      }
    };
    
    return modelInfo[modelType] || { name: modelType, description: 'AI Model', speed: 'Unknown', accuracy: 'Unknown' };
  };

// Utility function to convert audio blob to WAV format (unchanged)
const convertToWav = async (audioBlob, sampleRate = 16000) => {
  return new Promise((resolve, reject) => {
    try {
      // Create AudioContext
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: sampleRate
      });
      
      // Create a file reader to read the blob
      const reader = new FileReader();
      
      reader.onload = async (event) => {
        try {
          // Decode the audio data
          const audioData = await audioContext.decodeAudioData(event.target.result);
          
          // Get audio buffer data
          const numberOfChannels = audioData.numberOfChannels;
          const length = audioData.length;
          const sampleRate = audioData.sampleRate;
          const channelData = [];
          
          // Get data for each channel
          for (let channel = 0; channel < numberOfChannels; channel++) {
            channelData.push(audioData.getChannelData(channel));
          }
          
          // Create WAV file
          const wavFile = createWaveFile(channelData, {
            sampleRate: sampleRate,
            isFloat: false,  // Use PCM format (16-bit)
            numChannels: numberOfChannels
          });
          
          // Create new blob with WAV format
          const wavBlob = new Blob([wavFile], { type: 'audio/wav' });
          resolve(wavBlob);
        } catch (decodeError) {
          console.error('Error decoding audio:', decodeError);
          // If decoding fails, just return the original blob
          resolve(audioBlob);
        }
      };
      
      reader.onerror = (error) => {
        console.error('Error reading file:', error);
        reject(error);
      };
      
      // Read the blob as array buffer
      reader.readAsArrayBuffer(audioBlob);
    } catch (error) {
      console.error('Error converting to WAV:', error);
      // Return original blob if conversion fails
      resolve(audioBlob);
    }
  });
};

// Function to create a Wave file buffer (unchanged)
const createWaveFile = (channelData, options) => {
  const { sampleRate = 16000, isFloat = false, numChannels = 1 } = options;
  
  // Calculate bit depth and format code
  const bitDepth = isFloat ? 32 : 16;
  const formatCode = isFloat ? 3 : 1; // 3 for float, 1 for PCM
  
  // Calculate block align and byte rate
  const blockAlign = numChannels * (bitDepth / 8);
  const byteRate = sampleRate * blockAlign;
  
  // Find the max length across all channels
  const maxLength = Math.max(...channelData.map(channel => channel.length));
  
  // Calculate data size and file size
  const dataSize = maxLength * numChannels * (bitDepth / 8);
  const fileSize = 44 + dataSize; // 44 bytes for the header
  
  // Create buffer
  const buffer = new ArrayBuffer(fileSize);
  const view = new DataView(buffer);
  
  // Write WAV header
  // "RIFF" chunk descriptor
  writeString(view, 0, 'RIFF');
  view.setUint32(4, fileSize - 8, true); // File size - 8 bytes
  writeString(view, 8, 'WAVE');
  
  // "fmt " sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size (16 bytes)
  view.setUint16(20, formatCode, true); // Format code (1 for PCM, 3 for float)
  view.setUint16(22, numChannels, true); // Number of channels
  view.setUint32(24, sampleRate, true); // Sample rate
  view.setUint32(28, byteRate, true); // Byte rate
  view.setUint16(32, blockAlign, true); // Block align
  view.setUint16(34, bitDepth, true); // Bits per sample
  
  // "data" sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true); // Data size
  
  // Write audio data
  let offset = 44; // Start writing after the header
  
  if (isFloat) {
    // Float32 format (32-bit)
    for (let i = 0; i < maxLength; i++) {
      for (let channel = 0; channel < numChannels; channel++) {
        const sample = i < channelData[channel].length ? channelData[channel][i] : 0;
        view.setFloat32(offset, sample, true); // true for little-endian
        offset += 4; // 4 bytes per float32
      }
    }
  } else {
    // PCM format (16-bit)
    for (let i = 0; i < maxLength; i++) {
      for (let channel = 0; channel < numChannels; channel++) {
        const sample = i < channelData[channel].length ? channelData[channel][i] : 0;
        // Convert float -1.0 to 1.0 to 16-bit PCM
        const pcmSample = Math.max(-1, Math.min(1, sample));
        const int16Sample = pcmSample < 0 
          ? pcmSample * 32768 
          : pcmSample * 32767;
        view.setInt16(offset, int16Sample, true); // true for little-endian
        offset += 2; // 2 bytes per int16
      }
    }
  }
  
  return buffer;
};

// Helper function to write strings to DataView (unchanged)
const writeString = (view, offset, string) => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};

// Updated startRecording function (unchanged)
const startRecording = async () => {
  try {
    setErrorMessage(null);
    audioChunksRef.current = [];
    
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      } 
    });
    
    mediaRecorderRef.current = new MediaRecorder(stream, {
      mimeType: 'audio/webm'
    });
    
    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };
    
    mediaRecorderRef.current.onstop = async () => {
      try {
        // Create blob from recorded chunks
        const rawBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        // Convert to WAV format for better compatibility
        const wavBlob = await convertToWav(rawBlob);
        const audioUrl = URL.createObjectURL(wavBlob);
        
        // Save to local state for preview
        setAudioBlob(wavBlob);
        setAudioUrl(audioUrl);
        
        // Reset segments and results when recording new audio
        setSegments([]);
        setSegmentedAudio(null);
        setResults({});
        setOverallScore(null);
        
        // Save directly to server input folder
        try {
          const saveResult = await audioService.saveAudioToServer(wavBlob, harfInfo.name);
          console.log('Audio saved to server:', saveResult);
        } catch (saveError) {
          console.error('Failed to save audio to server:', saveError);
          // Continue with local blob since we still have it
        }
      } catch (error) {
        console.error('Error processing recorded audio:', error);
        setErrorMessage('Error processing the recording. Please try again.');
      }
    };
    
    mediaRecorderRef.current.start();
    setIsRecording(true);
  } catch (error) {
    console.error('Error starting recording:', error);
    setErrorMessage('Could not access microphone. Please ensure your browser has permission to use the microphone.');
  }
};

// No changes needed for stopRecording function (unchanged)
const stopRecording = () => {
  if (mediaRecorderRef.current && isRecording) {
    mediaRecorderRef.current.stop();
    setIsRecording(false);
    
    // Stop all audio tracks
    mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
  }
};

// Play reference audio (unchanged)
const playReferenceAudio = () => {
  const audio = new Audio(`/sounds/arabic_letter_voice_${parseInt(harfId) + 1}.wav`);
  audio.play();
};

// Toggle pipeline usage (unchanged)
const togglePipeline = () => {
  setUsePipeline(!usePipeline);
  console.log('Pipeline toggled:', !usePipeline);
};

// Segment the audio using Docker pipeline (unchanged)
const segmentAudio = async () => {
  if (!audioBlob) {
    setErrorMessage('Please record your pronunciation first.');
    return;
  }

  setIsProcessing(true);
  setErrorMessage(null);

  try {
    // Run the segmentation pipeline with the selected model
    const result = await audioService.segmentAudio(audioBlob, harfInfo.name, selectedModel);
    
    // Store the segments
    setSegments(result.segments);
    console.log('Segmentation result:', result);
    setSegmentedAudio(result);
    
    // Log success
    console.log('Audio segmented successfully:', result);
    
  } catch (error) {
    console.error('Error segmenting audio:', error);
    setErrorMessage(`Failed to segment audio: ${error.message}`);
    setSegments([]);
    setSegmentedAudio(null);
  } finally {
    setIsProcessing(false);
  }
};

// Updated pronunciation checking with CNN support
const checkPronunciation = async () => {
  if (!audioBlob) {
    setErrorMessage('Please record your pronunciation first.');
    return;
  }

  setIsProcessing(true);
  setErrorMessage(null);

  try {
    if (usePipeline) {
      // First segment the audio if using the pipeline and not already segmented
      if (!segmentedAudio) {
        console.log('Segmenting audio first...');
        await segmentAudio();
      }

      // Use the simplified analysis approach
      console.log('Starting simplified analysis of all segments...');
      const segmentResults = await audioService.analyzeAllSegments(segments, selectedModel);
      
      console.log('Analysis results:', segmentResults);
      
      // Set the results
      setResults(segmentResults);
      
      // Calculate overall score
      const score = audioService.calculateOverallScore(segmentResults);
      setOverallScore(score);
      
      // Log analysis summary
      const summary = audioService.getAnalysisSummary(segmentResults);
      console.log('Analysis summary:', summary);
      
    } else {
      // Use the direct analysis method (updated for CNN support)
      console.log('Using direct analysis (no segmentation)...');
      
      // Get supported phonemes in this harf that work with the selected model
      const supportedPhonemes = harfInfo.phonemes.filter(p => 
        supportedPhonemeMap[p] && supportedPhonemeMap[p][selectedModel]
      );
      
      if (supportedPhonemes.length === 0) {
        setErrorMessage(`This harf does not contain any phonemes supported by the ${selectedModel.toUpperCase()} model yet.`);
        setIsProcessing(false);
        return;
      }

      // For each supported phoneme, make an API call
      const resultMap = {};
      let totalScore = 0;
      let analyzedPhonemes = 0;

      for (const phoneme of supportedPhonemes) {
        const modelId = supportedPhonemeMap[phoneme][selectedModel];
        
        if (!modelId) continue;
        
        try {
          const formData = new FormData();
          formData.append('audio', audioBlob, 'recording.wav');
          formData.append('model', selectedModel);
          formData.append('phoneme', phoneme);
          formData.append('model_id', modelId);

          const response = await fetch(`${API_BASE_URL}/analyze-tajweed`, {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`Failed to analyze phoneme ${phoneme}`);
          }

          const data = await response.json();
          
          if (data.error) {
            throw new Error(data.error);
          }
          
          resultMap[phoneme] = data;
          
          // Calculate score (0-100)
          const score = data.correct ? data.confidence : (100 - data.confidence);
          totalScore += score;
          analyzedPhonemes++;
          
        } catch (error) {
          console.error(`Error analyzing phoneme ${phoneme}:`, error);
          resultMap[phoneme] = { error: error.message };
        }
      }

      // Set the results
      setResults(resultMap);
      
      // Calculate overall score
      if (analyzedPhonemes > 0) {
        setOverallScore(Math.round(totalScore / analyzedPhonemes * 100));
      }
    }
  } catch (error) {
    console.error('Error processing audio:', error);
    setErrorMessage(error.message || 'Failed to analyze pronunciation.');
  } finally {
    setIsProcessing(false);
  }
};

// Enhanced reset function (unchanged)
const resetPractice = () => {
  setAudioBlob(null);
  setAudioUrl('');
  setSegments([]);
  setSegmentedAudio(null);
  setResults({});
  setOverallScore(null);
  setErrorMessage(null);
  console.log('Practice session reset');
};

// Updated results display section JSX (unchanged)
const renderResultsSection = () => {
  if (overallScore === null) return null;

  const summary = audioService.getAnalysisSummary(results);
  
  return (
    <div className="results-section">
      <div className="">
        <div className="score-badge">
          <Award size={24} />
          <span>{overallScore}%</span>
        </div>
        <h3>Overall Pronunciation</h3>
        <div className="score-details">
          <p>Model: {getModelInfo(selectedModel).name}</p>
          <p>Analyzed: {summary.analyzedPhonemes}/{summary.totalPhonemes} phonemes</p>
          
        </div>
      </div>

      <div className="phoneme-results">
        <h3>Detailed Results</h3>
        {Object.entries(results).map(([phoneme, result]) => {
          return (
            <div 
              key={phoneme} 
              className={`phoneme-result ${result.correct ? 'correct' : 'incorrect'}`}
            >
              <div className="phoneme-name">{phoneme}</div>
              <div className="phoneme-status">
                {result.correct ? 'Correct' : 'Needs Practice'}
              </div>
              
              <div className="phoneme-feedback">{result.recommendation}</div>
              {result.model_used && (
                <div className="phoneme-model">
                  <small>Model: {getModelInfo(result.model_used).name}</small>
                </div>
              )}
              {result.processing_time && (
                <div className="phoneme-timing">
                  <small>Processed in: {result.processing_time}s</small>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

  return (
    <div className="harf-detail-container">
      <div className="harf-detail-header">
        <button 
          className="back-button" 
          onClick={() => navigate('/harf-selection')}
        >
          <ArrowLeft size={24} />
        </button>
        <h1 className="harf-title">{harf}</h1>
      </div>

      <div className="harf-info">
        <h2>{harfInfo.name}</h2>
        <div className="phonemes-list">
          {harfInfo.phonemes.map((phoneme, index) => {
            const isSupported = supportedPhonemeMap[phoneme] && supportedPhonemeMap[phoneme][selectedModel];
            return (
              <span 
                key={index} 
                className={`phoneme-tag ${
                  results[phoneme] 
                    ? (results[phoneme].correct ? 'correct' : 'incorrect') 
                    : isSupported 
                      ? 'supported' 
                      : 'unsupported'
                }`}
                title={isSupported ? `Supported by ${getModelInfo(selectedModel).name}` : `Not supported by ${getModelInfo(selectedModel).name}`}
              >
                {phoneme}
              </span>
            );
          })}
        </div>
      </div>

      <div className="practice-section">
        <div className="reference-section">
          <button 
            className="reference-button" 
            onClick={playReferenceAudio}
          >
            <Volume2 size={24} />
            <span>Listen to Reference</span>
          </button>
        </div>

        <div className="recording-section">
          {!isRecording ? (
            <button 
              className="record-button"
              onClick={startRecording}
              disabled={isProcessing}
            >
              <MicIcon size={24} />
              <span>Start Recording</span>
            </button>
          ) : (
            <button 
              className="stop-button"
              onClick={stopRecording}
            >
              <span>Stop Recording</span>
            </button>
          )}

          {audioUrl && (
            <div className="audio-preview">
              <audio src={audioUrl} controls />
            </div>
          )}
        </div>

        <div className="pipeline-toggle">
          <label className="toggle-label">
            <input 
              type="checkbox" 
              checked={usePipeline} 
              onChange={togglePipeline}
              disabled={isProcessing}
            />
            <span>Use segmentation pipeline</span>
          </label>
          <div className="pipeline-info">
            <Layers size={16} />
            <span>
              {usePipeline 
                ? "Audio will be segmented into individual phonemes before analysis"
                : "Audio will be analyzed as a whole (less accurate)"}
            </span>
          </div>
        </div>

        {/* Enhanced Model Selection UI */}
        <div className="model-selection" role="radiogroup" aria-label="Model Selection">
          <label className="model-label">Select AI Model:</label>
          <div className="model-options">
            {availableModels.map((model) => {
              const modelInfo = getModelInfo(model);
              return (
                <label className="model-option" key={model}>
                  <input
                    type="radio"
                    name="model"
                    value={model}
                    checked={selectedModel === model}
                    onChange={() => setSelectedModel(model)}
                    disabled={isProcessing}
                    aria-checked={selectedModel === model}
                  />
                  <div className="model-details">
                    <span className="model-name">{modelInfo.name}</span>
                    <span className="model-description">{modelInfo.description}</span>
                    <div className="model-specs">
                      <span className="model-speed">Speed: {modelInfo.speed}</span>
                      <span className="model-accuracy">Accuracy: {modelInfo.accuracy}</span>
                    </div>
                  </div>
                </label>
              );
            })}
          </div>
          
          {availableModels.length === 0 && (
            <div className="no-models-warning">
              <p>No AI models available for this letter's phonemes yet.</p>
            </div>
          )}
        </div>

        <div className="action-buttons">
          {usePipeline && !segmentedAudio && audioBlob && (
            <button 
              className="segment-button"
              onClick={segmentAudio}
              disabled={!audioBlob || isProcessing || availableModels.length === 0}
            >
              <Layers size={18} />
              <span>Segment Audio</span>
            </button>
          )}

          <button 
            className="analyze-button"
            onClick={checkPronunciation}
            disabled={!audioBlob || isProcessing || availableModels.length === 0}
          >
            {isProcessing ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              `Check with ${getModelInfo(selectedModel).name}${usePipeline && !segmentedAudio ? ' (Segment & Analyze)' : ''}`
            )}
          </button>

          <button 
            className="reset-button"
            onClick={resetPractice}
            disabled={isProcessing}
          >
            <RotateCcw size={18} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {errorMessage && (
        <div className="error-message">
          {errorMessage}
        </div>
      )}

      {segments.length > 0 && (
        <div className="segments-section">
          <h3>Segmented Phonemes</h3>
          <div className="segments-grid">
            {segments.map((segment, index) => (
              <div key={index} className="segment-item">
                <div className="segment-phoneme">{segment.phoneme}</div>
                <audio src={`${API_BASE_URL}${segment.url}`} controls />
              </div>
            ))}
          </div>
        </div>
      )}
      
      <div>{renderResultsSection()}</div>
    </div>
  );
};

export default HarfDetail;