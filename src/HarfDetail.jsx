// File: src/HarfDetail.jsx
import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Play, MicIcon, ArrowLeft, Volume2, Award, RotateCcw, Layers } from "lucide-react";
import "./HarfDetail.css";
import AudioService from "./services/AudioService";

// API Configuration
const API_BASE_URL = 'http://localhost:5000';
const audioService = new AudioService(API_BASE_URL);

// The phonetic mapping of Arabic letters
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
  'س': { name: 'seen', phonemes: ['s', 'ee', 'n'] },
  'ش': { name: 'sheen', phonemes: ['sh', 'ee', 'n'] },
  'ص': { name: 'saad', phonemes: ['s', 'aa', 'd'] },
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

// Map to our supported phonemes in the backend
const supportedPhonemeMap = {
  'ee': 'ee',
  'so': 'so',
  'si': 'si',
  // Add more mappings as they become available in the backend
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
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Start recording function
  const startRecording = async () => {
    try {
      setErrorMessage(null);
      audioChunksRef.current = [];
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorderRef.current.onstop = async () => {
        try {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
          const audioUrl = URL.createObjectURL(audioBlob);
          
          // Save to local state for preview
          setAudioBlob(audioBlob);
          setAudioUrl(audioUrl);
          
          // Reset segments and results when recording new audio
          setSegments([]);
          setSegmentedAudio(null);
          setResults({});
          setOverallScore(null);
          
          // Save directly to server input folder
          try {
            const saveResult = await audioService.saveAudioToServer(audioBlob, harfInfo.name);
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

  // Stop recording function
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      // Stop all audio tracks
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  // Play reference audio
  const playReferenceAudio = () => {
    const audio = new Audio(`/sounds/arabic_letter_voice_${parseInt(harfId) + 1}.wav`);
    audio.play();
  };

  // Reset the practice
  const resetPractice = () => {
    setAudioBlob(null);
    setAudioUrl('');
    setSegments([]);
    setSegmentedAudio(null);
    setResults({});
    setOverallScore(null);
    setErrorMessage(null);
  };

  // Toggle pipeline usage
  const togglePipeline = () => {
    setUsePipeline(!usePipeline);
  };

  // Segment the audio using Docker pipeline
  const segmentAudio = async () => {
    if (!audioBlob) {
      setErrorMessage('Please record your pronunciation first.');
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);

    try {
      // Run the segmentation pipeline
      const result = await audioService.segmentAudio(audioBlob, harfInfo.name);
      
      // Store the segments
      setSegments(result.segments);
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

  // Check pronunciation
  const checkPronunciation = async () => {
    if (!audioBlob) {
      setErrorMessage('Please record your pronunciation first.');
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);

    try {
      if (usePipeline && !segmentedAudio) {
        // First segment the audio if using the pipeline
        await segmentAudio();
      }

      if (usePipeline && segments.length > 0) {
        // Analyze each segment using the API
        const segmentResults = await audioService.analyzeAllSegments(segments, 'whisper');
        
        // Set the results
        setResults(segmentResults);
        
        // Calculate overall score
        const score = audioService.calculateOverallScore(segmentResults);
        setOverallScore(score);
      } else {
        // Use the old direct analysis method if not using pipeline or segmentation failed
        // Get supported phonemes in this harf
        const supportedPhonemes = harfInfo.phonemes.filter(p => supportedPhonemeMap[p]);
        
        if (supportedPhonemes.length === 0) {
          setErrorMessage('This harf does not contain any supported phonemes for analysis yet.');
          setIsProcessing(false);
          return;
        }

        // For each supported phoneme, make an API call
        const resultMap = {};
        let totalScore = 0;
        let analyzedPhonemes = 0;

        for (const phoneme of supportedPhonemes) {
          const mappedPhoneme = supportedPhonemeMap[phoneme];
          
          if (!mappedPhoneme) continue;
          
          try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');
            formData.append('model', 'whisper'); // Default to whisper
            formData.append('phoneme', mappedPhoneme);

            const response = await fetch(`${API_BASE_URL}/analyze-tajweed`, {
              method: 'POST',
              body: formData,
            });

            if (!response.ok) {
              throw new Error(`Failed to analyze phoneme ${mappedPhoneme}`);
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
          setOverallScore(Math.round(totalScore / analyzedPhonemes));
        }
      }
    } catch (error) {
      console.error('Error processing audio:', error);
      setErrorMessage(error.message || 'Failed to analyze pronunciation.');
    } finally {
      setIsProcessing(false);
    }
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
          {harfInfo.phonemes.map((phoneme, index) => (
            <span 
              key={index} 
              className={`phoneme-tag ${results[phoneme] ? (results[phoneme].correct ? 'correct' : 'incorrect') : ''}`}
            >
              {phoneme}
            </span>
          ))}
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

        <div className="action-buttons">
          {usePipeline && !segmentedAudio && audioBlob && (
            <button 
              className="segment-button"
              onClick={segmentAudio}
              disabled={!audioBlob || isProcessing}
            >
              <Layers size={18} />
              <span>Segment Audio</span>
            </button>
          )}

          <button 
            className="analyze-button"
            onClick={checkPronunciation}
            disabled={!audioBlob || isProcessing}
          >
            {isProcessing ? 'Analyzing...' : (usePipeline && !segmentedAudio ? 'Segment & Analyze' : 'Check Pronunciation')}
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

      {overallScore !== null && (
        <div className="results-section">
          <div className="overall-score">
            <div className="score-badge">
              <Award size={24} />
              <span>{overallScore}%</span>
            </div>
            <h3>Overall Pronunciation</h3>
          </div>

          <div className="phoneme-results">
            <h3>Detailed Results</h3>
            {Object.entries(results).map(([phoneme, result]) => (
              <div 
                key={phoneme} 
                className={`phoneme-result ${result.correct ? 'correct' : 'incorrect'}`}
              >
                <div className="phoneme-name">{phoneme}</div>
                <div className="phoneme-status">{result.correct ? 'Correct' : 'Needs Practice'}</div>
                <div className="phoneme-confidence">{result.confidence}% confidence</div>
                <div className="phoneme-feedback">{result.recommendation}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default HarfDetail;