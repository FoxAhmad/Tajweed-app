// File: src/App.jsx (Updated)
import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import { AudioRecorder } from './components/AudioRecorder';
import { AudioUploader } from './components/AudioUploader';
import { ResultDisplay } from './components/ResultDisplay';
import { ModelSelector } from './components/ModelSelector';
import { ModelLoadingStatus } from './components/ModelLoadingStatus';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';

// Import screens
import HarfSelection from './HarfSelection';
import HarfDetail from './HarfDetail';

// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// Phoneme-to-model mapping
const PHONEME_MODEL_MAPPING = {
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
};

// Available phonemes for selection
const AVAILABLE_PHONEMES = Object.keys(PHONEME_MODEL_MAPPING);

function HomePage({
  handleAudioUpload,
  handleModelChange,
  handlePhonemeChange,
  processAudio,
  audioBlob,
  audioUrl,
  selectedModel,
  selectedPhoneme,
  isProcessing,
  result,
  serverStatus,
  loadingPercentage,
  errorMessage,
}) {
  const navigate = useNavigate();

  return (
    <main className="app-content">
      {/* Server status indicator */}
      {serverStatus === 'loading' && (
        <ModelLoadingStatus percentage={loadingPercentage} />
      )}
      
      {/* Error message display */}
      {errorMessage && (
        <div className="error-message">
          {errorMessage}
        </div>
      )}
      
      <div className="dashboard-cards">
        <div className="dashboard-card" onClick={() => navigate('/harf-selection')}>
          <h2>Qaida Practice</h2>
          <p>Practice individual Arabic letters with pronunciation feedback</p>
        </div>
        
        <div className="dashboard-card">
          <h2>Recitation Practice</h2>
          <p>Practice reciting Quranic verses with Tajweed rules</p>
          <span className="coming-soon-badge">Coming Soon</span>
        </div>
        
        <div className="dashboard-card">
          <h2>Custom Analysis</h2>
          <p>Analyze your own audio recordings for specific phonemes</p>
          <button 
            className="card-button"
            onClick={() => document.getElementById('custom-analysis').scrollIntoView({ behavior: 'smooth' })}
          >
            Try Now
          </button>
        </div>
      </div>

      <div id="custom-analysis" className="custom-analysis-section">
        <h2>Custom Pronunciation Analysis</h2>
        <p className="section-description">
          Upload or record audio to analyze specific phoneme pronunciation
        </p>

        <div className="audio-input-section">
          <div className="input-methods">
            <div className="input-method">
              <h3>Record Audio</h3>
              <AudioRecorder onAudioCaptured={handleAudioUpload} />
            </div>

            <div className="input-method">
              <h3>Upload Audio</h3>
              <AudioUploader onAudioUploaded={handleAudioUpload} />
            </div>
          </div>

          {audioUrl && (
            <div className="audio-preview">
              <h3>Preview</h3>
              <audio src={audioUrl} controls />
            </div>
          )}
        </div>

        <div className="model-section">
          <h3>Analysis Configuration</h3>
          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
            selectedPhoneme={selectedPhoneme}
            onPhonemeChange={handlePhonemeChange}
            availablePhonemes={AVAILABLE_PHONEMES}
          />
        </div>
          
        <button
          className={`process-btn ${(!audioBlob || !selectedPhoneme || serverStatus !== 'ready') ? 'disabled' : ''}`}
          disabled={!audioBlob || !selectedPhoneme || isProcessing || serverStatus !== 'ready'}
          onClick={processAudio}
        >
          {isProcessing ? 'Analyzing...' : 'Analyze Pronunciation'}
        </button>

        {result && <ResultDisplay result={result} />}
      </div>
    </main>
  );
}

function App() {
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState('');
  const [selectedModel, setSelectedModel] = useState('whisper');
  const [selectedPhoneme, setSelectedPhoneme] = useState('');
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelId, setModelId] = useState('');
  const [serverStatus, setServerStatus] = useState('loading');
  const [loadingPercentage, setLoadingPercentage] = useState(0);
  const [errorMessage, setErrorMessage] = useState(null);

  // Check if the server is ready
  useEffect(() => {
    let statusCheckInterval;
    
    const checkServerStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/status`);
        
        if (response.ok) {
          const data = await response.json();
          setServerStatus(data.status);
          
          if (data.loading_progress) {
            setLoadingPercentage(data.loading_progress.percentage || 0);
          }
          
          // Clear error message if server is responding
          setErrorMessage(null);
          
          // If server is ready, stop checking
          if (data.status === 'ready') {
            clearInterval(statusCheckInterval);
          }
        } else {
          setServerStatus('error');
          setErrorMessage('Failed to connect to the server');
        }
      } catch (error) {
        console.error('Error checking server status:', error);
        setServerStatus('error');
        setErrorMessage('Could not reach the server. Make sure the backend is running.');
      }
    };
    
    // Initial check
    checkServerStatus();
    
    // Set up interval for continuous checking if not ready
    statusCheckInterval = setInterval(checkServerStatus, 5000);
    
    // Clean up interval on component unmount
    return () => {
      clearInterval(statusCheckInterval);
    };
  }, []);

  // Update modelId when model or phoneme changes
  useEffect(() => {
    if (selectedPhoneme && selectedModel) {
      const newModelId = PHONEME_MODEL_MAPPING[selectedPhoneme]?.[selectedModel] || '';
      setModelId(newModelId);
    } else {
      setModelId('');
    }
  }, [selectedModel, selectedPhoneme]);

  const handleAudioUpload = useCallback((blob, url) => {
    setAudioBlob(blob);
    setAudioUrl(url);
    setResult(null);
    setErrorMessage(null); // Clear any previous errors
  }, []);

  const handleModelChange = useCallback((model) => {
    setSelectedModel(model);
  }, []);

  const handlePhonemeChange = useCallback((phoneme) => {
    setSelectedPhoneme(phoneme);
  }, []);

  const processAudio = async () => {
    if (!audioBlob || !selectedPhoneme ) {
      setErrorMessage('Please ensure audio is uploaded, phoneme is selected, and server is ready.');
      return;
    }
    
    setIsProcessing(true);
    setErrorMessage(null);

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      formData.append('model', selectedModel);
      formData.append('phoneme', selectedPhoneme);
      
      if (modelId) {
        formData.append('model_id', modelId);
      }

      const response = await fetch(`${API_BASE_URL}/analyze-tajweed`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        
        try {
          // Try to parse as JSON
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.error || 'Failed to process audio';
        } catch {
          // If not JSON, use the raw text
          errorMessage = errorText || 'Failed to process audio';
        }
        
        throw new Error(errorMessage);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      setResult(data);
      setErrorMessage(null); // Clear any errors on success
    } catch (error) {
      console.error('Error processing audio:', error);
      const errorMsg = error.message || 'Failed to process audio. Please try again.';
      setErrorMessage(errorMsg);
      setResult({ error: errorMsg });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <Router>
      <div className="app">
        <Navbar />
        <Routes>
          <Route
            path="/"
            element={
              <HomePage
                handleAudioUpload={handleAudioUpload}
                handleModelChange={handleModelChange}
                handlePhonemeChange={handlePhonemeChange}
                processAudio={processAudio}
                audioBlob={audioBlob}
                audioUrl={audioUrl}
                selectedModel={selectedModel}
                selectedPhoneme={selectedPhoneme}
                isProcessing={isProcessing}
                result={result}
                serverStatus={serverStatus}
                loadingPercentage={loadingPercentage}
                errorMessage={errorMessage}
              />
            }
          />
          <Route path="/harf-selection" element={<HarfSelection />} />
          <Route path="/harf/:harfId" element={<HarfDetail />} />
        </Routes>
        <Footer />
      </div>
    </Router>
  );
}

export default App;