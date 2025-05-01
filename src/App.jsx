// File: src/App.js
import React, { useState } from 'react';
import './App.css';
import { AudioRecorder } from './components/AudioRecorder';
import { AudioUploader } from './components/AudioUploader';
import { ResultDisplay } from './components/ResultDisplay';
import { ModelSelector } from './components/ModelSelector';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';

// Import screens
import HarfSelection from './HarfSelection';
import HarfDetail from './HarfDetail';

function HomePage({
  handleAudioUpload,
  handleModelChange,
  handlePhonemeChange,
  processAudio,
  audioBlob,
  audioUrl,
  selectedModel,
  isProcessing,
  result,
}) {
  const navigate = useNavigate();

  return (
    <main className="app-content">
      {/* 👇 New Button to go to Harf Selection page */}
      <div style={{ textAlign: 'center', marginBottom: '20px' }}>
        <button className="btn" onClick={() => navigate('/harf-selection')}>
          Go to Harf Selection
        </button>
      </div>

      <div className="audio-input-section">
        <div className="input-methods">
          <div className="input-method">
            <h2>Record Audio</h2>
            <AudioRecorder onAudioCaptured={handleAudioUpload} />
          </div>

          <div className="input-method">
            <h2>Upload Audio</h2>
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
        <h2>Select Model</h2>
        <ModelSelector
          selectedModel={selectedModel}
          onModelChange={handleModelChange}
          onPhonemeChange={handlePhonemeChange}
        />
      </div>

      <button
        className={`process-btn ${!audioBlob ? 'disabled' : ''}`}
        disabled={!audioBlob || isProcessing}
        onClick={processAudio}
      >
        {isProcessing ? 'Processing...' : 'Analyze Pronunciation'}
      </button>

      {result && <ResultDisplay result={result} />}
    </main>
  );
}

function App() {
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState('');
  const [selectedModel, setSelectedModel] = useState('whisper');
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedPhoneme, setSelectedPhoneme] = useState('');

  const handleAudioUpload = (blob, url) => {
    setAudioBlob(blob);
    setAudioUrl(url);
    setResult(null);
  };

  const handleModelChange = (model) => {
    setSelectedModel(model);
  };

  const handlePhonemeChange = (phoneme) => {
    setSelectedPhoneme(phoneme);
  };

  const processAudio = async () => {
    if (!audioBlob) return;
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      formData.append('model', selectedModel);
      formData.append('phoneme', selectedPhoneme);

      const response = await fetch('http://localhost:5000/analyze-tajweed', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorMsg = await response.text();
        throw new Error(`Failed to process audio: ${errorMsg}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error processing audio:', error);
      setResult({ error: 'Failed to process audio. Please try again.' });
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
                isProcessing={isProcessing}
                result={result}
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
