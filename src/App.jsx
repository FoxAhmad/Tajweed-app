// File: src/App.js
import React, { useState } from 'react';
import './App.css';
import { AudioRecorder } from './components/AudioRecorder';
import { AudioUploader } from './components/AudioUploader';
import { ResultDisplay } from './components/ResultDisplay';
import { ModelSelector } from './components/ModelSelector';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';

function App() {
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState('');
  const [selectedModel, setSelectedModel] = useState('whisper'); // 'whisper' or 'wave2vec'
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  // In App.js, add a new state
const [selectedPhoneme, setSelectedPhoneme] = useState('');

  
  const handleAudioUpload = (blob, url) => {
    setAudioBlob(blob);
    setAudioUrl(url);
    setResult(null); // Clear previous result
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
        const errorMsg = await response.text(); // helpful for debugging
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
    <div className="app">
      <Navbar />
      
      <main className="app-content">
        <h1>Tajweed Pronunciation Checker</h1>
        <p className="app-description">
          Upload or record audio of a Quranic phoneme and get instant feedback on your pronunciation.
        </p>
        
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
            onModelChange={handleModelChange }
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
      
      <Footer />
    </div>
  );
}

export default App;




