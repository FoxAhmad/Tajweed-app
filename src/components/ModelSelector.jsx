// File: src/components/ModelSelector.js
import React, { useEffect } from 'react';

// Updated phoneme mapping to include CNN models
const PHONEME_MODEL_MAPPING = {
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

// All supported phonemes
const PHONEMES = Object.keys(PHONEME_MODEL_MAPPING);

export function ModelSelector({ 
  selectedModel, 
  onModelChange, 
  selectedPhoneme, 
  onPhonemeChange 
}) {
  // Update model when phoneme changes to make sure we're using compatible models
  useEffect(() => {
    if (selectedPhoneme && !PHONEME_MODEL_MAPPING[selectedPhoneme][selectedModel]) {
      // If current model doesn't support the selected phoneme, switch to a model that does
      const availableModels = Object.keys(PHONEME_MODEL_MAPPING[selectedPhoneme]);
      if (availableModels.length > 0) {
        onModelChange(availableModels[0]);
      }
    }
  }, [selectedPhoneme, selectedModel, onModelChange]);

  const getModelDescription = (modelType) => {
    switch (modelType) {
      case 'whisper':
        return 'Better for full sentence pronunciation and contextual analysis';
      case 'wave2vec':
        return 'Better for individual phoneme precision and specific tajweed rules';
      case 'cnn':
        return 'Fast spectrogram-based analysis, optimized for individual phoneme classification';
      default:
        return 'Advanced model for pronunciation analysis';
    }
  };

  const getModelDisplayName = (modelType) => {
    switch (modelType) {
      case 'whisper':
        return 'Whisper Model';
      case 'wave2vec':
        return 'Wave2Vec Model';
      case 'cnn':
        return 'CNN Model';
      default:
        return modelType;
    }
  };

  return (
    <div className="model-selector">
      <div className="phoneme-selector">
        <h3>Select Phoneme</h3>
        <p>Choose the Arabic phoneme you are practicing</p>
        <select 
          value={selectedPhoneme} 
          onChange={(e) => onPhonemeChange(e.target.value)}
          className="phoneme-dropdown"
        >
          <option value="">-- Select a phoneme --</option>
          {PHONEMES.map(phoneme => (
            <option key={phoneme} value={phoneme}>{phoneme}</option>
          ))}
        </select>
      </div>

      {selectedPhoneme && (
        <div>

        <div className="">
         
          
          {/* Get available models for the selected phoneme */}
          {Object.keys(PHONEME_MODEL_MAPPING[selectedPhoneme]).map((modelType) => (
            <div key={modelType} className="model-option">
              <input
                type="radio"
                id={modelType}
                name="model"
                value={modelType}
                checked={selectedModel === modelType}
                onChange={() => onModelChange(modelType)}
              />
              <label htmlFor={modelType}>
                <div className="model-details">
                  <h4>{getModelDisplayName(modelType)}</h4>
                  <p>{getModelDescription(modelType)}</p>
                  <small className="model-id">
                    Using: {PHONEME_MODEL_MAPPING[selectedPhoneme][modelType]}
                  </small>
                </div>
              </label>
            </div>
          ))}
          
          <div className="model-comparison">
            <h4>Model Comparison</h4>
            <div className="comparison-grid">
              <div className="comparison-item">
                <strong>Speed:</strong> CNN > Wave2Vec > Whisper
              </div>
              <div className="comparison-item">
                <strong>Accuracy:</strong> Whisper ≥ Wave2Vec ≥ CNN
              </div>
              <div className="comparison-item">
                <strong>Best for beginners:</strong> CNN (instant feedback)
              </div>
              <div className="comparison-item">
                <strong>Best for advanced:</strong> Whisper (detailed analysis)
              </div>
            </div>
          </div>
        </div>
        </div>
      )}
    </div>
  );
}