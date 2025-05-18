// File: src/components/ModelSelector.js
import React, { useEffect } from 'react';

// Phoneme mapping to model IDs
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
        <div className="model-options">
          <h3>Select Model Type</h3>
          <div className="model-option">
            <input
              type="radio"
              id="whisper"
              name="model"
              value="whisper"
              checked={selectedModel === 'whisper'}
              onChange={() => onModelChange('whisper')}
              disabled={!PHONEME_MODEL_MAPPING[selectedPhoneme]?.whisper}
            />
            <label htmlFor="whisper" className={!PHONEME_MODEL_MAPPING[selectedPhoneme]?.whisper ? 'disabled' : ''}>
              <div className="model-details">
                <h3>Whisper Model</h3>
                <p>Better for full sentence pronunciation and contextual analysis</p>
                {selectedPhoneme && PHONEME_MODEL_MAPPING[selectedPhoneme]?.whisper && (
                  <small>Using model: {PHONEME_MODEL_MAPPING[selectedPhoneme].whisper}</small>
                )}
              </div>
            </label>
          </div>
          
          <div className="model-option">
            <input
              type="radio"
              id="wave2vec"
              name="model"
              value="wave2vec"
              checked={selectedModel === 'wave2vec'}
              onChange={() => onModelChange('wave2vec')}
              disabled={!PHONEME_MODEL_MAPPING[selectedPhoneme]?.wave2vec}
            />
            <label htmlFor="wave2vec" className={!PHONEME_MODEL_MAPPING[selectedPhoneme]?.wave2vec ? 'disabled' : ''}>
              <div className="model-details">
                <h3>Wave2Vec Model</h3>
                <p>Better for individual phoneme precision and specific tajweed rules</p>
                {selectedPhoneme && PHONEME_MODEL_MAPPING[selectedPhoneme]?.wave2vec && (
                  <small>Using model: {PHONEME_MODEL_MAPPING[selectedPhoneme].wave2vec}</small>
                )}
              </div>
            </label>
          </div>
        </div>
      )}
    </div>
  );
}