
// File: src/components/ModelSelector.js
import React from 'react';


const PHONEMES = [
  'ee' , 'so' , 'si'
];

export function ModelSelector({ 
  selectedModel, 
  onModelChange, 
  selectedPhoneme, 
  onPhonemeChange 
}) {
  return (
    
    <div className="model-selector">
      <div className="model-option">
        <input
          type="radio"
          id="whisper"
          name="model"
          value="whisper"
          checked={selectedModel === 'whisper'}
          onChange={() => onModelChange('whisper')}
        />
        <label htmlFor="whisper">
          <div className="model-details">
            <h3>Whisper Model</h3>
            <p>Better for full sentence pronunciation and contextual analysis</p>
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
        />
        <label htmlFor="wave2vec">
          <div className="model-details">
            <h3>Wave2Vec Model</h3>
            <p>Better for individual phoneme precision and specific tajweed rules</p>
          </div>
        </label>
      </div>
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
    </div>
  );
}