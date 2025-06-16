// File: src/components/ResultDisplay.jsx (Updated)
import React from 'react';
import { Check, X, AlertCircle } from 'lucide-react';
import './ResultDisplay.css';

export const ResultDisplay = ({ result }) => {
  if (!result) return null;
  
  if (result.error) {
    return (
      <div className="result-container error">
        <AlertCircle size={32} />
        <h3>Error analyzing pronunciation</h3>
        <p>{result.error}</p>
      </div>
    );
  }

  const isCorrect = result.correct;
  const confidence = result.confidence || 0;
  const recommendation = result.recommendation || '';
  
  return (
    <div className={`result-container ${isCorrect ? 'correct' : 'incorrect'}`}>
      <div className="result-header">
        {isCorrect ? (
          <div className="result-icon correct">
            <Check size={32} />
          </div>
        ) : (
          <div className="result-icon incorrect">
            <X size={32} />
          </div>
        )}
        <h3>
          {isCorrect 
            ? 'Correct Pronunciation' 
            : 'Pronunciation Needs Improvement'}
        </h3>
      </div>
      
      <div className="result-details">
        
        
        <div className="recommendation">
          <h4>Feedback:</h4>
          <p>{recommendation}</p>
        </div>
        
        {result.processing_time && (
          <div className="processing-time">
            Analysis completed in {result.processing_time} seconds
          </div>
        )}
      </div>
    </div>
  );
};