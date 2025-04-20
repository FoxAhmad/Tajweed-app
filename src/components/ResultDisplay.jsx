import React from 'react';
import './ResultDisplay.css'; // Make sure to create this CSS file

export function ResultDisplay({ result }) {
  // Check if result contains an error
  if (result.error) {
    return (
      <div className="result-display error">
        <h2>Analysis Error</h2>
        <p className="error-message">{result.error}</p>
        <p className="error-help">Please try recording again or select a different audio file.</p>
      </div>
    );
  }

  // Calculate color based on confidence
  const getAccuracyColor = () => {
    const confidence = result.confidence || 0;
    if (confidence >= 80) return '#4caf50'; // Green for high confidence
    if (confidence >= 60) return '#ffb74d'; // Orange for medium confidence
    return '#f44336'; // Red for low confidence
  };

  // Format processing time
  const formatTime = (time) => {
    if (!time) return '';
    return `(processed in ${time}s)`;
  };

  return (
    <div className="result-display">
      <h2 className="result-title">Pronunciation Analysis</h2>
      
      <div className="result-content">
        <div 
          className="accuracy-circle" 
          style={{ 
            background: `conic-gradient(
              ${getAccuracyColor()} ${(result.confidence || 0)}%, 
              #e0e0e0 ${(result.confidence || 0)}% 100%
            )`
          }}
        >
          <div className="inner-circle">
            <span className="percentage-value">{Math.round(result.confidence || 0)}</span>
            <span className="percentage-symbol">%</span>
            <div className="accuracy-label">Overall Accuracy</div>
          </div>
        </div>
        
        <div className="result-details">
          <div className="result-section">
            <h3>Summary {formatTime(result.processing_time)}</h3>
            <div className={`pronunciation-status ${result.correct ? 'correct' : 'incorrect'}`}>
              <span className="status-icon">
                {result.correct ? '✓' : '✗'}
              </span>
              <span className="status-text">
                {result.correct 
                  ? "Your pronunciation is correct!" 
                  : "Your pronunciation needs improvement."}
              </span>
            </div>
            
            {result.phoneme && (
              <div className="phoneme-info">
                <span className="phoneme-label">Phoneme:</span> 
                <span className="phoneme-value">{result.phoneme}</span>
              </div>
            )}
          </div>
          
          <div className="result-section">
            <h3>Detailed Feedback</h3>
            <p className="feedback-text">{result.recommendation || "No detailed feedback available."}</p>
          </div>
          
          <div className="result-section tips">
            <h3>Tips for Improvement</h3>
            <ul className="tips-list">
              {result.correct ? (
                <>
                  <li>Keep practicing to maintain your excellent pronunciation</li>
                  <li>Try more complex phrases containing this sound</li>
                </>
              ) : (
                <>
                  <li>Listen carefully to reference audio examples</li>
                  <li>Focus on the proper mouth and tongue position</li>
                  <li>Practice with slower, deliberate pronunciations first</li>
                </>
              )}
            </ul>
          </div>
        </div>
      </div>
      
      <div className="result-actions">
        <button className="action-button try-again">
          Try Again
        </button>
        <button className="action-button share-result">
          Share Result
        </button>
      </div>
    </div>
  );
}