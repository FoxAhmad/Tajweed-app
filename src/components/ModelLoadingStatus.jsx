// File: src/components/ModelLoadingStatus.js
import React, { useState, useEffect } from 'react';
import './ModelLoadingStatus.css'; // You'll need to create this CSS file

export function ModelLoadingStatus() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Function to fetch status
    const fetchStatus = async () => {
      try {
        const response = await fetch('http://localhost:5000/status');
        if (!response.ok) {
          throw new Error(`Server returned ${response.status}`);
        }
        const data = await response.json();
        setStatus(data);
        setError(null);
        
        // Continue polling if models are still loading
        return data.status === 'loading';
      } catch (err) {
        console.error('Error fetching model status:', err);
        setError(err.message);
        return true; // Continue polling on error
      } finally {
        setLoading(false);
      }
    };
    
    // Initial fetch
    fetchStatus();
    
    // Set up polling
    const intervalId = setInterval(async () => {
      const shouldContinue = await fetchStatus();
      if (!shouldContinue) {
        clearInterval(intervalId);
      }
    }, 5000); // Poll every 5 seconds
    
    // Clean up interval on unmount
    return () => clearInterval(intervalId);
  }, []);
  
  if (loading) {
    return (
      <div className="model-loading-status loading">
        <h3>Checking model status...</h3>
        <div className="loading-spinner"></div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="model-loading-status error">
        <h3>Error checking model status</h3>
        <p>{error}</p>
      </div>
    );
  }
  
  if (!status) {
    return null;
  }
  
  return (
    <div className="model-loading-status">
      <h3>Model Loading Status</h3>
      
      <div className="loading-progress">
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${status.loading_progress.percentage}%` }}
          ></div>
        </div>
        <p>
          {status.loading_progress.loaded_models} of {status.loading_progress.total_models} models loaded 
          ({status.loading_progress.percentage}%)
        </p>
      </div>
      
      {status.status === 'loading' && (
        <p className="status-message">
          Server is still loading models. Some features may be unavailable until loading completes.
        </p>
      )}
      
      {status.status === 'ready' && (
        <p className="status-message success">
          All models are loaded and ready to use!
        </p>
      )}
      
      <div className="model-details">
        <h4>Model Details</h4>
        <table>
          <thead>
            <tr>
              <th>Phoneme</th>
              <th>Model Type</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(status.model_status).map(([key, model]) => (
              <tr key={key} className={model.loaded ? 'loaded' : 'not-loaded'}>
                <td>{model.phoneme}</td>
                <td>{model.model_type}</td>
                <td>
                  {model.loaded ? (
                    <span className="status-loaded">✓ Loaded</span>
                  ) : (
                    <span className="status-not-loaded">⏳ Loading...</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}