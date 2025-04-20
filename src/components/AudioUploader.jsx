import React, { useState, useRef } from 'react';


export function AudioUploader({ onAudioUploaded }) {
  const [isDragging, setIsDragging] = useState(false);
  const [audioURL, setAudioURL] = useState('');
  const [fileInfo, setFileInfo] = useState(null);
  const fileInputRef = useRef(null);
  
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    processAudioFile(file);
  };
  
  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  
  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    
    const file = event.dataTransfer.files[0];
    processAudioFile(file);
  };
  
  const handleButtonClick = () => {
    fileInputRef.current.click();
  };
  
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
  
  const processAudioFile = (file) => {
    if (!file) return;
    
    // Check if file is an audio file
    if (!file.type.startsWith('audio/')) {
      alert('Please upload an audio file (WAV, MP3, etc.)');
      return;
    }
    
    // Store file info
    setFileInfo({
      name: file.name,
      size: formatFileSize(file.size),
      type: file.type
    });
    
    // Create URL for the audio file
    const url = URL.createObjectURL(file);
    setAudioURL(url);
    
    // Convert file format to WAV if needed, or use as is
    const fileToSend = new File([file], 'recording.wav', { 
      type: 'audio/wav' 
    });
    
    // Notify parent component
    onAudioUploaded(fileToSend, url);
  };
  
  return (
    <div className="audio-uploader">
      <div 
        className={`drop-area ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleButtonClick}
      >
        <input 
          type="file" 
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="audio/*" 
          style={{ display: 'none' }}
        />
        <p>Drag and drop audio file here or click to browse</p>
      </div>
      
      {audioURL && (
        <div className="audio-preview">
          <audio src={audioURL} controls />
          
          {fileInfo && (
            <div className="file-info">
              <span className="file-name">{fileInfo.name}</span>
              <span className="file-size">({fileInfo.size})</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}