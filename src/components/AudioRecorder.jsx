import React, { useState, useRef, useEffect } from 'react';


export function AudioRecorder({ onAudioCaptured }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioURL, setAudioURL] = useState('');
  const [visualizerBars, setVisualizerBars] = useState([]);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const dataArrayRef = useRef(null);
  const visualizerInterval = useRef(null);
  
  // Initialize visualizer bars
  useEffect(() => {
    const barCount = 30;
    const initialBars = Array(barCount).fill(0);
    setVisualizerBars(initialBars);
  }, []);
  
  const startRecording = async () => {
    try {
      // Reset any previous recording state
      audioChunksRef.current = [];
      setAudioURL('');
      setRecordingTime(0);
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Set up audio context for visualization
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioContext.createAnalyser();
      const source = audioContext.createMediaStreamSource(stream);
      
      analyser.fftSize = 128;
      source.connect(analyser);
      
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      dataArrayRef.current = dataArray;
      
      // Create media recorder with proper MIME type that works well on both browsers
      const options = { mimeType: 'audio/webm' }; // This works better across browsers
      const mediaRecorder = new MediaRecorder(stream, options);
      
      // Set up event handlers
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        // Clean up visualization
        if (visualizerInterval.current) {
          clearInterval(visualizerInterval.current);
          visualizerInterval.current = null;
        }
        
        // Create a blob from the audio data
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        
        // Create a URL for the blob
        const url = URL.createObjectURL(audioBlob);
        setAudioURL(url);
        
        // Notify parent component
        onAudioCaptured(audioBlob, url);
        
        // Clean up
        clearInterval(timerRef.current);
        setRecordingTime(0);
        
        // Reset visualizer
        setVisualizerBars(visualizerBars.map(() => 0));
      };
      
      // Store media recorder
      mediaRecorderRef.current = mediaRecorder;
      
      // Start recording
      mediaRecorder.start(10); // Collect data in 10ms chunks
      setIsRecording(true);
      
      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime((prevTime) => prevTime + 1);
      }, 1000);
      
      // Start visualizer
      visualizerInterval.current = setInterval(() => {
        updateVisualizer();
      }, 100);
      
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Failed to access microphone. Please make sure it is connected and permissions are granted.');
    }
  };
  
  const updateVisualizer = () => {
    if (!analyserRef.current || !dataArrayRef.current) return;
    
    analyserRef.current.getByteFrequencyData(dataArrayRef.current);
    
    // Create new bar heights based on frequency data
    const bars = visualizerBars.map((_, index) => {
      // Map the data array to our bar count
      const dataIndex = Math.floor(index * dataArrayRef.current.length / visualizerBars.length);
      // Scale the value to the height of our visualizer
      return (dataArrayRef.current[dataIndex] / 255) * 100;
    });
    
    setVisualizerBars(bars);
  };
  
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      // Stop all tracks of the stream
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      
      // Close audio context
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    }
  };
  
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
    const secs = (seconds % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
  };
  
  return (
    <div className="audio-recorder">
      <div className="controls">
        {!isRecording ? (
          <button
            className="record-btn"
            onClick={startRecording}
            disabled={isRecording}
          >
            Start Recording
          </button>
        ) : (
          <button
            className="stop-btn"
            onClick={stopRecording}
            disabled={!isRecording}
          >
            Stop Recording <span className="recording-timer">{formatTime(recordingTime)}</span>
          </button>
        )}
      </div>
      
      {isRecording && (
        <>
          <div className="visualizer">
            <div className="visualizer-bars">
              {visualizerBars.map((height, index) => (
                <div 
                  key={index}
                  className="visualizer-bar"
                  style={{ height: `${height}%` }}
                />
              ))}
            </div>
          </div>
          
          <div className="recording-indicator">
            <div className="recording-dot"></div>
            <span className="recording-text">Recording in progress...</span>
          </div>
        </>
      )}
      
      {audioURL && (
        <div className="audio-preview">
          <audio src={audioURL} controls />
        </div>
      )}
    </div>
  );
}