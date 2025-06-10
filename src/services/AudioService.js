// File: src/services/AudioService.js (Updated)

/**
 * Service for handling audio recording, segmentation and analysis
 */
class AudioService {
  constructor(apiBaseUrl = 'http://localhost:5000') {
    this.apiBaseUrl = apiBaseUrl;
  }

  /**
   * Saves an audio file directly to the server's input folder
   * @param {Blob} audioBlob - The audio blob to save
   * @param {string} letter - The letter name (e.g., 'alif', 'ba')
   * @returns {Promise<Object>} - The save result with filename
   */
  async saveAudioToServer(audioBlob, letter) {
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      formData.append('letter', letter);

      const response = await fetch(`${this.apiBaseUrl}/save-audio`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        
        try {
          // Try to parse as JSON
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.error || 'Failed to save audio';
        } catch {
          // If not JSON, use the raw text
          errorMessage = errorText || 'Failed to save audio';
        }
        
        throw new Error(errorMessage);
      }

      return await response.json();
    } catch (error) {
      console.error('Error saving audio:', error);
      throw error;
    }
  }

  /**
   * Segments an audio file into phonemes using the backend pipeline
   * @param {Blob} audioBlob - The audio blob to segment
   * @param {string} letter - The letter name (e.g., 'alif', 'ba')
   * @param {string} model - The model to use (whisper or wave2vec)
   * @returns {Promise<Object>} - The segmentation results
   */
  async segmentAudio(audioBlob, letter, model = 'whisper') {
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      formData.append('letter', letter);
      formData.append('model', model);  // Include the model parameter

      const response = await fetch(`${this.apiBaseUrl}/segment-audio`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        
        try {
          // Try to parse as JSON
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.error || 'Failed to segment audio';
        } catch {
          // If not JSON, use the raw text
          errorMessage = errorText || 'Failed to segment audio';
        }
        
        throw new Error(errorMessage);
      }

      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Segmentation failed');
      }
      
      return data;
    } catch (error) {
      console.error('Error segmenting audio:', error);
      throw error;
    }
  }

  /**
   * Analyzes a segmented phoneme
   * @param {string} segmentId - The ID of the segment to analyze
   * @param {string} phoneme - The phoneme to analyze
   * @param {string} model - The model to use (whisper or wave2vec)
   * @returns {Promise<Object>} - The analysis results
   */
  async analyzeSegment(segmentId, phoneme, model = 'whisper') {
    try {
      const formData = new FormData();
      formData.append('segment_id', segmentId);
      formData.append('phoneme', phoneme);
      formData.append('model', model);

      const response = await fetch(`${this.apiBaseUrl}/analyze-segment`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        
        try {
          // Try to parse as JSON
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.error || 'Failed to analyze segment';
        } catch {
          // If not JSON, use the raw text
          errorMessage = errorText || 'Failed to analyze segment';
        }
        
        throw new Error(errorMessage);
      }

      return await response.json();
    } catch (error) {
      console.error('Error analyzing segment:', error);
      throw error;
    }
  }

  /**
   * Analyze all phonemes in a segmented recording
   * @param {Array} segments - Array of segment objects from segmentAudio
   * @param {string} model - The model to use (whisper or wave2vec)
   * @returns {Promise<Object>} - Results for each phoneme
   */
  async analyzeAllSegments(segments, model = 'whisper') {
    try {
      const results = {};
      
      // Analyze each segment sequentially
      for (const segment of segments) {
        try {
          const result = await this.analyzeSegment(
            segment.segment_id,
            segment.phoneme,
            model
          );
          
          results[segment.phoneme] = result;
        } catch (error) {
          console.error(`Error analyzing segment ${segment.phoneme}:`, error);
          results[segment.phoneme] = { error: error.message };
        }
      }
      
      return results;
    } catch (error) {
      console.error('Error analyzing segments:', error);
      throw error;
    }
  }

  /**
   * Calculate overall score from individual phoneme results
   * @param {Object} results - Results object from analyzeAllSegments
   * @returns {number} - Overall score (0-100)
   */
  calculateOverallScore(results) {
    const phonemes = Object.keys(results);
    
    if (phonemes.length === 0) {
      return null;
    }
    
    let totalScore = 0;
    let analyzedPhonemes = 0;
    
    for (const phoneme of phonemes) {
      const result = results[phoneme];
      
      if (result && !result.error) {
        // Calculate score (0-100)
        const score = result.correct ? result.confidence : (100 - result.confidence);
        totalScore += score;
        analyzedPhonemes++;
      }
    }
    
    // Calculate overall score
    if (analyzedPhonemes > 0) {
      return Math.round(totalScore / analyzedPhonemes);
    }
    
    return null;
  }
}

export default AudioService;