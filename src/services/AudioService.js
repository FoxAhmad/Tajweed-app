/**
 * Simplified Service for handling audio recording, segmentation and analysis
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
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.error || 'Failed to save audio';
        } catch {
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
      formData.append('model', model);

      const response = await fetch(`${this.apiBaseUrl}/segment-audio`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.error || 'Failed to segment audio';
        } catch {
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
   * Analyzes a single segmented phoneme
   * @param {string} segmentId - The ID of the segment to analyze
   * @param {string} phoneme - The phoneme to analyze
   * @param {string} model - The model to use (whisper or wave2vec)
   * @returns {Promise<Object>} - The analysis results
   */
  async analyzeSegment(segmentId, phoneme, model = 'whisper') {
    try {
      console.log(`Analyzing segment: ${segmentId} with model: ${model}`);
      
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
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.error || 'Failed to analyze segment';
        } catch {
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
   * Analyze all phonemes in a segmented recording using the simplified endpoint
   * @param {Array} segments - Array of segment objects from segmentAudio (optional, not used in new approach)
   * @param {string} model - The model to use (whisper or wave2vec)
   * @returns {Promise<Object>} - Results for each phoneme
   */
  async analyzeAllSegments(segments, model = 'whisper') {
    try {
      console.log('Starting analysis of all segments with simplified approach');
      console.log('Segments provided:', segments);
      console.log('Model:', model);
      
      // Use the new simplified endpoint that automatically finds and analyzes all segments
      const formData = new FormData();
      formData.append('model', model);

      const response = await fetch(`${this.apiBaseUrl}/analyze-all-segments`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.error || 'Failed to analyze segments';
        } catch {
          errorMessage = errorText || 'Failed to analyze segments';
        }
        
        throw new Error(errorMessage);
      }

      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Analysis failed');
      }
      
      console.log('Analysis completed successfully:', data);
      
      // Return the results in the expected format
      return data.results;
      
    } catch (error) {
      console.error('Error analyzing all segments:', error);
      throw error;
    }
  }

  /**
   * Legacy method: Analyze all segments individually (kept for backward compatibility)
   * @param {Array} segments - Array of segment objects from segmentAudio
   * @param {string} model - The model to use (whisper or wave2vec)
   * @returns {Promise<Object>} - Results for each phoneme
   */
  async analyzeAllSegmentsIndividually(segments, model = 'whisper') {
    try {
      const results = {};
      console.log('Analyzing segments individually:', segments);
      
      // Analyze each segment sequentially
      for (const segment of segments) {
        try {
          const result = await this.analyzeSegment(
            segment.segment_id,
            segment.phoneme,
            model
          );
          
          results[segment.phoneme] = result;
          console.log(`✓ Analyzed segment ${segment.phoneme}:`, result);
        } catch (error) {
          console.error(`✗ Error analyzing segment ${segment.phoneme}:`, error);
          results[segment.phoneme] = { error: error.message };
        }
      }
      
      return results;
    } catch (error) {
      console.error('Error analyzing segments individually:', error);
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

  /**
   * Get detailed analysis summary
   * @param {Object} results - Results object from analyzeAllSegments
   * @returns {Object} - Analysis summary
   */
  getAnalysisSummary(results) {
    const phonemes = Object.keys(results);
    let correctCount = 0;
    let totalAnalyzed = 0;
    let errors = [];
    
    for (const phoneme of phonemes) {
      const result = results[phoneme];
      
      if (result && !result.error) {
        totalAnalyzed++;
        if (result.correct) {
          correctCount++;
        }
      } else if (result && result.error) {
        errors.push({ phoneme, error: result.error });
      }
    }
    
    return {
      totalPhonemes: phonemes.length,
      analyzedPhonemes: totalAnalyzed,
      correctPhonemes: correctCount,
      incorrectPhonemes: totalAnalyzed - correctCount,
      errors: errors,
      accuracy: totalAnalyzed > 0 ? Math.round((correctCount / totalAnalyzed) * 100) : 0,
      overallScore: this.calculateOverallScore(results)
    };
  }
}

export default AudioService;