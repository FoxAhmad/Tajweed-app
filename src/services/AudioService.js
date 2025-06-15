/**
 * Enhanced AudioService for handling audio recording, segmentation and analysis
 * Now supports CNN, Whisper, and Wave2Vec models
 */
class AudioService {
  constructor(apiBaseUrl = 'http://localhost:5000') {
    this.apiBaseUrl = apiBaseUrl;
    this.supportedModels = ['cnn', 'whisper', 'wave2vec'];
    this.defaultModel = 'cnn'; // CNN for fastest feedback
    
    // Model capabilities for user guidance
    this.modelCapabilities = {
      'cnn': {
        speed: 'very-fast',
        accuracy: 'good',
        bestFor: 'quick-feedback',
        processingTime: '< 1s',
        description: 'Fast spectrogram analysis for instant feedback'
      },
      'whisper': {
        speed: 'slower',
        accuracy: 'very-high',
        bestFor: 'detailed-analysis',
        processingTime: '3-5s',
        description: 'Contextual analysis with highest accuracy'
      },
      'wave2vec': {
        speed: 'medium',
        accuracy: 'high',
        bestFor: 'phoneme-precision',
        processingTime: '2-3s',
        description: 'Precise phoneme-level analysis'
      }
    };
  }

  /**
   * Get model information and capabilities
   * @param {string} modelName - The model name
   * @returns {Object} - Model information
   */
  getModelInfo(modelName) {
    return this.modelCapabilities[modelName] || null;
  }

  /**
   * Get model recommendations based on user needs
   * @param {string} useCase - The use case ('beginner', 'quick-practice', 'detailed-analysis', etc.)
   * @returns {string} - Recommended model name
   */
  getModelRecommendation(useCase) {
    const recommendations = {
      'beginner': 'cnn',
      'quick-practice': 'cnn',
      'detailed-analysis': 'whisper',
      'phoneme-focus': 'wave2vec',
      'advanced-user': 'whisper',
      'mobile': 'cnn',
      'fast': 'cnn'
    };
    
    return recommendations[useCase] || this.defaultModel;
  }

  /**
   * Validate model name
   * @param {string} model - The model name to validate
   * @returns {boolean} - Whether the model is supported
   */
  isModelSupported(model) {
    return this.supportedModels.includes(model);
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
   * @param {string} model - The model to use (cnn, whisper, or wave2vec)
   * @returns {Promise<Object>} - The segmentation results
   */
  async segmentAudio(audioBlob, letter, model = 'cnn') {
    try {
      // Validate model
      if (!this.isModelSupported(model)) {
        throw new Error(`Unsupported model: ${model}. Supported models: ${this.supportedModels.join(', ')}`);
      }

      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      formData.append('letter', letter);
      formData.append('model', model);

      console.log(`Segmenting audio for letter '${letter}' using ${model.toUpperCase()} model`);

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
      
      console.log(`✓ Segmentation completed with ${model.toUpperCase()}: ${data.total_segments} segments`);
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
   * @param {string} model - The model to use (cnn, whisper, or wave2vec)
   * @returns {Promise<Object>} - The analysis results
   */
  async analyzeSegment(segmentId, phoneme, model = 'cnn') {
    try {
      // Validate model
      if (!this.isModelSupported(model)) {
        throw new Error(`Unsupported model: ${model}. Supported models: ${this.supportedModels.join(', ')}`);
      }

      console.log(`Analyzing segment: ${segmentId} (phoneme: ${phoneme}) with ${model.toUpperCase()} model`);
      
      const formData = new FormData();
      formData.append('segment_id', segmentId);
      formData.append('phoneme', phoneme);
      formData.append('model', model);

      const startTime = performance.now();

      const response = await fetch(`${this.apiBaseUrl}/analyze-segment`, {
        method: 'POST',
        body: formData,
      });

      const endTime = performance.now();
      const requestTime = (endTime - startTime) / 1000;

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

      const result = await response.json();
      
      // Add request timing info
      result.request_time = requestTime;
      result.model_used = model;
      
      console.log(`✓ ${model.toUpperCase()} analysis for '${phoneme}': ${result.correct ? 'Correct' : 'Incorrect'} (${result.confidence}% confidence, ${requestTime.toFixed(2)}s)`);
      
      return result;
    } catch (error) {
      console.error('Error analyzing segment:', error);
      throw error;
    }
  }

  /**
   * Analyze all phonemes in a segmented recording using the simplified endpoint
   * @param {Array} segments - Array of segment objects from segmentAudio (optional, not used in new approach)
   * @param {string} model - The model to use (cnn, whisper, or wave2vec)
   * @returns {Promise<Object>} - Results for each phoneme
   */
  async analyzeAllSegments(segments, model = 'cnn') {
    try {
      // Validate model
      if (!this.isModelSupported(model)) {
        throw new Error(`Unsupported model: ${model}. Supported models: ${this.supportedModels.join(', ')}`);
      }

      console.log(`Starting analysis of all segments with ${model.toUpperCase()} model`);
      console.log('Segments provided:', segments);
      
      const startTime = performance.now();
      
      // Use the simplified endpoint that automatically finds and analyzes all segments
      const formData = new FormData();
      formData.append('model', model);

      const response = await fetch(`${this.apiBaseUrl}/analyze-all-segments`, {
        method: 'POST',
        body: formData,
      });

      const endTime = performance.now();
      const totalTime = (endTime - startTime) / 1000;

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
      
      console.log(`✓ ${model.toUpperCase()} batch analysis completed: ${Object.keys(data.results).length} phonemes analyzed in ${totalTime.toFixed(2)}s`);
      
      // Add timing and model info to results
      const enhancedResults = {};
      Object.entries(data.results).forEach(([phoneme, result]) => {
        enhancedResults[phoneme] = {
          ...result,
          model_used: model,
          batch_analysis: true,
          total_batch_time: totalTime
        };
      });
      
      return enhancedResults;
      
    } catch (error) {
      console.error('Error analyzing all segments:', error);
      throw error;
    }
  }

  /**
   * Compare pronunciation using multiple models
   * @param {Blob} audioBlob - The audio blob to analyze
   * @param {string} phoneme - The phoneme to analyze
   * @param {Array} models - Array of models to use (default: ['cnn', 'wave2vec'])
   * @returns {Promise<Object>} - Comparison results from all models
   */
  async compareModels(audioBlob, phoneme, models = ['cnn', 'wave2vec']) {
    try {
      console.log(`Comparing models ${models.join(', ')} for phoneme '${phoneme}'`);
      
      const results = {};
      const errors = {};
      const timings = {};

      for (const model of models) {
        if (!this.isModelSupported(model)) {
          errors[model] = `Unsupported model: ${model}`;
          continue;
        }

        try {
          const startTime = performance.now();
          
          const formData = new FormData();
          formData.append('audio', audioBlob, 'recording.wav');
          formData.append('model', model);
          formData.append('phoneme', phoneme);

          const response = await fetch(`${this.apiBaseUrl}/analyze-tajweed`, {
            method: 'POST',
            body: formData,
          });

          const endTime = performance.now();
          timings[model] = (endTime - startTime) / 1000;

          if (!response.ok) {
            throw new Error(`${model} analysis failed: ${response.statusText}`);
          }

          const result = await response.json();
          
          if (result.error) {
            throw new Error(result.error);
          }

          results[model] = {
            ...result,
            request_time: timings[model],
            model_info: this.getModelInfo(model)
          };
          
          console.log(`✓ ${model.toUpperCase()}: ${result.correct ? 'Correct' : 'Incorrect'} (${result.confidence}%, ${timings[model].toFixed(2)}s)`);
          
        } catch (error) {
          console.error(`✗ ${model.toUpperCase()} failed:`, error);
          errors[model] = error.message;
        }
      }

      // Calculate consensus
      const consensus = this.calculateConsensus(results);
      
      // Generate comparison summary
      const comparison = this.generateModelComparison(results, timings);

      return {
        results,
        errors,
        consensus,
        comparison,
        phoneme,
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      console.error('Error in model comparison:', error);
      throw error;
    }
  }

  /**
   * Calculate consensus from multiple model results
   * @param {Object} results - Results from different models
   * @returns {Object} - Consensus analysis
   */
  calculateConsensus(results) {
    const modelCount = Object.keys(results).length;
    if (modelCount === 0) return null;

    let correctCount = 0;
    let totalConfidence = 0;
    const recommendations = [];

    Object.values(results).forEach(result => {
      if (result.correct) correctCount++;
      totalConfidence += result.confidence || 0;
      if (result.recommendation) recommendations.push(result.recommendation);
    });

    const consensusCorrect = correctCount > modelCount / 2;
    const averageConfidence = totalConfidence / modelCount;
    const agreement = (correctCount / modelCount) * 100;

    return {
      isCorrect: consensusCorrect,
      confidence: Math.round(averageConfidence),
      agreement: Math.round(agreement),
      modelsAgreeing: correctCount,
      totalModels: modelCount,
      recommendation: this.generateConsensusRecommendation(consensusCorrect, averageConfidence, agreement)
    };
  }

  /**
   * Generate consensus recommendation
   * @param {boolean} isCorrect - Whether the consensus is correct
   * @param {number} confidence - Average confidence
   * @param {number} agreement - Percentage agreement between models
   * @returns {string} - Consensus recommendation
   */
  generateConsensusRecommendation(isCorrect, confidence, agreement) {
    if (isCorrect && confidence > 85 && agreement > 80) {
      return "Excellent! Multiple AI models agree your pronunciation is very accurate.";
    } else if (isCorrect && confidence > 70) {
      return "Good pronunciation! Most models detected correct articulation with room for slight improvement.";
    } else if (isCorrect) {
      return "Your pronunciation is generally correct, but could benefit from more practice for clarity.";
    } else if (confidence > 70 && agreement > 60) {
      return "Multiple models detected pronunciation issues. Focus on the articulation point and try again.";
    } else {
      return "The pronunciation needs improvement. Listen to the reference audio and practice the correct articulation.";
    }
  }

  /**
   * Generate model comparison summary
   * @param {Object} results - Results from different models
   * @param {Object} timings - Timing data for each model
   * @returns {Object} - Comparison summary
   */
  generateModelComparison(results, timings) {
    const comparison = {};
    
    Object.entries(results).forEach(([model, result]) => {
      comparison[model] = {
        correct: result.correct,
        confidence: result.confidence,
        processingTime: result.processing_time || 0,
        requestTime: timings[model] || 0,
        modelInfo: this.getModelInfo(model),
        recommendation: result.recommendation
      };
    });

    // Find fastest and most confident
    const fastest = this.findFastestModel(comparison);
    const mostConfident = this.findMostConfidentModel(comparison);
    const mostAccurate = this.findMostAccurateModel(comparison);

    return {
      individual: comparison,
      fastest,
      mostConfident,
      mostAccurate,
      summary: this.generateComparisonSummary(comparison)
    };
  }

  /**
   * Find the fastest model
   * @param {Object} comparison - Comparison data
   * @returns {string} - Fastest model name
   */
  findFastestModel(comparison) {
    let fastest = null;
    let shortestTime = Infinity;

    Object.entries(comparison).forEach(([model, data]) => {
      const totalTime = (data.processingTime || 0) + (data.requestTime || 0);
      if (totalTime < shortestTime) {
        shortestTime = totalTime;
        fastest = model;
      }
    });

    return fastest;
  }

  /**
   * Find the most confident model
   * @param {Object} comparison - Comparison data
   * @returns {string} - Most confident model name
   */
  findMostConfidentModel(comparison) {
    let mostConfident = null;
    let highestConfidence = 0;

    Object.entries(comparison).forEach(([model, data]) => {
      if (data.confidence && data.confidence > highestConfidence) {
        highestConfidence = data.confidence;
        mostConfident = model;
      }
    });

    return mostConfident;
  }

  /**
   * Find the most accurate model (based on combination of correctness and confidence)
   * @param {Object} comparison - Comparison data
   * @returns {string} - Most accurate model name
   */
  findMostAccurateModel(comparison) {
    let mostAccurate = null;
    let bestScore = 0;

    Object.entries(comparison).forEach(([model, data]) => {
      // Calculate accuracy score: correctness weight + confidence
      const score = (data.correct ? 100 : 0) + (data.confidence || 0);
      if (score > bestScore) {
        bestScore = score;
        mostAccurate = model;
      }
    });

    return mostAccurate;
  }

  /**
   * Generate comparison summary
   * @param {Object} comparison - Comparison data
   * @returns {Object} - Summary statistics
   */
  generateComparisonSummary(comparison) {
    const models = Object.keys(comparison);
    const agreements = Object.values(comparison).filter(r => r.correct).length;
    const avgConfidence = Object.values(comparison).reduce((sum, r) => sum + (r.confidence || 0), 0) / models.length;
    const avgTime = Object.values(comparison).reduce((sum, r) => sum + ((r.processingTime || 0) + (r.requestTime || 0)), 0) / models.length;

    return {
      totalModels: models.length,
      agreement: Math.round((agreements / models.length) * 100),
      averageConfidence: Math.round(avgConfidence),
      averageTime: Math.round(avgTime * 1000) / 1000, // Round to 3 decimal places
      recommendation: agreements > models.length / 2 ? 'correct' : 'needs_improvement'
    };
  }

  /**
   * Legacy method: Analyze all segments individually (kept for backward compatibility)
   * @param {Array} segments - Array of segment objects from segmentAudio
   * @param {string} model - The model to use (cnn, whisper, or wave2vec)
   * @returns {Promise<Object>} - Results for each phoneme
   */
  async analyzeAllSegmentsIndividually(segments, model = 'cnn') {
    try {
      // Validate model
      if (!this.isModelSupported(model)) {
        throw new Error(`Unsupported model: ${model}. Supported models: ${this.supportedModels.join(', ')}`);
      }

      const results = {};
      console.log(`Analyzing ${segments.length} segments individually with ${model.toUpperCase()} model`);
      
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
      
      if (result && !result.error && typeof result.confidence === 'number') {
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
   * Get detailed analysis summary with model performance breakdown
   * @param {Object} results - Results object from analyzeAllSegments
   * @returns {Object} - Enhanced analysis summary
   */
  getAnalysisSummary(results) {
    const phonemes = Object.keys(results);
    let correctCount = 0;
    let totalAnalyzed = 0;
    let errors = [];
    let totalProcessingTime = 0;
    let modelBreakdown = {};
    
    for (const phoneme of phonemes) {
      const result = results[phoneme];
      
      if (result && !result.error) {
        totalAnalyzed++;
        if (result.correct) {
          correctCount++;
        }
        
        // Track processing time
        if (result.processing_time) {
          totalProcessingTime += result.processing_time;
        }
        
        // Track model performance
        const model = result.model_used || result.model_name || 'unknown';
        if (!modelBreakdown[model]) {
          modelBreakdown[model] = { 
            total: 0, 
            correct: 0, 
            totalTime: 0,
            avgConfidence: 0
          };
        }
        
        modelBreakdown[model].total++;
        if (result.correct) {
          modelBreakdown[model].correct++;
        }
        modelBreakdown[model].totalTime += result.processing_time || 0;
        modelBreakdown[model].avgConfidence += result.confidence || 0;
        
      } else if (result && result.error) {
        errors.push({ phoneme, error: result.error });
      }
    }
    
    // Calculate model statistics
    Object.values(modelBreakdown).forEach(stats => {
      stats.accuracy = stats.total > 0 ? Math.round((stats.correct / stats.total) * 100) : 0;
      stats.avgTime = stats.total > 0 ? Math.round((stats.totalTime / stats.total) * 1000) / 1000 : 0;
      stats.avgConfidence = stats.total > 0 ? Math.round(stats.avgConfidence / stats.total) : 0;
    });
    
    return {
      totalPhonemes: phonemes.length,
      analyzedPhonemes: totalAnalyzed,
      correctPhonemes: correctCount,
      incorrectPhonemes: totalAnalyzed - correctCount,
      errors: errors,
      accuracy: totalAnalyzed > 0 ? Math.round((correctCount / totalAnalyzed) * 100) : 0,
      overallScore: this.calculateOverallScore(results),
      totalProcessingTime: Math.round(totalProcessingTime * 1000) / 1000,
      avgProcessingTime: totalAnalyzed > 0 ? Math.round((totalProcessingTime / totalAnalyzed) * 1000) / 1000 : 0,
      modelBreakdown: modelBreakdown,
      hasErrors: errors.length > 0,
      errorCount: errors.length
    };
  }

  /**
   * Validate audio blob before processing
   * @param {Blob} audioBlob - The audio blob to validate
   * @param {Object} options - Validation options
   * @returns {Object} - Validation result
   */
  validateAudio(audioBlob, options = {}) {
    const { 
      maxSize = 10 * 1024 * 1024, // 10MB
      minSize = 1024, // 1KB
      allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/webm']
    } = options;

    if (!audioBlob) {
      return { valid: false, error: 'No audio file provided' };
    }

    if (audioBlob.size > maxSize) {
      return { 
        valid: false, 
        error: `Audio file too large. Maximum size is ${Math.round(maxSize / 1024 / 1024)}MB` 
      };
    }

    if (audioBlob.size < minSize) {
      return { 
        valid: false, 
        error: `Audio file too small. Minimum size is ${minSize} bytes` 
      };
    }

    if (audioBlob.type && !allowedTypes.includes(audioBlob.type)) {
      return { 
        valid: false, 
        error: `Unsupported audio format. Supported formats: ${allowedTypes.join(', ')}` 
      };
    }

    return { valid: true };
  }

  /**
   * Get processing recommendations based on user preferences
   * @param {Object} userProfile - User preferences and context
   * @returns {Object} - Processing recommendations
   */
  getProcessingRecommendations(userProfile = {}) {
    const { 
      experience = 'beginner', 
      priority = 'balanced', 
      deviceType = 'desktop',
      batteryLevel = 100 
    } = userProfile;

    // Adjust recommendations based on device constraints
    if (deviceType === 'mobile' || batteryLevel < 30) {
      return {
        primaryModel: 'cnn',
        fallbackModel: null,
        usePipeline: false,
        reason: 'Optimized for mobile performance and battery life'
      };
    }

    if (priority === 'speed') {
      return {
        primaryModel: 'cnn',
        fallbackModel: 'wave2vec',
        usePipeline: false,
        reason: 'Prioritizing speed for quick feedback'
      };
    }

    if (priority === 'accuracy') {
      return {
        primaryModel: 'whisper',
        fallbackModel: 'wave2vec',
        usePipeline: true,
        reason: 'Prioritizing accuracy for detailed analysis'
      };
    }

    // Experience-based recommendations
    const recommendations = {
      beginner: {
        primaryModel: 'cnn',
        fallbackModel: 'wave2vec',
        usePipeline: false,
        reason: 'Fast feedback helps build confidence'
      },
      intermediate: {
        primaryModel: 'wave2vec',
        fallbackModel: 'cnn',
        usePipeline: true,
        reason: 'Balanced accuracy and speed for skill development'
      },
      advanced: {
        primaryModel: 'whisper',
        fallbackModel: 'wave2vec',
        usePipeline: true,
        reason: 'Most accurate analysis for perfecting pronunciation'
      }
    };

    return recommendations[experience] || recommendations.beginner;
  }

  /**
   * Check server status and available models
   * @returns {Promise<Object>} - Server status information
   */
  async checkServerStatus() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/status`);
      
      if (!response.ok) {
        throw new Error(`Server status check failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error checking server status:', error);
      return {
        status: 'error',
        error: error.message,
        available_models: {}
      };
    }
  }

  /**
   * Get available models for a specific phoneme
   * @param {string} phoneme - The phoneme to check
   * @returns {Promise<Object>} - Available models for the phoneme
   */
  async getAvailableModelsForPhoneme(phoneme) {
    try {
      const response = await fetch(`${this.apiBaseUrl}/models`);
      
      if (!response.ok) {
        throw new Error(`Failed to get model info: ${response.statusText}`);
      }

      const data = await response.json();
      const phonemeMapping = data.phoneme_model_mapping || {};
      
      return phonemeMapping[phoneme] || {};
    } catch (error) {
      console.error('Error getting available models:', error);
      return {};
    }
  }
}

export default AudioService;