import { useState, useCallback } from 'react';
import { feedbackAPI } from '../services/api';
import toast from 'react-hot-toast';

export const useFeedback = () => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [feedbackStats, setFeedbackStats] = useState(null);
  const [insights, setInsights] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const submitFeedback = useCallback(async (feedbackData) => {
    setIsSubmitting(true);

    try {
      const response = await feedbackAPI.submitFeedback(feedbackData);

      if (response.success) {
        toast.success('Feedback submitted successfully!');
        return response;
      } else {
        toast.error(response.error || 'Failed to submit feedback');
        return null;
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
      toast.error(errorMessage);
      return null;
    } finally {
      setIsSubmitting(false);
    }
  }, []);

  const loadFeedbackStats = useCallback(async (days = 30) => {
    setIsLoading(true);

    try {
      const response = await feedbackAPI.getFeedbackStats(days);
      setFeedbackStats(response);
      return response;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
      toast.error(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadLearningInsights = useCallback(async () => {
    setIsLoading(true);

    try {
      const response = await feedbackAPI.getLearningInsights();
      
      if (response.success) {
        setInsights(response.insights);
        return response.insights;
      } else {
        toast.error('Failed to load learning insights');
        return null;
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
      toast.error(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const applyLearning = useCallback(async () => {
    setIsLoading(true);

    try {
      const response = await feedbackAPI.applyLearning();
      
      if (response.success) {
        toast.success('Learning applied successfully!');
        return response;
      } else {
        toast.error(response.error || 'Failed to apply learning');
        return null;
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
      toast.error(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getImprovementSuggestions = useCallback(async (question) => {
    try {
      const response = await feedbackAPI.getImprovementSuggestions(question);
      
      if (response.success) {
        return response.suggestions;
      } else {
        toast.error('Failed to get improvement suggestions');
        return [];
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
      toast.error(errorMessage);
      return [];
    }
  }, []);

  const analyzeWithDSPy = useCallback(async (feedbackText, question, solution) => {
    setIsLoading(true);

    try {
      const response = await feedbackAPI.analyzeWithDSPy(feedbackText, question, solution);
      
      if (response.success) {
        toast.success('Advanced analysis completed!');
        return response;
      } else {
        if (response.error.includes('not available')) {
          toast.info('Advanced analysis not available');
        } else {
          toast.error(response.error || 'Failed to analyze with DSPy');
        }
        return null;
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
      toast.error(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    // State
    isSubmitting,
    isLoading,
    feedbackStats,
    insights,
    
    // Actions
    submitFeedback,
    loadFeedbackStats,
    loadLearningInsights,
    applyLearning,
    getImprovementSuggestions,
    analyzeWithDSPy,
    
    // Computed values
    hasStats: feedbackStats !== null,
    hasInsights: insights !== null
  };
};