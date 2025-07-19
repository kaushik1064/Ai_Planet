// Custom hook for math agent
// Custom hook for feedback
import { useState, useCallback } from 'react';
import { mathAPI } from '../services/api';
import toast from 'react-hot-toast';

export const useMathAgent = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [currentSolution, setCurrentSolution] = useState(null);
  const [solutionHistory, setSolutionHistory] = useState([]);
  const [error, setError] = useState(null);

  const solveProblem = useCallback(async (question, difficulty = 'medium', userId = null) => {
    setIsLoading(true);
    setError(null);

    try {
      const requestData = {
        question,
        difficulty,
        user_id: userId
      };

      const response = await mathAPI.solveProblem(requestData);

      if (response.success) {
        setCurrentSolution(response);
        setSolutionHistory(prev => [response, ...prev.slice(0, 9)]); // Keep last 10
        toast.success('Problem solved successfully!');
        return response;
      } else {
        setError(response.error || 'Failed to solve problem');
        toast.error(response.error || 'Failed to solve problem');
        return null;
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
      setError(errorMessage);
      toast.error(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const improveSolution = useCallback(async (question, solution, feedbackData) => {
    setIsLoading(true);
    setError(null);

    try {
      const improvementData = {
        question,
        solution,
        feedback: feedbackData
      };

      const response = await mathAPI.improveSolution(improvementData);

      if (response.success) {
        setCurrentSolution(prev => ({
          ...prev,
          solution: response.improved_solution,
          improvement_reason: response.improvement_reason,
          improved: true
        }));
        toast.success('Solution improved successfully!');
        return response;
      } else {
        setError(response.error || 'Failed to improve solution');
        toast.error(response.error || 'Failed to improve solution');
        return null;
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
      setError(errorMessage);
      toast.error(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const generateSimilarProblems = useCallback(async (question, count = 3) => {
    try {
      const response = await mathAPI.generateSimilarProblems({ question, count });
      
      if (response.success) {
        toast.success(`Generated ${response.problems.length} similar problems`);
        return response;
      } else {
        toast.error(response.error || 'Failed to generate similar problems');
        return null;
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
      toast.error(errorMessage);
      return null;
    }
  }, []);

  const checkAnswer = useCallback(async (question, studentAnswer, correctAnswer) => {
    try {
      const response = await mathAPI.checkAnswer({
        question,
        student_answer: studentAnswer,
        correct_answer: correctAnswer
      });

      if (response.success) {
        if (response.is_correct) {
          toast.success('Correct answer!');
        } else {
          toast.error('Incorrect answer. Try again!');
        }
        return response;
      } else {
        toast.error(response.error || 'Failed to check answer');
        return null;
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
      toast.error(errorMessage);
      return null;
    }
  }, []);

  const simplifySolution = useCallback(async (solution, targetLevel = 'high_school') => {
    try {
      const response = await mathAPI.simplifySolution(solution, targetLevel);
      
      if (response.success) {
        toast.success('Solution simplified successfully!');
        return response;
      } else {
        toast.error(response.error || 'Failed to simplify solution');
        return null;
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
      toast.error(errorMessage);
      return null;
    }
  }, []);

  const searchKnowledgeBase = useCallback(async (query, limit = 5) => {
    try {
      const response = await mathAPI.searchKnowledgeBase(query, limit);
      
      if (response.success) {
        return response.results;
      } else {
        toast.error('Failed to search knowledge base');
        return [];
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
      toast.error(errorMessage);
      return [];
    }
  }, []);

  const clearHistory = useCallback(() => {
    setSolutionHistory([]);
    toast.success('Solution history cleared');
  }, []);

  const clearCurrentSolution = useCallback(() => {
    setCurrentSolution(null);
    setError(null);
  }, []);

  return {
    // State
    isLoading,
    currentSolution,
    solutionHistory,
    error,
    
    // Actions
    solveProblem,
    improveSolution,
    generateSimilarProblems,
    checkAnswer,
    simplifySolution,
    searchKnowledgeBase,
    clearHistory,
    clearCurrentSolution,
    
    // Computed values
    hasHistory: solutionHistory.length > 0,
    hasSolution: currentSolution !== null
  };
};