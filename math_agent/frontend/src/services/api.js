// API service layer
import axios from 'axios';
import toast from 'react-hot-toast';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth headers if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 500) {
      toast.error('Server error occurred. Please try again.');
    } else if (error.response?.status === 400) {
      toast.error(error.response.data.detail || 'Invalid request');
    } else if (error.code === 'ECONNABORTED') {
      toast.error('Request timeout. Please check your connection.');
    }
    return Promise.reject(error);
  }
);

export const mathAPI = {
  solveProblem: async (questionData) => {
    const response = await api.post('/solve', questionData);
    return response.data;
  },

  improveSolution: async (improvementData) => {
    const response = await api.post('/improve', improvementData);
    return response.data;
  },

  generateSimilarProblems: async (questionData) => {
    const response = await api.post('/similar-problems', questionData);
    return response.data;
  },

  checkAnswer: async (questionData) => {
    const response = await api.post('/check-answer', null, {
      params: questionData
    });
    return response.data;
  },

  simplifySolution: async (solution, targetLevel) => {
    const response = await api.post('/simplify', null, {
      params: { solution, target_level: targetLevel }
    });
    return response.data;
  },

  searchKnowledgeBase: async (query, limit = 5) => {
    const response = await api.get('/knowledge-base/search', {
      params: { query, limit }
    });
    return response.data;
  }
};

export const feedbackAPI = {
  submitFeedback: async (feedbackData) => {
    const response = await api.post('/feedback/submit', feedbackData);
    return response.data;
  },

  getFeedbackStats: async (days = 30) => {
    const response = await api.get(`/feedback/stats?days=${days}`);
    return response.data;
  },

  getLearningInsights: async () => {
    const response = await api.get('/feedback/insights');
    return response.data;
  },

  applyLearning: async () => {
    const response = await api.post('/feedback/apply-learning');
    return response.data;
  },

  getImprovementSuggestions: async (question) => {
    const response = await api.get(`/feedback/suggestions/${encodeURIComponent(question)}`);
    return response.data;
  },

  analyzeWithDSPy: async (feedbackText, question, solution) => {
    const response = await api.post('/feedback/analyze-with-dspy', null, {
      params: { feedback_text: feedbackText, question, solution }
    });
    return response.data;
  }
};

export const healthAPI = {
  getHealth: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  getDetailedHealth: async () => {
    const response = await api.get('/health/detailed');
    return response.data;
  },

  getServiceHealth: async (serviceName) => {
    const response = await api.get(`/health/services/${serviceName}`);
    return response.data;
  }
};

export default api;