import React, { useState } from 'react';
import { Star, Send, AlertCircle } from 'lucide-react';

const FeedbackForm = ({ onSubmit, isSubmitting, onCancel }) => {
  const [feedbackData, setFeedbackData] = useState({
    rating: 5,
    feedback_type: 'general',
    feedback_text: '',
    user_suggestions: ''
  });

  const [errors, setErrors] = useState({});

  const validateForm = () => {
    const newErrors = {};
    
    if (!feedbackData.feedback_text.trim() && !feedbackData.user_suggestions.trim()) {
      newErrors.content = 'Please provide either feedback text or suggestions';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    onSubmit(feedbackData);
  };

  const handleRatingChange = (rating) => {
    setFeedbackData(prev => ({ ...prev, rating }));
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-2xl mx-auto">
      <h3 className="text-xl font-semibold mb-6 flex items-center">
        <Star className="w-5 h-5 mr-2 text-yellow-500" />
        Share Your Feedback
      </h3>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Rating Section */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            How would you rate this solution? *
          </label>
          <div className="flex space-x-2">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                type="button"
                onClick={() => handleRatingChange(star)}
                className={`p-1 transition-colors ${
                  star <= feedbackData.rating
                    ? 'text-yellow-400 hover:text-yellow-500'
                    : 'text-gray-300 hover:text-gray-400'
                }`}
              >
                <Star className={`w-6 h-6 ${star <= feedbackData.rating ? 'fill-current' : ''}`} />
              </button>
            ))}
            <span className="ml-2 text-sm text-gray-600">
              ({feedbackData.rating}/5)
            </span>
          </div>
        </div>

        {/* Feedback Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            What aspect would you like to comment on?
          </label>
          <select
            value={feedbackData.feedback_type}
            onChange={(e) => setFeedbackData(prev => ({ ...prev, feedback_type: e.target.value }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="general">General Feedback</option>
            <option value="correctness">Mathematical Correctness</option>
            <option value="clarity">Explanation Clarity</option>
            <option value="completeness">Solution Completeness</option>
            <option value="difficulty">Difficulty Level</option>
            <option value="relevance">Relevance to Question</option>
          </select>
        </div>

        {/* Feedback Text */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Comments
          </label>
          <textarea
            value={feedbackData.feedback_text}
            onChange={(e) => setFeedbackData(prev => ({ ...prev, feedback_text: e.target.value }))}
            placeholder="Share your thoughts about this solution..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={4}
            maxLength={1000}
          />
          <div className="text-sm text-gray-500 mt-1">
            {feedbackData.feedback_text.length}/1000 characters
          </div>
        </div>

        {/* Suggestions */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Suggestions for Improvement
          </label>
          <textarea
            value={feedbackData.user_suggestions}
            onChange={(e) => setFeedbackData(prev => ({ ...prev, user_suggestions: e.target.value }))}
            placeholder="How could this solution be improved?"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={3}
            maxLength={500}
          />
          <div className="text-sm text-gray-500 mt-1">
            {feedbackData.user_suggestions.length}/500 characters
          </div>
        </div>

        {/* Error Display */}
        {errors.content && (
          <div className="flex items-center p-3 bg-red-50 border border-red-200 rounded-lg">
            <AlertCircle className="w-5 h-5 text-red-600 mr-2" />
            <span className="text-sm text-red-800">{errors.content}</span>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-3 pt-4">
          <button
            type="submit"
            disabled={isSubmitting}
            className="flex items-center px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4 mr-2" />
            {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
          </button>
          
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
            >
              Cancel
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default FeedbackForm;