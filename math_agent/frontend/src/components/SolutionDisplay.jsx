// Step-by-step solution display
import React, { useState } from 'react';
import { ThumbsUp, ThumbsDown, Star, RefreshCw, BookOpen, MessageSquare } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const SolutionDisplay = ({ solution, onFeedback, onImprove, onGenerateSimilar }) => {
  const [showFeedback, setShowFeedback] = useState(false);
  const [rating, setRating] = useState(0);
  const [feedbackText, setFeedbackText] = useState('');
  const [feedbackType, setFeedbackType] = useState('general');

  const handleFeedbackSubmit = () => {
    if (rating > 0) {
      onFeedback({
        rating,
        feedback_text: feedbackText,
        feedback_type: feedbackType
      });
      setShowFeedback(false);
      setRating(0);
      setFeedbackText('');
      setFeedbackType('general');
    }
  };

  const handleQuickFeedback = (isPositive) => {
    onFeedback({
      rating: isPositive ? 5 : 2,
      feedback_text: isPositive ? 'Good solution' : 'Needs improvement',
      feedback_type: 'general'
    });
  };

  if (!solution) return null;

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold">Solution</h2>
        <div className="flex items-center space-x-2 text-sm text-gray-600">
          <span>Route: {solution.routing_info?.route_taken || 'Unknown'}</span>
          <span>•</span>
          <span>Time: {solution.processing_time?.toFixed(2)}s</span>
        </div>
      </div>
      
      {solution.success ? (
        <div className="space-y-6">
          {/* Solution Content */}
          <div className="prose max-w-none">
            <ReactMarkdown>{solution.solution}</ReactMarkdown>
          </div>
          
          {/* Steps */}
          {solution.steps && solution.steps.length > 0 && (
            <div>
              <h3 className="text-lg font-medium mb-3">Step-by-Step Solution</h3>
              <div className="space-y-3">
                {solution.steps.map((step, index) => (
                  <div key={index} className="flex">
                    <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                      {step.step_number || index + 1}
                    </div>
                    <div className="ml-3 flex-1">
                      <p className="text-gray-900">{step.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Similar Problems */}
          {solution.similar_problems && solution.similar_problems.length > 0 && (
            <div>
              <h3 className="text-lg font-medium mb-3">Similar Problems Found</h3>
              <div className="space-y-2">
                {solution.similar_problems.slice(0, 3).map((problem, index) => (
                  <div key={index} className="p-3 bg-gray-50 rounded border">
                    <p className="text-sm text-gray-700">{problem.question}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      Relevance: {(problem.relevance_score * 100).toFixed(1)}%
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Web Sources */}
          {solution.web_sources && solution.web_sources.length > 0 && (
            <div>
              <h3 className="text-lg font-medium mb-3">Additional Resources</h3>
              <div className="space-y-2">
                {solution.web_sources.slice(0, 3).map((source, index) => (
                  <div key={index} className="p-3 bg-blue-50 rounded border">
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800 font-medium"
                    >
                      {source.title}
                    </a>
                    <p className="text-sm text-gray-600 mt-1">{source.summary}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Warnings */}
          {solution.warnings && solution.warnings.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded p-3">
              <h4 className="text-yellow-800 font-medium">Note:</h4>
              <ul className="list-disc list-inside text-yellow-700 text-sm mt-1">
                {solution.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Action Buttons */}
          <div className="flex flex-wrap gap-3 pt-4 border-t">
            <button
              onClick={() => handleQuickFeedback(true)}
              className="flex items-center px-3 py-2 bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors"
            >
              <ThumbsUp className="w-4 h-4 mr-1" />
              Helpful
            </button>
            
            <button
              onClick={() => handleQuickFeedback(false)}
              className="flex items-center px-3 py-2 bg-red-100 text-red-700 rounded hover:bg-red-200 transition-colors"
            >
              <ThumbsDown className="w-4 h-4 mr-1" />
              Needs Work
            </button>
            
            <button
              onClick={() => setShowFeedback(true)}
              className="flex items-center px-3 py-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
            >
              <MessageSquare className="w-4 h-4 mr-1" />
              Detailed Feedback
            </button>
            
            <button
              onClick={onGenerateSimilar}
              className="flex items-center px-3 py-2 bg-purple-100 text-purple-700 rounded hover:bg-purple-200 transition-colors"
            >
              <BookOpen className="w-4 h-4 mr-1" />
              Similar Problems
            </button>
            
            <button
              onClick={() => onImprove(solution.question, solution.solution, { rating: 3, feedback_text: 'Please improve this solution' })}
              className="flex items-center px-3 py-2 bg-orange-100 text-orange-700 rounded hover:bg-orange-200 transition-colors"
            >
              <RefreshCw className="w-4 h-4 mr-1" />
              Improve Solution
            </button>
          </div>
          
          {/* Detailed Feedback Modal */}
          {showFeedback && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
                <h3 className="text-lg font-semibold mb-4">Provide Feedback</h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Rating (1-5 stars)
                    </label>
                    <div className="flex space-x-1">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <button
                          key={star}
                          onClick={() => setRating(star)}
                          className={`w-8 h-8 ${
                            star <= rating ? 'text-yellow-400' : 'text-gray-300'
                          } hover:text-yellow-400 transition-colors`}
                        >
                          <Star className="w-full h-full fill-current" />
                        </button>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Feedback Type
                    </label>
                    <select
                      value={feedbackType}
                      onChange={(e) => setFeedbackType(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="general">General</option>
                      <option value="correctness">Correctness</option>
                      <option value="clarity">Clarity</option>
                      <option value="completeness">Completeness</option>
                      <option value="difficulty">Difficulty</option>
                      <option value="relevance">Relevance</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Comments (optional)
                    </label>
                    <textarea
                      value={feedbackText}
                      onChange={(e) => setFeedbackText(e.target.value)}
                      placeholder="Share your thoughts on this solution..."
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      rows={3}
                    />
                  </div>
                  
                  <div className="flex space-x-3">
                    <button
                      onClick={handleFeedbackSubmit}
                      disabled={rating === 0}
                      className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Submit Feedback
                    </button>
                    <button
                      onClick={() => setShowFeedback(false)}
                      className="flex-1 bg-gray-300 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-400"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-red-600">
          <h3 className="font-medium">Error occurred while solving the problem:</h3>
          <p className="mt-2">{solution.error}</p>
        </div>
      )}
    </div>
  );
};