import React, { useState } from 'react';
import { Send, Loader2, Calculator, HelpCircle } from 'lucide-react';

const MathInput = ({ onSubmit, isLoading }) => {
  const [question, setQuestion] = useState('');
  const [difficulty, setDifficulty] = useState('medium');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim() && !isLoading) {
      onSubmit(question.trim(), difficulty);
    }
  };

  const exampleQuestions = [
    "Solve the equation: 2x + 5 = 13",
    "Find the derivative of f(x) = x² + 3x - 2",
    "Calculate the area of a circle with radius 7 cm",
    "Simplify: (x² - 4)/(x + 2)",
    "What is the probability of rolling a sum of 7 with two dice?"
  ];

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="flex items-center mb-4">
        <Calculator className="w-6 h-6 text-blue-600 mr-2" />
        <h2 className="text-2xl font-bold text-gray-800">Ask a Math Question</h2>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="question" className="block text-sm font-medium text-gray-700 mb-2">
            Your Math Problem
          </label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your mathematical question here..."
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={4}
            disabled={isLoading}
            maxLength={2000}
          />
          <div className="text-sm text-gray-500 mt-1">
            {question.length}/2000 characters
          </div>
        </div>

        <div>
          <label htmlFor="difficulty" className="block text-sm font-medium text-gray-700 mb-2">
            Difficulty Level
          </label>
          <select
            id="difficulty"
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isLoading}
          >
            <option value="easy">Easy - Basic concepts</option>
            <option value="medium">Medium - Standard level</option>
            <option value="hard">Hard - Advanced concepts</option>
          </select>
        </div>

        <button
          type="submit"
          disabled={!question.trim() || isLoading}
          className="w-full flex items-center justify-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Solving...
            </>
          ) : (
            <>
              <Send className="w-4 h-4 mr-2" />
              Solve Problem
            </>
          )}
        </button>
      </form>

      <div className="mt-6">
        <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
          <HelpCircle className="w-4 h-4 mr-1" />
          Example Questions:
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {exampleQuestions.map((example, index) => (
            <button
              key={index}
              onClick={() => setQuestion(example)}
              className="text-left p-2 text-sm text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
              disabled={isLoading}
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MathInput;