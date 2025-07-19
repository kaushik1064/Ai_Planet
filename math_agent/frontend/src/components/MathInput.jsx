// Math problem input
import React, { useState } from 'react';
import { Send, Loader2 } from 'lucide-react';

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
    "Solve the equation 2x + 5 = 13",
    "Find the derivative of f(x) = x² + 3x + 2",
    "Calculate the area of a circle with radius 5",
    "Evaluate the integral ∫(2x + 1)dx",
    "Find the probability of rolling a sum of 7 with two dice"
  ];

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-semibold mb-4">Enter Your Math Problem</h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="question" className="block text-sm font-medium text-gray-700 mb-2">
            Mathematical Question
          </label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your mathematical problem here..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={4}
            disabled={isLoading}
          />
        </div>
        
        <div>
          <label htmlFor="difficulty" className="block text-sm font-medium text-gray-700 mb-2">
            Difficulty Level
          </label>
          <select
            id="difficulty"
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isLoading}
          >
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>
        </div>
        
        <button
          type="submit"
          disabled={!question.trim() || isLoading}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
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
        <h3 className="text-lg font-medium mb-3">Example Questions</h3>
        <div className="grid gap-2">
          {exampleQuestions.map((example, index) => (
            <button
              key={index}
              onClick={() => setQuestion(example)}
              className="text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded border transition-colors"
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