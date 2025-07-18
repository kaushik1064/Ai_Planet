import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { Calculator, BarChart3, MessageSquare, Settings, Home, Activity } from 'lucide-react';

import MathInput from './components/MathInput';
import SolutionDisplay from './components/SolutionDisplay';
import FeedbackDashboard from './components/FeedbackDashboard';
import HealthMonitor from './components/HealthMonitor';
import SimilarProblems from './components/SimilarProblems';

import { useMathAgent } from './hooks/useMathAgent';
import { useFeedback } from './hooks/useFeedback';

import './App.css';

const Navigation = () => {
  const location = useLocation();
  
  const navItems = [
    { path: '/', icon: Home, label: 'Home' },
    { path: '/feedback', icon: MessageSquare, label: 'Feedback' },
    { path: '/analytics', icon: BarChart3, label: 'Analytics' },
    { path: '/health', icon: Activity, label: 'System Health' }
  ];

  return (
    <nav className="bg-white shadow-lg border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <Calculator className="w-8 h-8 text-blue-600 mr-2" />
              <span className="text-xl font-bold text-gray-900">Math Agent</span>
            </div>
          </div>
          
          <div className="flex space-x-8">
            {navItems.map(({ path, icon: Icon, label }) => (
              <Link
                key={path}
                to={path}
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors ${
                  location.pathname === path
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                }`}
              >
                <Icon className="w-4 h-4 mr-2" />
                {label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

const HomePage = () => {
  const { 
    isLoading, 
    currentSolution, 
    solutionHistory, 
    solveProblem, 
    improveSolution, 
    generateSimilarProblems 
  } = useMathAgent();
  
  const { submitFeedback } = useFeedback();
  
  const [showSimilarProblems, setShowSimilarProblems] = useState(false);
  const [similarProblems, setSimilarProblems] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState('');

  const handleSolveProblem = async (question, difficulty) => {
    setCurrentQuestion(question);
    await solveProblem(question, difficulty);
    setShowSimilarProblems(false);
  };

  const handleFeedback = async (feedbackData) => {
    if (currentSolution && currentQuestion) {
      const fullFeedbackData = {
        question: currentQuestion,
        solution: currentSolution.solution,
        ...feedbackData
      };
      
      const result = await submitFeedback(fullFeedbackData);
      
      if (result.success && result.should_trigger_improvement) {
        // Automatically improve the solution based on feedback
        await improveSolution(currentQuestion, currentSolution.solution, feedbackData);
      }
    }
  };

  const handleGenerateSimilar = async () => {
    if (currentQuestion) {
      const result = await generateSimilarProblems(currentQuestion, 3);
      if (result.success) {
        setSimilarProblems(result.problems);
        setShowSimilarProblems(true);
      }
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            AI-Powered Math Problem Solver
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Get step-by-step solutions to mathematical problems with intelligent routing, 
            human-in-the-loop feedback, and continuous learning capabilities.
          </p>
        </div>

        <div className="space-y-8">
          <MathInput 
            onSubmit={handleSolveProblem} 
            isLoading={isLoading} 
          />
          
          {currentSolution && (
            <SolutionDisplay
              solution={currentSolution}
              onFeedback={handleFeedback}
              onImprove={improveSolution}
              onGenerateSimilar={handleGenerateSimilar}
            />
          )}
          
          {showSimilarProblems && similarProblems.length > 0 && (
            <SimilarProblems 
              problems={similarProblems}
              onSolveProblem={handleSolveProblem}
            />
          )}
          
          {solutionHistory.length > 0 && (
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Recent Solutions</h2>
              <div className="space-y-3">
                {solutionHistory.slice(0, 5).map((solution, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="text-sm text-gray-600 mb-1">
                          {solution.routing_info?.route_taken} • {solution.processing_time?.toFixed(2)}s
                        </div>
                        <div className="font-medium text-gray-900 truncate">
                          {solution.solution?.substring(0, 100)}...
                        </div>
                      </div>
                      <button 
                        onClick={() => setCurrentSolution(solution)}
                        className="ml-4 px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                      >
                        View
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Analytics Dashboard placeholder component
const AnalyticsDashboard = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">Analytics Dashboard</h1>
          <p className="text-gray-600 mb-8">Detailed analytics and performance metrics</p>
          
          <div className="bg-white rounded-lg shadow-lg p-12">
            <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Coming Soon</h2>
            <p className="text-gray-600">
              Advanced analytics dashboard with performance metrics, usage patterns, and AI model performance analysis.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <Router>
      <div className="App">
        <Navigation />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/feedback" element={<FeedbackDashboard />} />
          <Route path="/analytics" element={<AnalyticsDashboard />} />
          <Route path="/health" element={<HealthMonitor />} />
        </Routes>
        <Toaster position="top-right" />
      </div>
    </Router>
  );
}

export default App;