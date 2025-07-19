# Main routing logic - Complete Implementation
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from datetime import datetime
import re
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from app.services.groq_service import groq_service
from app.services.knowledge_base import KnowledgeBase
from app.services.vector_db import VectorDatabase
from app.services.web_search import WebSearchService, MCPManager
from app.agents.math_solver import MathSolver
from app.agents.evaluation_agent import EvaluationAgent
from app.agents.feedback_agent import feedback_agent
from app.utils.embeddings import EmbeddingGenerator
from app.core.config import get_settings
from app.utils.helpers import MathUtils, TextProcessor, CacheUtils

logger = logging.getLogger(__name__)
settings = get_settings()

class RoutingAgent:
    """
    Complete routing agent that intelligently routes mathematical problems
    to the best solving strategy based on analysis and available resources.
    """
    
    def __init__(self):
        # Initialize core components
        self.knowledge_base = KnowledgeBase()
        self.vector_db = VectorDatabase()
        self.web_search = WebSearchService
        self.math_solver = MathSolver()
        self.evaluator = EvaluationAgent()
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize Qdrant client for vector search
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )
        
        # Routing strategies
        self.routing_strategies = {
            "knowledge_base": self._route_to_knowledge_base,
            "direct_solving": self._route_to_direct_solving,
            "web_search": self._route_to_web_search,
            "hybrid": self._route_hybrid_approach,
            "vector_search": self._route_to_vector_search
        }
        
        # Configuration parameters
        self.confidence_threshold = 0.7
        self.similarity_threshold = 0.8
        self.vector_search_threshold = 0.75
        self.max_similar_problems = 5
        
        # Caching for performance
        self.cache = CacheUtils(max_size=1000, ttl_seconds=1800)  # 30-minute cache
        
        # Performance tracking
        self.routing_stats = {
            "total_requests": 0,
            "strategy_usage": {},
            "success_rates": {},
            "avg_processing_times": {}
        }
        
        # Problem classification mapping
        self.problem_categories = {
            "algebra": {
                "keywords": ["solve", "equation", "variable", "linear", "quadratic", "polynomial", "x", "y"],
                "complexity_indicators": ["system", "simultaneous", "matrix", "determinant"],
                "difficulty_modifiers": {"easy": 0.8, "medium": 1.0, "hard": 1.2}
            },
            "calculus": {
                "keywords": ["derivative", "integral", "limit", "differential", "continuous", "function"],
                "complexity_indicators": ["partial", "multiple", "chain rule", "integration by parts"],
                "difficulty_modifiers": {"easy": 1.0, "medium": 1.3, "hard": 1.6}
            },
            "geometry": {
                "keywords": ["triangle", "circle", "area", "volume", "angle", "polygon", "coordinate"],
                "complexity_indicators": ["3d", "solid", "surface", "projection", "transformation"],
                "difficulty_modifiers": {"easy": 0.7, "medium": 1.0, "hard": 1.4}
            },
            "statistics": {
                "keywords": ["probability", "mean", "median", "variance", "distribution", "sample"],
                "complexity_indicators": ["hypothesis", "regression", "correlation", "bayesian"],
                "difficulty_modifiers": {"easy": 0.9, "medium": 1.1, "hard": 1.5}
            },
            "trigonometry": {
                "keywords": ["sin", "cos", "tan", "trigonometric", "angle", "triangle"],
                "complexity_indicators": ["inverse", "identities", "equations", "graphs"],
                "difficulty_modifiers": {"easy": 0.8, "medium": 1.0, "hard": 1.3}
            }
        }
    
    async def route_and_solve(self, question: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main routing and solving logic with comprehensive analysis and strategy selection
        """
        start_time = datetime.now()
        self.routing_stats["total_requests"] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(question, user_context)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached result")
            cached_result["from_cache"] = True
            return cached_result
        
        try:
            # Step 1: Comprehensive question analysis
            logger.info(f"Analyzing question: {question[:100]}...")
            question_analysis = await self._analyze_question_comprehensive(question)
            
            # Step 2: Check for immediate error conditions
            validation_result = self._validate_question(question)
            if not validation_result["is_valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "routing_info": {"route_taken": "validation_failed", "confidence": 0.0}
                }
            
            # Step 3: Determine optimal routing strategy
            routing_decision = await self._make_routing_decision_comprehensive(
                question, question_analysis, user_context
            )
            
            logger.info(f"Selected routing strategy: {routing_decision['strategy']} "
                       f"(confidence: {routing_decision['confidence']:.2f})")
            
            # Step 4: Execute the chosen strategy
            solution_result = await self._execute_strategy_with_fallback(
                question, 
                routing_decision["strategy"], 
                question_analysis, 
                user_context,
                routing_decision
            )
            
            # Step 5: Post-process and enhance the result
            enhanced_result = await self._enhance_solution_comprehensive(
                solution_result, question_analysis, routing_decision
            )
            
            # Step 6: Add comprehensive routing information
            processing_time = (datetime.now() - start_time).total_seconds()
            enhanced_result["routing_info"] = {
                "route_taken": routing_decision["strategy"],
                "confidence": routing_decision["confidence"],
                "reasoning": routing_decision["reasoning"],
                "question_analysis": question_analysis,
                "processing_time": processing_time,
                "alternative_strategies": routing_decision.get("alternatives", []),
                "fallback_used": routing_decision.get("fallback_used", False)
            }
            
            # Step 7: Update performance statistics
            self._update_routing_stats(routing_decision["strategy"], enhanced_result, processing_time)
            
            # Step 8: Cache successful results
            if enhanced_result.get("success"):
                self.cache.set(cache_key, enhanced_result)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in routing and solving: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Internal routing error: {str(e)}",
                "routing_info": {
                    "route_taken": "error",
                    "confidence": 0.0,
                    "reasoning": "An unexpected error occurred during routing"
                }
            }
    
    async def _analyze_question_comprehensive(self, question: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of the mathematical question using multiple approaches
        """
        try:
            # Basic text analysis
            basic_analysis = self._analyze_question_basic(question)
            
            # AI-powered analysis using Groq
            ai_analysis = await self._analyze_question_with_ai(question)
            
            # Mathematical content analysis
            math_analysis = self._analyze_mathematical_content(question)
            
            # Vector similarity analysis (if knowledge base available)
            vector_analysis = await self._analyze_question_similarity(question)
            
            # Combine all analyses
            comprehensive_analysis = {
                **basic_analysis,
                **ai_analysis,
                **math_analysis,
                "similarity_analysis": vector_analysis,
                "analysis_timestamp": datetime.now().isoformat(),
                "confidence_score": self._calculate_analysis_confidence(
                    basic_analysis, ai_analysis, math_analysis
                )
            }
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive question analysis: {e}")
            # Fallback to basic analysis
            return self._analyze_question_basic(question)
    
    def _analyze_question_basic(self, question: str) -> Dict[str, Any]:
        """Basic question analysis without external dependencies"""
        
        question_lower = question.lower()
        
        # Determine topic/category
        topic_scores = {}
        for category, info in self.problem_categories.items():
            score = sum(1 for keyword in info["keywords"] if keyword in question_lower)
            if score > 0:
                topic_scores[category] = score
        
        primary_topic = max(topic_scores.items(), key=lambda x: x[1])[0] if topic_scores else "general"
        
        # Estimate complexity
        complexity_score = 0
        for category, info in self.problem_categories.items():
            complexity_score += sum(1 for indicator in info["complexity_indicators"] 
                                  if indicator in question_lower)
        
        # Determine difficulty
        difficulty_indicators = {
            "easy": ["simple", "basic", "elementary", "find", "calculate"],
            "medium": ["solve", "determine", "evaluate", "analyze"],
            "hard": ["prove", "derive", "optimize", "complex", "advanced"]
        }
        
        difficulty_scores = {}
        for level, indicators in difficulty_indicators.items():
            score = sum(1 for indicator in indicators if indicator in question_lower)
            if score > 0:
                difficulty_scores[level] = score
        
        estimated_difficulty = max(difficulty_scores.items(), key=lambda x: x[1])[0] if difficulty_scores else "medium"
        
        # Extract mathematical expressions
        math_expressions = MathUtils.extract_mathematical_expressions(question)
        
        return {
            "topic": primary_topic,
            "difficulty": estimated_difficulty,
            "complexity": min(complexity_score, 10),  # Cap at 10
            "length": len(question),
            "word_count": len(question.split()),
            "has_equations": bool(math_expressions),
            "math_expressions": math_expressions,
            "requires_visualization": any(word in question_lower for word in 
                                        ["graph", "plot", "draw", "sketch", "diagram"]),
            "is_word_problem": any(phrase in question_lower for phrase in 
                                 ["if", "when", "how many", "how much", "person", "people"])
        }
    
    async def _analyze_question_with_ai(self, question: str) -> Dict[str, Any]:
        """AI-powered question analysis using Groq"""
        
        try:
            analysis_prompt = f"""
            Analyze this mathematical question comprehensively:
            
            Question: {question}
            
            Provide analysis in JSON format:
            {{
                "topic": "primary mathematical topic",
                "subtopics": ["list of subtopics"],
                "difficulty": "easy/medium/hard",
                "problem_type": "equation/calculation/proof/word_problem/etc",
                "concepts": ["list of mathematical concepts"],
                "solution_approach": "recommended approach",
                "estimated_steps": number_of_steps,
                "requires_special_methods": true/false,
                "complexity_rating": 1-10
            }}
            """
            
            response = await groq_service.client.chat.completions.create(
                model=groq_service.model,
                messages=[
                    {"role": "system", "content": "You are a mathematics education expert. Analyze questions and respond with valid JSON only."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                # Extract JSON from response (handle cases where AI adds extra text)
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    ai_analysis = json.loads(json_match.group())
                else:
                    ai_analysis = json.loads(analysis_text)
                
                return {
                    "ai_topic": ai_analysis.get("topic", "general"),
                    "ai_subtopics": ai_analysis.get("subtopics", []),
                    "ai_difficulty": ai_analysis.get("difficulty", "medium"),
                    "ai_problem_type": ai_analysis.get("problem_type", "general"),
                    "ai_concepts": ai_analysis.get("concepts", []),
                    "ai_solution_approach": ai_analysis.get("solution_approach", "standard"),
                    "ai_estimated_steps": ai_analysis.get("estimated_steps", 3),
                    "ai_requires_special_methods": ai_analysis.get("requires_special_methods", False),
                    "ai_complexity_rating": ai_analysis.get("complexity_rating", 5)
                }
                
            except json.JSONDecodeError:
                logger.warning("AI analysis response was not valid JSON, parsing manually")
                return self._parse_ai_analysis_manually(analysis_text)
                
        except Exception as e:
            logger.error(f"Error in AI question analysis: {e}")
            return {
                "ai_topic": "general",
                "ai_difficulty": "medium",
                "ai_problem_type": "general",
                "ai_concepts": [],
                "ai_solution_approach": "standard",
                "ai_estimated_steps": 3,
                "ai_requires_special_methods": False,
                "ai_complexity_rating": 5
            }
    
    def _parse_ai_analysis_manually(self, analysis_text: str) -> Dict[str, Any]:
        """Manually parse AI analysis when JSON parsing fails"""
        
        analysis = {
            "ai_topic": "general",
            "ai_difficulty": "medium",
            "ai_problem_type": "general",
            "ai_concepts": [],
            "ai_solution_approach": "standard",
            "ai_estimated_steps": 3,
            "ai_requires_special_methods": False,
            "ai_complexity_rating": 5
        }
        
        # Extract information using regex patterns
        patterns = {
            "ai_topic": r'topic["\s:]+([^",\n]+)',
            "ai_difficulty": r'difficulty["\s:]+([^",\n]+)',
            "ai_problem_type": r'problem_type["\s:]+([^",\n]+)',
            "ai_solution_approach": r'solution_approach["\s:]+([^",\n]+)',
            "ai_estimated_steps": r'estimated_steps["\s:]+(\d+)',
            "ai_complexity_rating": r'complexity_rating["\s:]+(\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip().strip('"')
                if key in ["ai_estimated_steps", "ai_complexity_rating"]:
                    try:
                        analysis[key] = int(value)
                    except ValueError:
                        pass
                else:
                    analysis[key] = value
        
        return analysis
    
    def _analyze_mathematical_content(self, question: str) -> Dict[str, Any]:
        """Analyze mathematical content and complexity"""
        
        # Count mathematical operators and symbols
        operators = re.findall(r'[+\-*/=<>≤≥≠∫∑∏√∂]', question)
        variables = re.findall(r'\b[a-zA-Z]\b', question)
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)
        
        # Detect special mathematical functions
        functions = re.findall(r'\b(?:sin|cos|tan|log|ln|exp|sqrt|abs|max|min)\b', question.lower())
        
        # Detect LaTeX or mathematical notation
        latex_patterns = re.findall(r'\\[a-zA-Z]+|\$[^$]*\$', question)
        
        return {
            "operator_count": len(operators),
            "variable_count": len(set(variables)),
            "number_count": len(numbers),
            "function_count": len(functions),
            "has_latex": bool(latex_patterns),
            "mathematical_density": len(operators + functions + latex_patterns) / max(len(question.split()), 1),
            "detected_functions": list(set(functions)),
            "complexity_indicators": {
                "multiple_variables": len(set(variables)) > 2,
                "complex_operations": len(functions) > 1,
                "advanced_notation": bool(latex_patterns)
            }
        }
    
    async def _analyze_question_similarity(self, question: str) -> Dict[str, Any]:
        """Analyze question similarity using vector search"""
        
        try:
            # Generate embedding for the question
            question_embedding = self.embedding_generator.generate_embedding(question)
            
            if not question_embedding:
                return {"similar_problems": [], "max_similarity": 0.0}
            
            # Search for similar problems in vector database
            similar_results = await self.vector_db.search_similar(
                query_vector=question_embedding,
                limit=self.max_similar_problems,
                score_threshold=0.5
            )
            
            if similar_results:
                max_similarity = max(result["score"] for result in similar_results)
                return {
                    "similar_problems": similar_results,
                    "max_similarity": max_similarity,
                    "has_high_similarity": max_similarity > self.similarity_threshold
                }
            
            return {"similar_problems": [], "max_similarity": 0.0, "has_high_similarity": False}
            
        except Exception as e:
            logger.error(f"Error in similarity analysis: {e}")
            return {"similar_problems": [], "max_similarity": 0.0, "has_high_similarity": False}
    
    def _calculate_analysis_confidence(self, basic: Dict, ai: Dict, math: Dict) -> float:
        """Calculate confidence score for the analysis"""
        
        confidence_factors = []
        
        # Topic consistency between basic and AI analysis
        if basic.get("topic") == ai.get("ai_topic"):
            confidence_factors.append(0.3)
        
        # Difficulty consistency
        if basic.get("difficulty") == ai.get("ai_difficulty"):
            confidence_factors.append(0.2)
        
        # Mathematical content richness
        if math.get("mathematical_density", 0) > 0.1:
            confidence_factors.append(0.2)
        
        # Problem complexity indicators
        if ai.get("ai_complexity_rating", 5) <= 7:
            confidence_factors.append(0.1)
        
        # Basic analysis completeness
        if basic.get("has_equations") and basic.get("word_count", 0) > 5:
            confidence_factors.append(0.2)
        
        return sum(confidence_factors)
    
    async def _make_routing_decision_comprehensive(self, question: str, analysis: Dict[str, Any], 
                                                 user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make comprehensive routing decision based on multiple factors
        """
        
        # Calculate scores for each strategy
        strategy_scores = {}
        
        # Vector Search Strategy
        vector_score = await self._score_vector_search_strategy(question, analysis)
        strategy_scores["vector_search"] = vector_score
        
        # Knowledge Base Strategy
        kb_score = await self._score_knowledge_base_strategy(question, analysis)
        strategy_scores["knowledge_base"] = kb_score
        
        # Direct Solving Strategy
        direct_score = self._score_direct_solving_strategy(analysis)
        strategy_scores["direct_solving"] = direct_score
        
        # Web Search Strategy
        web_score = self._score_web_search_strategy(analysis)
        strategy_scores["web_search"] = web_score
        
        # Hybrid Strategy
        hybrid_score = self._score_hybrid_strategy(analysis, strategy_scores)
        strategy_scores["hybrid"] = hybrid_score
        
        # Apply user context modifiers
        if user_context:
            strategy_scores = self._apply_user_context_modifiers(strategy_scores, user_context)
        
        # Select best strategy with fallback options
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        best_strategy = sorted_strategies[0]
        
        # Prepare alternative strategies
        alternatives = [
            {"strategy": strategy, "score": info["score"], "reasoning": info["reasoning"]}
            for strategy, info in sorted_strategies[1:3]  # Top 2 alternatives
        ]
        
        return {
            "strategy": best_strategy[0],
            "confidence": best_strategy[1]["score"],
            "reasoning": best_strategy[1]["reasoning"],
            "all_scores": strategy_scores,
            "alternatives": alternatives
        }
    
    async def _score_vector_search_strategy(self, question: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Score the vector search strategy"""
        
        similarity_data = analysis.get("similarity_analysis", {})
        max_similarity = similarity_data.get("max_similarity", 0.0)
        similar_problems = similarity_data.get("similar_problems", [])
        
        if max_similarity > self.vector_search_threshold:
            score = 0.95
            reasoning = f"High similarity found ({max_similarity:.2f}) with {len(similar_problems)} problems"
        elif max_similarity > 0.6:
            score = 0.75
            reasoning = f"Good similarity found ({max_similarity:.2f}) with existing problems"
        elif len(similar_problems) > 0:
            score = 0.5
            reasoning = f"Some similar problems found but low similarity ({max_similarity:.2f})"
        else:
            score = 0.1
            reasoning = "No similar problems found in vector database"
        
        return {
            "score": score,
            "reasoning": reasoning,
            "similar_problems": similar_problems
        }
    
    async def _score_knowledge_base_strategy(self, question: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Score the knowledge base strategy"""
        
        # Search for similar problems in knowledge base
        topic = analysis.get("ai_topic", analysis.get("topic", "general"))
        difficulty = analysis.get("ai_difficulty", analysis.get("difficulty", "medium"))
        
        similar_problems = self.knowledge_base.search_problems(
            question, 
            category=topic if topic != "general" else None,
            difficulty=difficulty,
            limit=3
        )
        
        if similar_problems:
            best_relevance = max(p.get("relevance_score", 0) for p in similar_problems)
            
            if best_relevance > self.similarity_threshold:
                score = 0.9
                reasoning = f"High relevance ({best_relevance:.2f}) found in knowledge base"
            elif best_relevance > 0.5:
                score = 0.7
                reasoning = f"Good relevance ({best_relevance:.2f}) found in knowledge base"
            else:
                score = 0.4
                reasoning = f"Low relevance ({best_relevance:.2f}) in knowledge base"
        else:
            score = 0.2
            reasoning = "No similar problems found in knowledge base"
        
        return {
            "score": score,
            "reasoning": reasoning,
            "similar_problems": similar_problems
        }
    
    def _score_direct_solving_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Score the direct solving strategy"""
        
        complexity = analysis.get("ai_complexity_rating", analysis.get("complexity", 5))
        difficulty = analysis.get("ai_difficulty", analysis.get("difficulty", "medium"))
        requires_special = analysis.get("ai_requires_special_methods", False)
        
        # Base score calculation
        if complexity <= 5 and difficulty in ["easy", "medium"] and not requires_special:
            score = 0.85
            reasoning = "Standard problem suitable for direct AI solving"
        elif complexity <= 7 and not requires_special:
            score = 0.7
            reasoning = "Moderately complex problem, AI can handle well"
        elif complexity <= 8:
            score = 0.5
            reasoning = "Complex problem may need additional context"
        else:
            score = 0.3
            reasoning = "Very complex problem may require specialized approach"
        
        # Adjust based on problem type
        problem_type = analysis.get("ai_problem_type", "general")
        if problem_type in ["equation", "calculation"]:
            score += 0.1
        elif problem_type in ["proof", "optimization"]:
            score -= 0.1
        
        return {
            "score": min(score, 1.0),
            "reasoning": reasoning
        }
    
    def _score_web_search_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Score the web search strategy"""
        
        requires_special = analysis.get("ai_requires_special_methods", False)
        complexity = analysis.get("ai_complexity_rating", analysis.get("complexity", 5))
        topic = analysis.get("ai_topic", analysis.get("topic", "general"))
        
        # Topics that benefit from web search
        specialized_topics = [
            "topology", "abstract_algebra", "real_analysis", "number_theory",
            "differential_equations", "complex_analysis", "mathematical_logic"
        ]
        
        if requires_special or topic in specialized_topics:
            score = 0.8
            reasoning = "Specialized topic benefits from web resources"
        elif complexity >= 8:
            score = 0.6
            reasoning = "High complexity may benefit from additional examples"
        elif analysis.get("is_word_problem", False):
            score = 0.4
            reasoning = "Word problems may benefit from similar examples"
        else:
            score = 0.2
            reasoning = "Standard problem doesn't require web search"
        
        return {
            "score": score,
            "reasoning": reasoning
        }
    
    def _score_hybrid_strategy(self, analysis: Dict[str, Any], other_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Score the hybrid strategy"""
        
        scores = [info["score"] for info in other_scores.values()]
        max_score = max(scores) if scores else 0
        score_variance = np.var(scores) if len(scores) > 1 else 0
        
        # Hybrid is good when:
        # 1. No single strategy dominates
        # 2. Multiple strategies have reasonable scores
        # 3. Problem is complex enough to benefit from multiple approaches
        
        complexity = analysis.get("ai_complexity_rating", analysis.get("complexity", 5))
        
        if max_score < 0.7 and score_variance < 0.1 and complexity >= 6:
            score = 0.85
            reasoning = "Multiple strategies viable, hybrid approach recommended"
        elif max_score < 0.8 and complexity >= 7:
            score = 0.7
            reasoning = "Complex problem benefits from multiple approaches"
        elif score_variance > 0.2:
            score = 0.6
            reasoning = "Diverse strategy scores suggest hybrid approach"
        else:
            score = 0.3
            reasoning = "Single strategy approach preferred"
        
        return {
            "score": score,
            "reasoning": reasoning
        }
    
    def _apply_user_context_modifiers(self, strategy_scores: Dict[str, Any], 
                                    user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user context to modify strategy scores"""
        
        # User preference for detailed explanations
        if user_context.get("prefers_detailed_explanation"):
            strategy_scores["hybrid"]["score"] += 0.1
            strategy_scores["knowledge_base"]["score"] += 0.05
        
        # User's mathematical background
        user_level = user_context.get("mathematical_level", "intermediate")
        if user_level == "beginner":
            strategy_scores["knowledge_base"]["score"] += 0.1
            strategy_scores["web_search"]["score"] += 0.05
        elif user_level == "advanced":
            strategy_scores["direct_solving"]["score"] += 0.1
        
        # Previous interaction history
        if user_context.get("previous_successful_strategy"):
            prev_strategy = user_context["previous_successful_strategy"]
            if prev_strategy in strategy_scores:
                strategy_scores[prev_strategy]["score"] += 0.05
        
        # Ensure scores don't exceed 1.0
        for strategy in strategy_scores:
            strategy_scores[strategy]["score"] = min(strategy_scores[strategy]["score"], 1.0)
        
        return strategy_scores
    
    async def _execute_strategy_with_fallback(self, question: str, strategy: str, analysis: Dict[str, Any], 
                                            user_context: Optional[Dict[str, Any]] = None,
                                            routing_decision: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute strategy with intelligent fallback"""
        
        primary_result = None
        fallback_used = False
        
        try:
            # Execute primary strategy
            if strategy in self.routing_strategies:
                primary_result = await self.routing_strategies[strategy](
                    question, analysis, user_context
                )
            else:
                # Fallback to direct solving for unknown strategies
                primary_result = await self._route_to_direct_solving(question, analysis, user_context)
                fallback_used = True
            
            # Check if primary strategy succeeded
            if primary_result and primary_result.get("success"):
                if fallback_used:
                    primary_result["fallback_used"] = True
                return primary_result
            
        except Exception as e:
            logger.error(f"Primary strategy {strategy} failed: {e}")
        
        # Implement intelligent fallback chain
        fallback_chain = self._get_fallback_chain(strategy, routing_decision)
        
        for fallback_strategy in fallback_chain:
            try:
                logger.info(f"Trying fallback strategy: {fallback_strategy}")
                
                fallback_result = await self.routing_strategies[fallback_strategy](
                    question, analysis, user_context
                )
                if fallback_result and fallback_result.get("success"):
                    fallback_result["fallback_used"] = True
                    fallback_result["original_strategy"] = strategy
                    return fallback_result
                
            except Exception as e:
                logger.error(f"Fallback strategy {fallback_strategy} failed: {e}")
                continue
        
        # If all fallbacks fail, try direct solving as last resort
        last_resort_result = await self._route_to_direct_solving(question, analysis, user_context)
        last_resort_result["fallback_used"] = True
        last_resort_result["original_strategy"] = strategy
        return last_resort_result
    
    def _get_fallback_chain(self, strategy: str, routing_decision: Dict[str, Any]) -> List[str]:
        """Get ordered fallback chain based on strategy and routing decision"""
        
        # Default fallback chains for each strategy
        fallback_chains = {
            "vector_search": ["knowledge_base", "hybrid", "direct_solving"],
            "knowledge_base": ["vector_search", "hybrid", "direct_solving"],
            "direct_solving": ["hybrid", "knowledge_base", "web_search"],
            "web_search": ["hybrid", "knowledge_base", "direct_solving"],
            "hybrid": ["knowledge_base", "direct_solving", "web_search"]
        }
        
        # Get default chain for the strategy
        chain = fallback_chains.get(strategy, ["direct_solving"])
        
        # If we have routing decision with alternatives, use those first
        if routing_decision and routing_decision.get("alternatives"):
            alt_strategies = [alt["strategy"] for alt in routing_decision["alternatives"]]
            chain = alt_strategies + [s for s in chain if s not in alt_strategies]
        
        return chain
    
    async def _route_to_knowledge_base(self, question: str, analysis: Dict[str, Any], 
                                     user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route question to knowledge base for solution"""
        
        try:
            topic = analysis.get("ai_topic", analysis.get("topic", "general"))
            difficulty = analysis.get("ai_difficulty", analysis.get("difficulty", "medium"))
            
            # Search for similar problems
            similar_problems = self.knowledge_base.search_problems(
                question, 
                category=topic if topic != "general" else None,
                difficulty=difficulty,
                limit=3
            )
            
            if not similar_problems:
                return {
                    "success": False,
                    "error": "No matching problems found in knowledge base",
                    "strategy": "knowledge_base"
                }
            
            # Get the most relevant problem
            best_match = max(similar_problems, key=lambda x: x.get("relevance_score", 0))
            
            # Enhance with additional context if needed
            enhanced_solution = await self._enhance_knowledge_base_solution(
                best_match, question, analysis
            )
            
            return {
                "success": True,
                "solution": enhanced_solution,
                "source": "knowledge_base",
                "similar_problems": similar_problems,
                "confidence": best_match.get("relevance_score", 0.7),
                "strategy": "knowledge_base"
            }
            
        except Exception as e:
            logger.error(f"Error routing to knowledge base: {e}")
            return {
                "success": False,
                "error": f"Knowledge base error: {str(e)}",
                "strategy": "knowledge_base"
            }
    
    async def _enhance_knowledge_base_solution(self, solution: Dict[str, Any], 
                                              question: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance knowledge base solution with additional context"""
        
        enhanced = solution.copy()
        
        # Add question-specific adaptation if needed
        if not TextProcessor.is_similar(question, solution.get("question", "")):
            adaptation_prompt = f"""
            The following solution was found for a similar problem:
            
            Original Problem: {solution.get('question')}
            Solution: {solution.get('solution')}
            
            Please adapt this solution to answer the following question:
            {question}
            
            Provide the adapted solution in clear steps.
            """
            
            try:
                adapted_response = await groq_service.client.chat.completions.create(
                    model=groq_service.model,
                    messages=[
                        {"role": "system", "content": "You are a mathematics tutor. Adapt solutions to similar problems."},
                        {"role": "user", "content": adaptation_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )
                
                enhanced["adapted_solution"] = adapted_response.choices[0].message.content.strip()
                enhanced["is_adapted"] = True
                
            except Exception as e:
                logger.error(f"Failed to adapt knowledge base solution: {e}")
                enhanced["adapted_solution"] = "Unable to adapt solution automatically. Here's the original solution for a similar problem:\n\n" + solution.get("solution", "")
        
        return enhanced
    
    async def _route_to_direct_solving(self, question: str, analysis: Dict[str, Any], 
                                     user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route question to direct AI solving"""
        
        try:
            # Prepare the prompt with context
            prompt = self._build_direct_solving_prompt(question, analysis, user_context)
            
            # Get solution from Groq
            response = await groq_service.client.chat.completions.create(
                model=groq_service.model,
                messages=[
                    {"role": "system", "content": "You are an expert mathematics tutor. Solve problems step-by-step with clear explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            solution = response.choices[0].message.content.strip()
            
            # Evaluate solution quality
            evaluation = await self.evaluator.evaluate_solution(
                question=question,
                solution=solution,
                context=analysis
            )
            
            return {
                "success": evaluation["is_correct"],
                "solution": solution,
                "source": "direct_ai",
                "confidence": evaluation["confidence"],
                "evaluation": evaluation,
                "strategy": "direct_solving"
            }
            
        except Exception as e:
            logger.error(f"Error in direct solving: {e}")
            return {
                "success": False,
                "error": f"Direct solving error: {str(e)}",
                "strategy": "direct_solving"
            }
    
    def _build_direct_solving_prompt(self, question: str, analysis: Dict[str, Any], 
                                   user_context: Optional[Dict[str, Any]] = None) -> str:
        """Build comprehensive prompt for direct solving"""
        
        prompt_parts = [
            "Solve the following mathematical problem step by step with clear explanations:",
            f"Problem: {question}",
            "\nAdditional context:",
            f"- Topic: {analysis.get('ai_topic', analysis.get('topic', 'general'))}",
            f"- Difficulty: {analysis.get('ai_difficulty', analysis.get('difficulty', 'medium'))}",
            f"- Problem type: {analysis.get('ai_problem_type', 'general')}",
            f"- Key concepts: {', '.join(analysis.get('ai_concepts', [])) if analysis.get('ai_concepts') else 'Not specified'}"
        ]
        
        # Add user context if available
        if user_context:
            if user_context.get("mathematical_level"):
                prompt_parts.append(f"- User's math level: {user_context['mathematical_level']}")
            if user_context.get("learning_goals"):
                prompt_parts.append(f"- User's goals: {user_context['learning_goals']}")
        
        # Add specific instructions based on problem type
        problem_type = analysis.get("ai_problem_type", "general")
        if problem_type == "proof":
            prompt_parts.append("\nProvide a rigorous proof with clear logical steps.")
        elif problem_type == "word_problem":
            prompt_parts.append("\nFirst explain how to translate the word problem into mathematical expressions, then solve.")
        elif problem_type == "calculation":
            prompt_parts.append("\nShow all intermediate calculation steps.")
        
        # Final instructions
        prompt_parts.append("\nStructure your answer with:\n1. Problem understanding\n2. Solution approach\n3. Step-by-step solution\n4. Final answer\n5. Verification (if applicable)")
        
        return "\n".join(prompt_parts)
    
    async def _route_to_web_search(self, question: str, analysis: Dict[str, Any], 
                                 user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route question to web search for solution"""
        
        try:
            # First try MCP (Math Problem Corpus) search
            mcp_results = await mcp_manager.search_math_problems(question)
            
            if mcp_results and mcp_results.get("success"):
                best_match = mcp_results["results"][0]
                return {
                    "success": True,
                    "solution": best_match["solution"],
                    "source": "mcp",
                    "confidence": best_match.get("similarity_score", 0.8),
                    "references": [best_match["source"]],
                    "strategy": "web_search"
                }
            
            # Fallback to general web search
            search_results = await web_search_service.search(question, num_results=3)
            
            if not search_results:
                return {
                    "success": False,
                    "error": "No relevant web results found",
                    "strategy": "web_search"
                }
            
            # Process and synthesize web results
            synthesized_solution = await self._synthesize_web_results(
                question, search_results, analysis
            )
            
            return {
                "success": True,
                "solution": synthesized_solution,
                "source": "web_search",
                "confidence": 0.7,  # Moderate confidence for synthesized results
                "references": [r["url"] for r in search_results],
                "strategy": "web_search"
            }
            
        except Exception as e:
            logger.error(f"Error in web search routing: {e}")
            return {
                "success": False,
                "error": f"Web search error: {str(e)}",
                "strategy": "web_search"
            }
    
    async def _synthesize_web_results(self, question: str, search_results: List[Dict[str, Any]], 
                                    analysis: Dict[str, Any]) -> str:
        """Synthesize information from multiple web results"""
        
        # Prepare context for synthesis
        context_parts = [
            f"Original question: {question}",
            "\nFound the following relevant resources:"
        ]
        
        for i, result in enumerate(search_results[:3]):  # Use top 3 results
            context_parts.append(
                f"\nSource {i+1} ({result.get('url', 'unknown')}):\n"
                f"{TextProcessor.summarize(result.get('content', ''), 300)}"
            )
        
        context_parts.append(
            "\nBased on these resources, provide a comprehensive solution "
            "to the original question, combining the most relevant information "
            "from these sources. Cite sources where appropriate."
        )
        
        synthesis_prompt = "\n".join(context_parts)
        
        # Get synthesized solution
        response = await groq_service.client.chat.completions.create(
            model=groq_service.model,
            messages=[
                {"role": "system", "content": "You are a research assistant. Synthesize information from multiple sources to answer math questions."},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.4,
            max_tokens=1200
        )
        
        return response.choices[0].message.content.strip()
    
    async def _route_to_vector_search(self, question: str, analysis: Dict[str, Any], 
                                    user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route question to vector similarity search"""
        
        try:
            # Get embedding for the question
            question_embedding = self.embedding_generator.generate_embedding(question)
            
            if not question_embedding:
                return {
                    "success": False,
                    "error": "Failed to generate question embedding",
                    "strategy": "vector_search"
                }
            
            # Search for similar problems
            similar_results = await self.vector_db.search_similar(
                query_vector=question_embedding,
                limit=3,
                score_threshold=0.6
            )
            
            if not similar_results:
                return {
                    "success": False,
                    "error": "No similar problems found in vector database",
                    "strategy": "vector_search"
                }
            
            # Get the most similar problem
            best_match = max(similar_results, key=lambda x: x["score"])
            
            # Adapt the solution if needed
            adapted_solution = await self._adapt_vector_solution(
                question, best_match["solution"], analysis
            )
            
            return {
                "success": True,
                "solution": adapted_solution,
                "source": "vector_db",
                "original_problem": best_match["question"],
                "confidence": best_match["score"],
                "similar_problems": similar_results,
                "strategy": "vector_search"
            }
            
        except Exception as e:
            logger.error(f"Error in vector search routing: {e}")
            return {
                "success": False,
                "error": f"Vector search error: {str(e)}",
                "strategy": "vector_search"
            }
    
    async def _adapt_vector_solution(self, question: str, solution: str, 
                                   analysis: Dict[str, Any]) -> str:
        """Adapt a solution from a similar vector-matched problem"""
        
        # Check if direct solution is sufficient
        if TextProcessor.is_similar(question, solution, threshold=0.7):
            return solution
        
        # Otherwise adapt the solution
        adaptation_prompt = f"""
        The following solution was found for a similar problem:
        
        Similar Problem: {solution['question']}
        Solution: {solution['solution']}
        
        Please adapt this solution to answer the following question:
        {question}
        
        Provide the adapted solution with clear explanations of any changes needed.
        """
        
        response = await groq_service.client.chat.completions.create(
            model=groq_service.model,
            messages=[
                {"role": "system", "content": "You adapt mathematical solutions to similar problems."},
                {"role": "user", "content": adaptation_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
    
    async def _route_hybrid_approach(self, question: str, analysis: Dict[str, Any], 
                                    user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use a hybrid approach combining multiple strategies"""
        
        try:
            # Execute multiple strategies in parallel
            tasks = [
                self._route_to_knowledge_base(question, analysis, user_context),
                self._route_to_direct_solving(question, analysis, user_context),
                self._route_to_vector_search(question, analysis, user_context)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in hybrid strategy component: {result}")
                    continue
                if result.get("success"):
                    successful_results.append(result)
            
            if not successful_results:
                return {
                    "success": False,
                    "error": "All hybrid components failed",
                    "strategy": "hybrid"
                }
            
            # Select best result based on confidence
            best_result = max(successful_results, key=lambda x: x.get("confidence", 0))
            
            # Enhance with additional context from other results
            enhanced_solution = await self._enhance_hybrid_solution(
                question, best_result, successful_results, analysis
            )
            
            return {
                "success": True,
                "solution": enhanced_solution,
                "source": "hybrid",
                "component_results": [r["strategy"] for r in successful_results],
                "confidence": best_result.get("confidence", 0.8) * 0.9,  # Slightly reduce confidence for hybrid
                "strategy": "hybrid"
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid routing: {e}")
            return {
                "success": False,
                "error": f"Hybrid approach error: {str(e)}",
                "strategy": "hybrid"
            }
    
    async def _enhance_hybrid_solution(self, question: str, primary_result: Dict[str, Any], 
                                     all_results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """Enhance hybrid solution with context from multiple sources"""
        
        # Prepare comparison of different approaches
        comparison_prompt = f"""
        Original Question: {question}
        
        Different approaches were used to solve this problem:
        
        {self._format_hybrid_comparison(all_results)}
        
        Please synthesize the best solution by combining the strengths of each approach.
        Focus on:
        - Correctness and completeness
        - Clarity of explanation
        - Appropriate level of detail
        - Mathematical rigor
        
        Provide your synthesized solution with clear attribution to the sources.
        """
        
        response = await groq_service.client.chat.completions.create(
            model=groq_service.model,
            messages=[
                {"role": "system", "content": "You synthesize mathematical solutions from multiple approaches."},
                {"role": "user", "content": comparison_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        return response.choices[0].message.content.strip()
    
    def _format_hybrid_comparison(self, results: List[Dict[str, Any]]) -> str:
        """Format comparison of hybrid approach results"""
        
        comparison = []
        for i, result in enumerate(results, 1):
            comparison.append(
                f"\nApproach {i} ({result['strategy']}, confidence: {result.get('confidence', 0):.2f}):\n"
                f"{TextProcessor.summarize(result['solution'], 300)}"
            )
        
        return "\n".join(comparison)
    
    async def _enhance_solution_comprehensive(self, solution_result: Dict[str, Any], 
                                           question_analysis: Dict[str, Any],
                                           routing_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the solution with additional information and verification"""
        
        if not solution_result.get("success"):
            return solution_result
        
        enhanced_result = solution_result.copy()
        solution = enhanced_result["solution"]
        
        # Add verification step
        if question_analysis.get("has_equations"):
            verification = await self._verify_solution(
                solution, 
                question_analysis.get("math_expressions", []),
                routing_decision["strategy"]
            )
            enhanced_result["verification"] = verification
        
        # Add learning resources if needed
        if routing_decision["strategy"] != "knowledge_base":
            learning_resources = self.knowledge_base.get_learning_resources(
                topic=question_analysis.get("ai_topic", question_analysis.get("topic")),
                difficulty=question_analysis.get("ai_difficulty", question_analysis.get("difficulty"))
            )
            if learning_resources:
                enhanced_result["learning_resources"] = learning_resources
        
        # Format solution based on problem type
        problem_type = question_analysis.get("ai_problem_type", "general")
        if problem_type == "word_problem":
            formatted_solution = await self._format_word_problem_solution(solution)
            enhanced_result["formatted_solution"] = formatted_solution
        elif problem_type == "proof":
            formatted_solution = await self._format_proof_solution(solution)
            enhanced_result["formatted_solution"] = formatted_solution
        
        # Add feedback mechanism
        feedback_prompt = feedback_agent.generate_feedback_prompt(
            question=None,  # We don't have the original question here
            solution=solution,
            context=question_analysis
        )
        enhanced_result["feedback_prompt"] = feedback_prompt
        
        return enhanced_result
    
    async def _verify_solution(self, solution: str, math_expressions: List[str], 
                             strategy: str) -> Dict[str, Any]:
        """Verify the correctness of the solution"""
        
        verification_prompt = f"""
        Verify the correctness of this mathematical solution:
        
        Solution:
        {solution}
        
        Pay special attention to these key expressions:
        {', '.join(math_expressions) if math_expressions else 'None identified'}
        
        Provide verification in this format:
        - Correctness: [True/False/Partial]
        - Errors found: [list or None]
        - Confidence: [0-1]
        - Suggestions: [if any]
        """
        
        try:
            response = await groq_service.client.chat.completions.create(
                model=groq_service.model,
                messages=[
                    {"role": "system", "content": "You verify mathematical solutions for correctness."},
                    {"role": "user", "content": verification_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            verification_text = response.choices[0].message.content.strip()
            return self._parse_verification_response(verification_text)
            
        except Exception as e:
            logger.error(f"Error in solution verification: {e}")
            return {
                "correctness": "unknown",
                "errors": ["Verification failed"],
                "confidence": 0.0,
                "suggestions": []
            }
    
    def _parse_verification_response(self, verification_text: str) -> Dict[str, Any]:
        """Parse the verification response into structured data"""
        
        # Default result
        result = {
            "correctness": "unknown",
            "errors": [],
            "confidence": 0.0,
            "suggestions": []
        }
        
        # Extract correctness
        correctness_match = re.search(r'Correctness:\s*([^\n]+)', verification_text, re.IGNORECASE)
        if correctness_match:
            result["correctness"] = correctness_match.group(1).strip().lower()
        
        # Extract errors
        errors_match = re.search(r'Errors found:\s*([^\n]+)', verification_text, re.IGNORECASE)
        if errors_match:
            errors = errors_match.group(1).strip()
            if errors.lower() not in ["none", "null"]:
                result["errors"] = [e.strip() for e in errors.split(",")]
        
        # Extract confidence
        confidence_match = re.search(r'Confidence:\s*([0-9.]+)', verification_text, re.IGNORECASE)
        if confidence_match:
            try:
                result["confidence"] = float(confidence_match.group(1))
            except ValueError:
                pass
        
        # Extract suggestions
        suggestions_match = re.search(r'Suggestions:\s*([^\n]+)', verification_text, re.IGNORECASE)
        if suggestions_match:
            suggestions = suggestions_match.group(1).strip()
            if suggestions.lower() not in ["none", "null"]:
                result["suggestions"] = [s.strip() for s in suggestions.split(";")]
        
        return result
    
    async def _format_word_problem_solution(self, solution: str) -> str:
        """Format solution for word problems"""
        
        formatting_prompt = f"""
        The following is a solution to a math word problem:
        
        {solution}
        
        Please reformat it to clearly show:
        1. Problem understanding and translation to math
        2. Variables and relationships
        3. Solution steps
        4. Final answer with units (if applicable)
        5. Verification (if possible)
        
        Use clear section headings and bullet points where appropriate.
        """
        
        response = await groq_service.client.chat.completions.create(
            model=groq_service.model,
            messages=[
                {"role": "system", "content": "You format math word problem solutions clearly."},
                {"role": "user", "content": formatting_prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        return response.choices[0].message.content.strip()
    
    async def _format_proof_solution(self, solution: str) -> str:
        """Format solution for proofs"""
        
        formatting_prompt = f"""
        The following is a mathematical proof:
        
        {solution}
        
        Please reformat it to clearly show:
        1. Given information
        2. Proof approach/method
        3. Logical steps with justifications
        4. Conclusion
        
        Use proper mathematical notation and clearly mark assumptions, deductions, etc.
        """
        
        response = await groq_service.client.chat.completions.create(
            model=groq_service.model,
            messages=[
                {"role": "system", "content": "You format mathematical proofs clearly."},
                {"role": "user", "content": formatting_prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        return response.choices[0].message.content.strip()
    
    def _validate_question(self, question: str) -> Dict[str, Any]:
        """Validate the question before processing"""
        
        if not question or len(question.strip()) < 5:
            return {
                "is_valid": False,
                "error": "Question is too short or empty"
            }
        
        # Check for offensive content
        offensive_terms = ["fuck", "shit", "asshole", "bitch", "retard"]  # Add more as needed
        question_lower = question.lower()
        if any(term in question_lower for term in offensive_terms):
            return {
                "is_valid": False,
                "error": "Question contains inappropriate content"
            }
        
        # Check for mathematical content
        math_expressions = MathUtils.extract_mathematical_expressions(question)
        math_keywords = ["solve", "calculate", "find", "prove", "equation", "function"]
        
        if not math_expressions and not any(keyword in question_lower for keyword in math_keywords):
            return {
                "is_valid": False,
                "error": "Question doesn't appear to be mathematical"
            }
        
        return {
            "is_valid": True,
            "error": None
        }
    
    def _generate_cache_key(self, question: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a consistent cache key for the question"""
        
        base_key = TextProcessor.normalize_text(question)
        
        if user_context:
            # Include relevant context elements in cache key
            context_elements = [
                user_context.get("mathematical_level", ""),
                user_context.get("preferred_language", "en"),
                str(user_context.get("user_id", ""))
            ]
            base_key += "_" + "_".join(context_elements)
        
        return CacheUtils.generate_hash(base_key)
    
    def _update_routing_stats(self, strategy: str, result: Dict[str, Any], processing_time: float):
        """Update routing performance statistics"""
        
        # Update strategy usage
        if strategy not in self.routing_stats["strategy_usage"]:
            self.routing_stats["strategy_usage"][strategy] = 0
        self.routing_stats["strategy_usage"][strategy] += 1
        
        # Update success rates
        if strategy not in self.routing_stats["success_rates"]:
            self.routing_stats["success_rates"][strategy] = {"success": 0, "total": 0}
        
        self.routing_stats["success_rates"][strategy]["total"] += 1
        if result.get("success"):
            self.routing_stats["success_rates"][strategy]["success"] += 1
        
        # Update processing times
        if strategy not in self.routing_stats["avg_processing_times"]:
            self.routing_stats["avg_processing_times"][strategy] = []
        
        self.routing_stats["avg_processing_times"][strategy].append(processing_time)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get current routing statistics with computed metrics"""
        
        stats = self.routing_stats.copy()
        
        # Compute success percentages
        for strategy, counts in stats["success_rates"].items():
            if counts["total"] > 0:
                stats["success_rates"][strategy]["percentage"] = (
                    counts["success"] / counts["total"] * 100
                )
        
        # Compute average processing times
        for strategy, times in stats["avg_processing_times"].items():
            if times:
                stats["avg_processing_times"][strategy] = sum(times) / len(times)
        
        return stats                