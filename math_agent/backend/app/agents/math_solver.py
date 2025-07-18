# Math problem solving
import asyncio
from typing import Dict, Any, List, Optional
import logging
import re
from app.services.groq_service import groq_service
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class MathSolver:
    def __init__(self):
        self.solution_templates = {
            "algebra": "Step-by-step algebraic solution",
            "calculus": "Calculus problem with derivatives/integrals",
            "geometry": "Geometric problem with diagrams and proofs",
            "statistics": "Statistical analysis with data interpretation",
            "general": "General mathematical problem solving"
        }
    
    async def solve_problem(self, question: str, context: Optional[str] = None, 
                          difficulty: str = "medium", topic: str = "general") -> Dict[str, Any]:
        """Solve a mathematical problem using appropriate methodology"""
        
        try:
            # Determine problem type
            problem_type = self._classify_problem(question)
            
            # Build specialized prompt
            solving_prompt = self._build_solving_prompt(question, context, difficulty, problem_type)
            
            # Generate solution
            solution_result = await groq_service.generate_step_by_step_solution(
                question=question,
                context=solving_prompt,
                difficulty=difficulty
            )
            
            if solution_result["success"]:
                # Post-process solution
                enhanced_solution = self._enhance_solution(
                    solution_result["solution"], 
                    problem_type, 
                    question
                )
                
                return {
                    "success": True,
                    "solution": enhanced_solution,
                    "steps": solution_result["steps"],
                    "problem_type": problem_type,
                    "methodology": self._get_methodology(problem_type),
                    "tokens_used": solution_result.get("tokens_used", 0)
                }
            else:
                return solution_result
                
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _classify_problem(self, question: str) -> str:
        """Classify the mathematical problem type"""
        
        question_lower = question.lower()
        
        classification_rules = {
            "algebra": ["solve", "equation", "variable", "linear", "quadratic", "polynomial"],
            "calculus": ["derivative", "integral", "limit", "differential", "continuous", "function"],
            "geometry": ["triangle", "circle", "area", "volume", "angle", "polygon", "coordinate"],
            "statistics": ["probability", "mean", "median", "variance", "distribution", "sample"],
            "trigonometry": ["sin", "cos", "tan", "trigonometric", "angle", "triangle"],
            "number_theory": ["prime", "factor", "divisible", "modular", "gcd", "lcm"]
        }
        
        for prob_type, keywords in classification_rules.items():
            if any(keyword in question_lower for keyword in keywords):
                return prob_type
        
        return "general"
    
    def _build_solving_prompt(self, question: str, context: Optional[str], 
                            difficulty: str, problem_type: str) -> str:
        """Build specialized solving prompt based on problem type"""
        
        base_context = f"""
        Problem Type: {problem_type.title()}
        Difficulty: {difficulty.title()}
        
        Solving Strategy for {problem_type.title()} Problems:
        {self.solution_templates.get(problem_type, self.solution_templates["general"])}
        
        Additional Guidelines:
        - Show all mathematical steps clearly
        - Explain the reasoning behind each step
        - Use proper mathematical notation
        - Verify the final answer
        """
        
        if context:
            base_context += f"\n\nAdditional Context:\n{context}"
        
        return base_context
    
    def _enhance_solution(self, solution: str, problem_type: str, question: str) -> str:
        """Enhance the solution with additional insights"""
        
        enhancements = {
            "algebra": "\n\n**Key Algebraic Concepts Used:**\n- Variable manipulation\n- Equation solving techniques\n- Mathematical properties",
            "calculus": "\n\n**Calculus Concepts Applied:**\n- Differentiation/Integration rules\n- Limit properties\n- Function analysis",
            "geometry": "\n\n**Geometric Principles:**\n- Spatial relationships\n- Measurement formulas\n- Geometric theorems",
            "statistics": "\n\n**Statistical Methods:**\n- Data analysis techniques\n- Probability principles\n- Statistical inference"
        }
        
        enhancement = enhancements.get(problem_type, "")
        
        # Add verification section
        verification = "\n\n**Solution Verification:**\nDouble-check by substituting back into the original problem or using alternative methods."
        
        return solution + enhancement + verification
    
    def _get_methodology(self, problem_type: str) -> Dict[str, Any]:
        """Get methodology information for the problem type"""
        
        methodologies = {
            "algebra": {
                "approach": "Systematic equation manipulation",
                "key_steps": ["Identify variables", "Apply algebraic rules", "Solve systematically", "Verify solution"],
                "common_techniques": ["Substitution", "Elimination", "Factoring", "Completing the square"]
            },
            "calculus": {
                "approach": "Analytical differentiation/integration",
                "key_steps": ["Identify function type", "Apply calculus rules", "Simplify result", "Interpret meaning"],
                "common_techniques": ["Chain rule", "Product rule", "Integration by parts", "Substitution"]
            },
            "geometry": {
                "approach": "Spatial analysis and measurement",
                "key_steps": ["Visualize problem", "Apply geometric formulas", "Calculate measurements", "Verify results"],
                "common_techniques": ["Pythagorean theorem", "Area formulas", "Volume calculations", "Angle relationships"]
            },
            "general": {
                "approach": "Systematic problem analysis",
                "key_steps": ["Understand problem", "Plan solution", "Execute steps", "Verify answer"],
                "common_techniques": ["Logical reasoning", "Mathematical operations", "Pattern recognition"]
            }
        }
        
        return methodologies.get(problem_type, methodologies["general"])