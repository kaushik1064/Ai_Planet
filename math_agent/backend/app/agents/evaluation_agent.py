# Solution evaluation
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import logging
import re
from datetime import datetime
from app.services.groq_service import groq_service

logger = logging.getLogger(__name__)

class EvaluationAgent:
    def __init__(self):
        self.evaluation_criteria = {
            "correctness": "Mathematical accuracy and validity",
            "clarity": "Explanation clarity and understandability", 
            "completeness": "Solution completeness and thoroughness",
            "methodology": "Appropriateness of solution approach",
            "presentation": "Quality of presentation and formatting"
        }
    
    async def evaluate_solution(self, question: str, solution: str, 
                              expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive solution evaluation"""
        
        try:
            # Perform multiple evaluation checks
            evaluations = await asyncio.gather(
                self._evaluate_correctness(question, solution, expected_answer),
                self._evaluate_clarity(solution),
                self._evaluate_completeness(question, solution),
                self._evaluate_methodology(question, solution),
                self._evaluate_presentation(solution)
            )
            
            correctness_eval, clarity_eval, completeness_eval, methodology_eval, presentation_eval = evaluations
            
            # Calculate overall score
            scores = {
                "correctness": correctness_eval["score"],
                "clarity": clarity_eval["score"],
                "completeness": completeness_eval["score"], 
                "methodology": methodology_eval["score"],
                "presentation": presentation_eval["score"]
            }
            
            # Weighted average (correctness is most important)
            weights = {
                "correctness": 0.4,
                "clarity": 0.2,
                "completeness": 0.2,
                "methodology": 0.15,
                "presentation": 0.05
            }
            
            overall_score = sum(scores[criteria] * weights[criteria] for criteria in scores)
            
            return {
                "success": True,
                "overall_score": round(overall_score, 2),
                "individual_scores": scores,
                "detailed_feedback": {
                    "correctness": correctness_eval,
                    "clarity": clarity_eval,
                    "completeness": completeness_eval,
                    "methodology": methodology_eval,
                    "presentation": presentation_eval
                },
                "recommendations": self._generate_recommendations(scores),
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating solution: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _evaluate_correctness(self, question: str, solution: str, 
                                  expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate mathematical correctness"""
        
        correctness_prompt = f"""
        Evaluate the mathematical correctness of this solution:
        
        Question: {question}
        Solution: {solution}
        {f"Expected Answer: {expected_answer}" if expected_answer else ""}
        
        Rate the correctness on a scale of 1-10 and provide specific feedback on:
        1. Mathematical accuracy
        2. Logical consistency
        3. Proper use of mathematical concepts
        4. Validity of the final answer
        
        Format: Score: X/10, Feedback: [detailed explanation]
        """
        
        try:
            response = await groq_service.client.chat.completions.create(
                model=groq_service.model,
                messages=[
                    {"role": "system", "content": "You are a mathematics expert evaluating solution correctness."},
                    {"role": "user", "content": correctness_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            evaluation_text = response.choices[0].message.content
            score = self._extract_score(evaluation_text)
            
            return {
                "score": score,
                "feedback": evaluation_text,
                "criteria": "Mathematical correctness and accuracy"
            }
            
        except Exception as e:
            logger.error(f"Error evaluating correctness: {e}")
            return {"score": 5.0, "feedback": "Could not evaluate correctness", "criteria": "Mathematical correctness"}
    
    async def _evaluate_clarity(self, solution: str) -> Dict[str, Any]:
        """Evaluate explanation clarity"""
        
        clarity_indicators = {
            "clear_steps": len(re.findall(r'step\s*\d+|first|next|then|finally', solution.lower())),
            "explanations": len(re.findall(r'because|since|therefore|thus|hence', solution.lower())),
            "organization": len(re.findall(r'\n|paragraph|section', solution.lower())),
            "mathematical_notation": len(re.findall(r'\$|\\|=|\+|\-|\*|\/', solution))
        }
        
        # Simple scoring based on indicators
        clarity_score = min(10, (
            min(clarity_indicators["clear_steps"], 5) * 1.5 +
            min(clarity_indicators["explanations"], 3) * 1.0 +
            min(clarity_indicators["organization"], 3) * 0.5 +
            min(clarity_indicators["mathematical_notation"], 10) * 0.3
        ))
        
        feedback = f"Solution shows {clarity_indicators['clear_steps']} clear steps, {clarity_indicators['explanations']} explanatory words, and good mathematical notation."
        
        return {
            "score": round(clarity_score, 1),
            "feedback": feedback,
            "criteria": "Explanation clarity and understandability",
            "indicators": clarity_indicators
        }
    
    async def _evaluate_completeness(self, question: str, solution: str) -> Dict[str, Any]:
        """Evaluate solution completeness"""
        
        required_elements = {
            "problem_understanding": any(word in solution.lower() for word in ["given", "find", "solve", "determine"]),
            "solution_steps": "step" in solution.lower() or any(word in solution.lower() for word in ["first", "next", "then"]),
            "final_answer": any(word in solution.lower() for word in ["answer", "result", "solution", "therefore"]),
            "verification": any(word in solution.lower() for word in ["check", "verify", "confirm", "substitute"])
        }
        
        completeness_score = sum(required_elements.values()) / len(required_elements) * 10
        
        missing_elements = [element for element, present in required_elements.items() if not present]
        
        feedback = f"Solution includes {sum(required_elements.values())}/{len(required_elements)} key elements."
        if missing_elements:
            feedback += f" Missing: {', '.join(missing_elements)}"
        
        return {
            "score": round(completeness_score, 1),
            "feedback": feedback,
            "criteria": "Solution completeness and thoroughness",
            "required_elements": required_elements
        }
    
    async def _evaluate_methodology(self, question: str, solution: str) -> Dict[str, Any]:
        """Evaluate solution methodology"""
        
        methodology_prompt = f"""
        Evaluate the methodology used in this mathematical solution:
        
        Question: {question}
        Solution: {solution}
        
        Rate the methodology on a scale of 1-10 considering:
        1. Appropriateness of the chosen approach
        2. Efficiency of the solution method
        3. Use of standard mathematical techniques
        4. Logical flow of the solution
        
        Format: Score: X/10, Feedback: [brief explanation]
        """
        
        try:
            response = await groq_service.client.chat.completions.create(
                model=groq_service.model,
                messages=[
                    {"role": "system", "content": "You are a mathematics pedagogy expert."},
                    {"role": "user", "content": methodology_prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            evaluation_text = response.choices[0].message.content
            score = self._extract_score(evaluation_text)
            
            return {
                "score": score,
                "feedback": evaluation_text,
                "criteria": "Appropriateness of solution methodology"
            }
            
        except Exception as e:
            logger.error(f"Error evaluating methodology: {e}")
            return {"score": 7.0, "feedback": "Standard methodology used", "criteria": "Solution methodology"}
    
    async def _evaluate_presentation(self, solution: str) -> Dict[str, Any]:
        """Evaluate solution presentation quality"""
        
        presentation_indicators = {
            "formatting": len(re.findall(r'\n\n|###|##|\*\*|\*', solution)),
            "mathematical_symbols": len(re.findall(r'\$|\\\(|\\\[|\\begin', solution)),
            "structure": len(re.findall(r'step|section|part', solution.lower())),
            "readability": len(solution.split()) > 20  # Sufficient detail
        }
        
        presentation_score = min(10, (
            min(presentation_indicators["formatting"], 5) * 1.0 +
            min(presentation_indicators["mathematical_symbols"], 5) * 1.0 +
            min(presentation_indicators["structure"], 3) * 1.5 +
            (2 if presentation_indicators["readability"] else 0)
        ))
        
        feedback = "Solution has good formatting and mathematical notation." if presentation_score > 7 else "Presentation could be improved with better formatting."
        
        return {
            "score": round(presentation_score, 1),
            "feedback": feedback,
            "criteria": "Quality of presentation and formatting"
        }
    
    def _extract_score(self, evaluation_text: str) -> float:
        """Extract numerical score from evaluation text"""
        
        # Look for patterns like "Score: 8/10" or "8 out of 10" or "Rating: 7"
        patterns = [
            r'score:\s*(\d+(?:\.\d+)?)/10',
            r'score:\s*(\d+(?:\.\d+)?)',
            r'rating:\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*/\s*10',
            r'(\d+(?:\.\d+)?)\s*out\s*of\s*10'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, evaluation_text.lower())
            if match:
                score = float(match.group(1))
                return min(score, 10.0)  # Ensure score doesn't exceed 10
        
        # Default score if no pattern matches
        return 7.0
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on scores"""
        
        recommendations = []
        
        if scores["correctness"] < 7:
            recommendations.append("Review mathematical concepts and double-check calculations")
        
        if scores["clarity"] < 7:
            recommendations.append("Improve explanation clarity with more step-by-step details")
        
        if scores["completeness"] < 7:
            recommendations.append("Include all solution steps and verify the final answer")
        
        if scores["methodology"] < 7:
            recommendations.append("Consider alternative solution approaches or methods")
        
        if scores["presentation"] < 7:
            recommendations.append("Enhance presentation with better formatting and mathematical notation")
        
        if not recommendations:
            recommendations.append("Excellent solution! Continue with this approach.")
        
        return recommendations
    
    async def compare_solutions(self, question: str, solutions: List[str]) -> Dict[str, Any]:
        """Compare multiple solutions to the same problem"""
        
        try:
            evaluations = []
            
            for i, solution in enumerate(solutions):
                eval_result = await self.evaluate_solution(question, solution)
                eval_result["solution_index"] = i
                evaluations.append(eval_result)
            
            # Rank solutions by overall score
            evaluations.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
            
            return {
                "success": True,
                "best_solution_index": evaluations[0]["solution_index"],
                "solution_rankings": evaluations,
                "comparison_summary": self._generate_comparison_summary(evaluations)
            }
            
        except Exception as e:
            logger.error(f"Error comparing solutions: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_comparison_summary(self, evaluations: List[Dict[str, Any]]) -> str:
        """Generate a summary comparing multiple solutions"""
        
        if not evaluations:
            return "No solutions to compare."
        
        best = evaluations[0]
        worst = evaluations[-1]
        
        summary = f"Best solution (Index {best['solution_index']}) scored {best['overall_score']}/10. "
        
        if len(evaluations) > 1:
            summary += f"Lowest scoring solution (Index {worst['solution_index']}) scored {worst['overall_score']}/10. "
        
        # Identify strengths and weaknesses
        best_scores = best.get("individual_scores", {})
        strongest_aspect = max(best_scores.items(), key=lambda x: x[1]) if best_scores else ("overall", 0)
        
        summary += f"The best solution excelled in {strongest_aspect[0]} with a score of {strongest_aspect[1]}/10."
        
        return summary