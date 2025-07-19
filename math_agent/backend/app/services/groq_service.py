# Groq API integration
# Groq API integration
import asyncio
from typing import Dict, Any, List, Optional
import logging
import re
from groq import AsyncGroq
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class GroqService:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL
        self.temperature = settings.GROQ_TEMPERATURE
        self.max_tokens = settings.GROQ_MAX_TOKENS
        
    async def generate_step_by_step_solution(self, question: str, context: Optional[str] = None, 
                                           difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a step-by-step mathematical solution"""
        
        try:
            # Build the prompt for step-by-step solution
            system_prompt = self._build_system_prompt(difficulty)
            user_prompt = self._build_user_prompt(question, context)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            solution = response.choices[0].message.content
            steps = self._extract_steps_from_solution(solution)
            
            return {
                "success": True,
                "solution": solution,
                "steps": steps,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_similar_problems(self, original_question: str, count: int = 3) -> Dict[str, Any]:
        """Generate similar practice problems"""
        
        try:
            prompt = f"""
            Based on this mathematical problem, generate {count} similar problems with varying difficulty:
            
            Original: {original_question}
            
            For each problem, provide:
            1. The problem statement
            2. Difficulty level (easy/medium/hard)
            3. Brief solution approach
            
            Format each problem as:
            Problem X:
            Statement: [problem statement]
            Difficulty: [level]
            Approach: [solution approach]
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a mathematics tutor creating practice problems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            problems = self._parse_similar_problems(response.choices[0].message.content)
            
            return {
                "success": True,
                "problems": problems,
                "count": len(problems)
            }
            
        except Exception as e:
            logger.error(f"Error generating similar problems: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def check_answer_correctness(self, question: str, student_answer: str, 
                                     correct_answer: str) -> Dict[str, Any]:
        """Check if student's answer is correct"""
        
        try:
            prompt = f"""
            Question: {question}
            Student Answer: {student_answer}
            Correct Answer: {correct_answer}
            
            Analyze if the student's answer is mathematically equivalent to the correct answer.
            Consider different valid forms (simplified vs. unsimplified, different notations, etc.)
            
            Respond with:
            1. CORRECT or INCORRECT
            2. Brief explanation
            3. If incorrect, provide hints for improvement
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a mathematics teacher checking student work."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            is_correct = "CORRECT" in result.upper() and "INCORRECT" not in result.upper()
            
            return {
                "success": True,
                "is_correct": is_correct,
                "explanation": result,
                "feedback": self._extract_feedback_from_result(result)
            }
            
        except Exception as e:
            logger.error(f"Error checking answer: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def simplify_explanation(self, solution: str, target_level: str) -> Dict[str, Any]:
        """Simplify solution for different education levels"""
        
        level_descriptions = {
            "basic": "elementary school level with simple language",
            "middle_school": "middle school level with basic algebra",
            "high_school": "high school level with advanced concepts",
            "undergraduate": "university level with rigorous notation"
        }
        
        try:
            prompt = f"""
            Original solution: {solution}
            
            Rewrite this solution for {level_descriptions.get(target_level, "general")} students.
            
            Guidelines:
            - Use appropriate vocabulary for the level
            - Include more explanatory steps if needed
            - Use visual analogies when helpful
            - Maintain mathematical accuracy
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an educational content adapter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=self.max_tokens
            )
            
            simplified_solution = response.choices[0].message.content
            
            return {
                "success": True,
                "simplified_solution": simplified_solution,
                "target_level": target_level,
                "original_length": len(solution),
                "simplified_length": len(simplified_solution)
            }
            
        except Exception as e:
            logger.error(f"Error simplifying solution: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_system_prompt(self, difficulty: str) -> str:
        """Build system prompt based on difficulty"""
        
        base_prompt = """You are an expert mathematics tutor. Provide clear, step-by-step solutions to mathematical problems."""
        
        difficulty_instructions = {
            "easy": "Use simple language and explain each step thoroughly. Include basic concepts.",
            "medium": "Provide detailed explanations with intermediate mathematical concepts.",
            "hard": "Use advanced mathematical notation and concepts. Assume strong mathematical background."
        }
        
        return f"{base_prompt} {difficulty_instructions.get(difficulty, difficulty_instructions['medium'])}"
    
    def _build_user_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Build user prompt with question and context"""
        
        prompt = f"""
        Solve this mathematical problem step by step:
        
        {question}
        
        Please provide:
        1. Clear identification of what needs to be solved
        2. Step-by-step solution with explanations
        3. Final answer clearly stated
        4. Verification of the result if possible
        """
        
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
        return prompt
    
    def _extract_steps_from_solution(self, solution: str) -> List[Dict[str, str]]:
        """Extract individual steps from the solution"""
        
        steps = []
        
        # Look for numbered steps or step indicators
        step_patterns = [
            r'(?:Step\s*)?(\d+)[:.]?\s*(.+?)(?=(?:Step\s*)?\d+[:.]|$)',
            r'(\d+)\.\s*(.+?)(?=\d+\.|$)',
            r'(First|Second|Third|Fourth|Fifth|Next|Then|Finally)[,:]?\s*(.+?)(?=(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally)|$)'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, solution, re.IGNORECASE | re.DOTALL)
            if matches:
                for i, match in enumerate(matches):
                    steps.append({
                        "step_number": str(i + 1),
                        "description": match[1].strip() if len(match) > 1 else match[0].strip()
                    })
                break
        
        # If no clear steps found, split by paragraphs
        if not steps:
            paragraphs = [p.strip() for p in solution.split('\n\n') if p.strip()]
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 20:  # Only include substantial paragraphs
                    steps.append({
                        "step_number": str(i + 1),
                        "description": paragraph
                    })
        
        return steps
    
    def _parse_similar_problems(self, content: str) -> List[Dict[str, str]]:
        """Parse generated similar problems"""
        
        problems = []
        
        # Split by "Problem" keyword
        problem_sections = re.split(r'Problem\s+\d+:', content, flags=re.IGNORECASE)
        
        for section in problem_sections[1:]:  # Skip first empty section
            problem = {}
            
            # Extract statement
            statement_match = re.search(r'Statement:\s*(.+?)(?=Difficulty:|$)', section, re.DOTALL | re.IGNORECASE)
            if statement_match:
                problem["statement"] = statement_match.group(1).strip()
            
            # Extract difficulty
            difficulty_match = re.search(r'Difficulty:\s*(\w+)', section, re.IGNORECASE)
            if difficulty_match:
                problem["difficulty"] = difficulty_match.group(1).lower()
            
            # Extract approach
            approach_match = re.search(r'Approach:\s*(.+?)(?=Problem|\Z)', section, re.DOTALL | re.IGNORECASE)
            if approach_match:
                problem["approach"] = approach_match.group(1).strip()
            
            if problem.get("statement"):
                problems.append(problem)
        
        return problems
    
    def _extract_feedback_from_result(self, result: str) -> str:
        """Extract helpful feedback from checking result"""
        
        # Look for feedback patterns
        feedback_patterns = [
            r'(?:Hints?|Suggestions?|Feedback):\s*(.+?)(?=\n\n|\Z)',
            r'(?:To improve|For improvement):\s*(.+?)(?=\n\n|\Z)',
            r'(?:Consider|Try|Remember):\s*(.+?)(?=\n\n|\Z)'
        ]
        
        for pattern in feedback_patterns:
            match = re.search(pattern, result, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return "Keep practicing! Mathematics improves with consistent effort."

# Global service instance
groq_service = GroqService()