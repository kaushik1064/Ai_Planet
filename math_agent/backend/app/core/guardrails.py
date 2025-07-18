# Input/Output guardrails
import re
import asyncio
from typing import Dict, Any, List, Optional
import logging
from enum import Enum

from app.core.config import get_settings
from app.services.groq_service import groq_service

logger = logging.getLogger(__name__)
settings = get_settings()

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class InputGuardrails:
    def __init__(self):
        self.max_input_length = settings.MAX_INPUT_LENGTH
        self.allowed_domains = settings.ALLOWED_DOMAINS
        self.blocked_patterns = [
            r'(?i)(?:hack|exploit|attack|malware|virus)',
            r'(?i)(?:password|login|credentials|token)',
            r'(?i)(?:personal|private|confidential|secret)',
            r'(?i)(?:sql|injection|xss|csrf)',
            r'(?i)(?:admin|root|sudo|system)',
        ]
        self.math_keywords = [
            'solve', 'calculate', 'find', 'determine', 'prove', 'show',
            'equation', 'function', 'derivative', 'integral', 'limit',
            'algebra', 'geometry', 'calculus', 'statistics', 'probability',
            'matrix', 'vector', 'graph', 'plot', 'formula', 'theorem'
        ]
    
    async def validate_input(self, question: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive input validation"""
        
        validation_result = {
            "is_valid": True,
            "error": None,
            "warnings": [],
            "threat_level": ThreatLevel.LOW,
            "sanitized_input": question
        }
        
        # Step 1: Basic validation
        basic_validation = self._validate_basic_requirements(question)
        if not basic_validation["is_valid"]:
            validation_result.update(basic_validation)
            return validation_result
        
        # Step 2: Security validation
        security_validation = await self._validate_security(question)
        if not security_validation["is_valid"]:
            validation_result.update(security_validation)
            return validation_result
        
        # Step 3: Domain validation
        domain_validation = self._validate_domain_relevance(question)
        if not domain_validation["is_valid"]:
            validation_result.update(domain_validation)
            return validation_result
        
        # Step 4: Content appropriateness
        content_validation = await self._validate_content_appropriateness(question)
        if not content_validation["is_valid"]:
            validation_result.update(content_validation)
            return validation_result
        
        # Step 5: Privacy validation
        privacy_validation = self._validate_privacy(question, user_context)
        validation_result["warnings"].extend(privacy_validation.get("warnings", []))
        
        # Step 6: Input sanitization
        validation_result["sanitized_input"] = self._sanitize_input(question)
        
        return validation_result
    
    def _validate_basic_requirements(self, question: str) -> Dict[str, Any]:
        """Validate basic input requirements"""
        
        # Check if input is empty
        if not question or not question.strip():
            return {
                "is_valid": False,
                "error": "Question cannot be empty",
                "threat_level": ThreatLevel.LOW
            }
        
        # Check input length
        if len(question) > self.max_input_length:
            return {
                "is_valid": False,
                "error": f"Question too long. Maximum {self.max_input_length} characters allowed",
                "threat_level": ThreatLevel.MEDIUM
            }
        
        # Check for minimum length
        if len(question.strip()) < 3:
            return {
                "is_valid": False,
                "error": "Question too short. Please provide more details",
                "threat_level": ThreatLevel.LOW
            }
        
        return {"is_valid": True}
    
    async def _validate_security(self, question: str) -> Dict[str, Any]:
        """Validate against security threats"""
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, question):
                return {
                    "is_valid": False,
                    "error": "Input contains potentially harmful content",
                    "threat_level": ThreatLevel.HIGH
                }
        
        # Check for code injection attempts
        code_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'__import__\s*\(',
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, question, re.IGNORECASE | re.DOTALL):
                return {
                    "is_valid": False,
                    "error": "Input contains potentially malicious code",
                    "threat_level": ThreatLevel.CRITICAL
                }
        
        # Check for excessive special characters (potential obfuscation)
        special_char_ratio = len(re.findall(r'[^\w\s]', question)) / len(question)
        if special_char_ratio > 0.3:
            return {
                "is_valid": False,
                "error": "Input contains excessive special characters",
                "threat_level": ThreatLevel.MEDIUM
            }
        
        return {"is_valid": True}
    
    def _validate_domain_relevance(self, question: str) -> Dict[str, Any]:
        """Validate that question is mathematics-related"""
        
        question_lower = question.lower()
        
        # Check for math-related keywords
        has_math_keywords = any(keyword in question_lower for keyword in self.math_keywords)
        
        # Check for mathematical symbols and patterns
        math_patterns = [
            r'[+\-*/=]',  # Basic operators
            r'\b\d+\b',   # Numbers
            r'\b[xy]\b',  # Common variables
            r'[(){}[\]]', # Brackets
            r'∫|∑|∏|√|π|∞|≤|≥|≠|±|°',  # Mathematical symbols
            r'\b(?:sin|cos|tan|log|ln|exp|abs|max|min)\b',  # Math functions
        ]
        
        has_math_patterns = any(re.search(pattern, question) for pattern in math_patterns)
        
        # Check for mathematical problem indicators
        problem_indicators = [
            'solve', 'find', 'calculate', 'compute', 'determine', 'evaluate',
            'prove', 'show', 'verify', 'simplify', 'factor', 'expand',
            'derivative', 'integral', 'limit', 'equation', 'inequality'
        ]
        
        has_problem_indicators = any(indicator in question_lower for indicator in problem_indicators)
        
        # Determine if question is math-related
        is_math_related = has_math_keywords or has_math_patterns or has_problem_indicators
        
        if not is_math_related:
            return {
                "is_valid": False,
                "error": "Question does not appear to be mathematics-related. This system only handles mathematical problems.",
                "threat_level": ThreatLevel.LOW
            }
        
        return {"is_valid": True}
    
    async def _validate_content_appropriateness(self, question: str) -> Dict[str, Any]:
        """Validate content appropriateness using AI"""
        
        try:
            # Use Groq to check content appropriateness
            appropriateness_prompt = f"""
            Analyze this mathematical question for appropriateness in an educational context.
            
            Question: {question}
            
            Check for:
            1. Is this a legitimate mathematical question?
            2. Is the content appropriate for educational use?
            3. Does it contain any harmful, offensive, or inappropriate content?
            4. Is it trying to trick or manipulate the system?
            
            Respond with only: APPROPRIATE or INAPPROPRIATE
            If inappropriate, briefly explain why.
            """
            
            # Use a quick, simple check with Groq
            response = await groq_service.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a content moderator for educational mathematics content."},
                    {"role": "user", "content": appropriateness_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip().upper()
            
            if "INAPPROPRIATE" in result:
                return {
                    "is_valid": False,
                    "error": "Content deemed inappropriate for educational use",
                    "threat_level": ThreatLevel.MEDIUM
                }
            
            return {"is_valid": True}
            
        except Exception as e:
            logger.warning(f"Content appropriateness check failed: {e}")
            # If AI check fails, allow the content but log the warning
            return {"is_valid": True, "warnings": ["Could not verify content appropriateness"]}
    
    def _validate_privacy(self, question: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate for privacy concerns"""
        
        warnings = []
        
        # Check for potential PII patterns
        pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', "Potential SSN detected"),
            (r'\b\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b', "Potential credit card number detected"),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email address detected"),
            (r'\b\d{3}-\d{3}-\d{4}\b', "Phone number detected"),
            (r'\b\d{1,5}\s\w+\s(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b', "Address detected"),
        ]
        
        for pattern, warning in pii_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                warnings.append(warning)
        
        # Check for names (simple heuristic)
        potential_names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', question)
        if potential_names:
            warnings.append("Potential personal names detected")
        
        return {"warnings": warnings}
    
    def _sanitize_input(self, question: str) -> str:
        """Sanitize input while preserving mathematical content"""
        
        # Remove potentially harmful HTML/JS
        question = re.sub(r'<[^>]+>', '', question)
        
        # Remove null bytes and control characters
        question = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', question)
        
        # Normalize whitespace
        question = re.sub(r'\s+', ' ', question).strip()
        
        # Remove excessive punctuation (keep mathematical symbols)
        question = re.sub(r'([!?.]){3,}', r'\1\1', question)
        
        return question

class OutputGuardrails:
    def __init__(self):
        self.max_output_length = settings.MAX_OUTPUT_LENGTH
        self.forbidden_phrases = [
            "I cannot", "I'm unable", "It's impossible", "This is not possible",
            "hack", "exploit", "attack", "malware", "virus",
            "personal information", "private data", "confidential"
        ]
        self.required_elements = [
            "step", "solution", "answer"
        ]
    
    async def validate_output(self, solution: str) -> Dict[str, Any]:
        """Validate generated solution output"""
        
        validation_result = {
            "is_valid": True,
            "filtered_content": solution,
            "warnings": [],
            "modifications_made": []
        }
        
        # Step 1: Length validation
        if len(solution) > self.max_output_length:
            validation_result["filtered_content"] = solution[:self.max_output_length] + "...[truncated]"
            validation_result["warnings"].append("Output truncated due to length")
            validation_result["modifications_made"].append("length_truncation")
        
        # Step 2: Content filtering
        filtered_content = self._filter_forbidden_content(validation_result["filtered_content"])
        if filtered_content != validation_result["filtered_content"]:
            validation_result["filtered_content"] = filtered_content
            validation_result["warnings"].append("Some content was filtered")
            validation_result["modifications_made"].append("content_filtering")
        
        # Step 3: Completeness check
        completeness_check = self._check_solution_completeness(validation_result["filtered_content"])
        validation_result["warnings"].extend(completeness_check.get("warnings", []))
        
        # Step 4: Mathematical accuracy check
        accuracy_check = await self._check_mathematical_accuracy(validation_result["filtered_content"])
        validation_result["warnings"].extend(accuracy_check.get("warnings", []))
        
        # Step 5: Educational value check
        educational_check = self._check_educational_value(validation_result["filtered_content"])
        validation_result["warnings"].extend(educational_check.get("warnings", []))
        
        return validation_result
    
    def _filter_forbidden_content(self, content: str) -> str:
        """Filter out forbidden phrases and content"""
        
        filtered_content = content
        
        # Remove forbidden phrases
        for phrase in self.forbidden_phrases:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            filtered_content = pattern.sub("[FILTERED]", filtered_content)
        
        # Remove potential code injection
        filtered_content = re.sub(r'<script[^>]*>.*?</script>', '[FILTERED]', filtered_content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove potential SQL injection
        filtered_content = re.sub(r'\b(?:DROP|DELETE|INSERT|UPDATE|SELECT)\s+\w+', '[FILTERED]', filtered_content, flags=re.IGNORECASE)
        
        return filtered_content
    
    def _check_solution_completeness(self, solution: str) -> Dict[str, Any]:
        """Check if solution is complete and well-structured"""
        
        warnings = []
        
        # Check for required elements
        solution_lower = solution.lower()
        missing_elements = []
        
        for element in self.required_elements:
            if element not in solution_lower:
                missing_elements.append(element)
        
        if missing_elements:
            warnings.append(f"Solution may be incomplete. Missing: {', '.join(missing_elements)}")
        
        # Check for step-by-step structure
        step_patterns = [r'step\s*\d+', r'\d+\.\s', r'first', r'next', r'then', r'finally']
        has_steps = any(re.search(pattern, solution_lower) for pattern in step_patterns)
        
        if not has_steps:
            warnings.append("Solution may lack clear step-by-step structure")
        
        # Check for final answer
        answer_patterns = [r'final\s*answer', r'answer\s*:', r'solution\s*:', r'result\s*:', r'therefore']
        has_answer = any(re.search(pattern, solution_lower) for pattern in answer_patterns)
        
        if not has_answer:
            warnings.append("Solution may not clearly state the final answer")
        
        return {"warnings": warnings}
    
    async def _check_mathematical_accuracy(self, solution: str) -> Dict[str, Any]:
        """Basic mathematical accuracy checks"""
        
        warnings = []
        
        try:
            # Check for mathematical symbols and expressions
            math_expressions = re.findall(r'[0-9+\-*/=()^√∫∑∏]', solution)
            
            # Basic sanity checks
            if '=' in solution:
                # Check for obviously wrong equations like 2+2=5
                simple_equations = re.findall(r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)', solution)
                for eq in simple_equations:
                    left = int(eq[0]) + int(eq[1])
                    right = int(eq[2])
                    if left != right:
                        warnings.append(f"Potential mathematical error detected: {eq[0]}+{eq[1]}={eq[2]}")
            
            # Check for division by zero
            if re.search(r'/\s*0\b', solution):
                warnings.append("Potential division by zero detected")
            
        except Exception as e:
            logger.warning(f"Mathematical accuracy check failed: {e}")
        
        return {"warnings": warnings}
    
    def _check_educational_value(self, solution: str) -> Dict[str, Any]:
        """Check educational value of the solution"""
        
        warnings = []
        
        # Check for explanatory content
        explanation_indicators = [
            'because', 'since', 'therefore', 'thus', 'hence', 'so',
            'explain', 'reason', 'why', 'how', 'what', 'understand'
        ]
        
        solution_lower = solution.lower()
        explanation_count = sum(1 for indicator in explanation_indicators if indicator in solution_lower)
        
        if explanation_count < 2:
            warnings.append("Solution may lack sufficient explanatory content")
        
        # Check for mathematical terminology
        math_terms = [
            'equation', 'variable', 'coefficient', 'constant', 'function',
            'derivative', 'integral', 'limit', 'theorem', 'formula',
            'property', 'rule', 'method', 'technique'
        ]
        
        term_count = sum(1 for term in math_terms if term in solution_lower)
        
        if term_count < 1:
            warnings.append("Solution may lack mathematical terminology")
        
        return {"warnings": warnings}

class GuardrailsManager:
    def __init__(self):
        self.input_guardrails = InputGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.violation_log = []
    
    async def process_request(self, question: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a complete request through guardrails"""
        
        # Input validation
        input_validation = await self.input_guardrails.validate_input(question, user_context)
        
        if not input_validation["is_valid"]:
            self._log_violation("input", input_validation)
            return {
                "approved": False,
                "error": input_validation["error"],
                "threat_level": input_validation["threat_level"].value
            }
        
        return {
            "approved": True,
            "sanitized_input": input_validation["sanitized_input"],
            "warnings": input_validation.get("warnings", [])
        }
    
    async def process_response(self, solution: str) -> Dict[str, Any]:
        """Process a response through output guardrails"""
        
        output_validation = await self.output_guardrails.validate_output(solution)
        
        return {
            "approved": output_validation["is_valid"],
            "filtered_content": output_validation["filtered_content"],
            "warnings": output_validation["warnings"],
            "modifications": output_validation["modifications_made"]
        }
    
    def _log_violation(self, violation_type: str, details: Dict[str, Any]):
        """Log guardrail violations"""
        
        violation_entry = {
            "timestamp": asyncio.get_event_loop().time(),
            "type": violation_type,
            "threat_level": details.get("threat_level", ThreatLevel.LOW).value,
            "error": details.get("error", "Unknown"),
            "details": details
        }
        
        self.violation_log.append(violation_entry)
        logger.warning(f"Guardrail violation: {violation_entry}")
        
        # Keep only recent violations (last 1000)
        if len(self.violation_log) > 1000:
            self.violation_log = self.violation_log[-1000:]
    
    def get_violation_stats(self) -> Dict[str, Any]:
        """Get statistics about guardrail violations"""
        
        if not self.violation_log:
            return {"total_violations": 0}
        
        threat_counts = {}
        type_counts = {}
        
        for violation in self.violation_log:
            threat_level = violation["threat_level"]
            violation_type = violation["type"]
            
            threat_counts[threat_level] = threat_counts.get(threat_level, 0) + 1
            type_counts[violation_type] = type_counts.get(violation_type, 0) + 1
        
        return {
            "total_violations": len(self.violation_log),
            "by_threat_level": threat_counts,
            "by_type": type_counts,
            "recent_violations": self.violation_log[-10:]  # Last 10
        }

# Singleton instances
input_guardrails = InputGuardrails()
output_guardrails = OutputGuardrails()
guardrails_manager = GuardrailsManager()