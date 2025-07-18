# Knowledge base operations
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self, kb_path: str = "data/processed/processed_math_problems.json"):
        self.kb_path = Path(kb_path)
        self.problems = self._load_problems()
    
    def _load_problems(self) -> List[Dict[str, Any]]:
        """Load problems from the knowledge base file"""
        try:
            if self.kb_path.exists():
                with open(self.kb_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Knowledge base file not found: {self.kb_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return []
    
    def search_problems(self, query: str, category: Optional[str] = None, 
                       difficulty: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Search problems by text similarity"""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for problem in self.problems:
            # Apply filters
            if category and problem.get("category") != category:
                continue
            if difficulty and problem.get("difficulty") != difficulty:
                continue
            
            # Calculate relevance score
            question_text = problem.get("question", "").lower()
            question_words = set(question_text.split())
            
            # Jaccard similarity
            intersection = query_words.intersection(question_words)
            union = query_words.union(question_words)
            similarity = len(intersection) / len(union) if union else 0
            
            if similarity > 0.1:  # Minimum similarity threshold
                problem_copy = problem.copy()
                problem_copy["relevance_score"] = similarity
                results.append(problem_copy)
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:limit]
    
    def get_problem_by_id(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific problem by ID"""
        for problem in self.problems:
            if problem.get("id") == problem_id:
                return problem
        return None
    
    def get_problems_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get problems by category"""
        results = []
        for problem in self.problems:
            if problem.get("category") == category:
                results.append(problem)
                if len(results) >= limit:
                    break
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        if not self.problems:
            return {"total": 0}
        
        categories = {}
        difficulties = {}
        source_files = {}
        
        for problem in self.problems:
            cat = problem.get("category", "unknown")
            diff = problem.get("difficulty", "unknown")
            source = problem.get("source_file", "unknown")
            
            categories[cat] = categories.get(cat, 0) + 1
            difficulties[diff] = difficulties.get(diff, 0) + 1
            source_files[source] = source_files.get(source, 0) + 1
        
        return {
            "total": len(self.problems),
            "categories": categories,
            "difficulties": difficulties,
            "source_files": len(source_files),
            "avg_question_length": sum(len(p.get("question", "")) for p in self.problems) / len(self.problems)
        }
    
    def get_random_problems(self, count: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get random problems for testing"""
        import random
        
        filtered_problems = self.problems
        if category:
            filtered_problems = [p for p in self.problems if p.get("category") == category]
        
        if len(filtered_problems) <= count:
            return filtered_problems
        
        return random.sample(filtered_problems, count)
    
    def refresh_knowledge_base(self) -> bool:
        """Reload the knowledge base from file"""
        try:
            self.problems = self._load_problems()
            logger.info(f"Knowledge base refreshed: {len(self.problems)} problems loaded")
            return True
        except Exception as e:
            logger.error(f"Error refreshing knowledge base: {e}")
            return False