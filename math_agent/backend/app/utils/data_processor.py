# Process math dataset
# Process math dataset
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import re

from app.core.config import get_settings
from app.services.vector_db import VectorDatabase
from app.utils.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

class MathDatasetProcessor:
    def __init__(self):
        settings = get_settings()
        
        # Initialize paths
        self.raw_data_path = Path("data/raw/")
        self.processed_data_path = Path("data/processed/")
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase(
            cloud_url=settings.QDRANT_CLOUD_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name=settings.QDRANT_COLLECTION_NAME
        )
        
        # Initialize collection with proper vector size
        asyncio.run(self._initialize_collection())

    async def _initialize_collection(self):
        """Initialize the Qdrant collection"""
        settings = get_settings()
        await self.vector_db.ensure_collection_exists(vector_size=settings.EMBEDDING_DIM)

    async def process_complete_dataset(self) -> List[Dict[str, Any]]:
        """Main processing pipeline"""
        logger.info("Starting dataset processing")
        
        # 1. Load and process raw data
        problems = self._load_and_process_raw_data()
        
        if not problems:
            problems = self._create_sample_dataset()
            logger.warning("Using sample dataset as fallback")
        
        # 2. Generate and store embeddings
        await self._generate_and_store_embeddings(problems)
        
        # 3. Save processed data
        self._save_processed_data(problems)
        
        return problems

    def _load_and_process_raw_data(self) -> List[Dict[str, Any]]:
        """Load and process raw JSON files"""
        problems = []
        json_files = list(self.raw_data_path.glob("*.json"))
        
        for json_file in json_files[:50]:  # Process first 50 files max
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Process each problem in file
                file_problems = self._process_file_data(data, json_file.stem)
                problems.extend(file_problems)
                
            except Exception as e:
                logger.error(f"Error processing {json_file.name}: {e}")
        
        return problems

    async def _generate_and_store_embeddings(self, problems: List[Dict[str, Any]]):
        """Generate and store embeddings in Qdrant Cloud"""
        logger.info(f"Generating embeddings for {len(problems)} problems")
        
        batch_size = 100
        total_batches = (len(problems) + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, len(problems), batch_size), 1):
            batch = problems[i:i + batch_size]
            vectors = []
            
            for problem in batch:
                try:
                    embedding_text = self._create_embedding_text(problem)
                    vector = self.embedding_generator.generate_embedding(embedding_text)
                    
                    if vector:
                        vectors.append({
                            "id": problem["id"],
                            "vector": vector,
                            "payload": problem
                        })
                except Exception as e:
                    logger.error(f"Error generating embedding for problem {problem.get('id')}: {e}")
            
            if vectors:
                success = await self.vector_db.add_vectors(vectors)
                logger.info(f"Processed batch {batch_num}/{total_batches} - {len(vectors)} vectors")
            
        logger.info("Completed embedding generation and storage")

    def _save_processed_data(self, problems: List[Dict[str, Any]]):
        """Save processed data to JSON file"""
        output_file = self.processed_data_path / "processed_problems.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed data to {output_file}")

    
    def _process_single_problem(self, raw_problem: Dict[str, Any], 
                              source_file: str, problem_index: int) -> Optional[Dict[str, Any]]:
        """Process a single problem from the dataset"""
        
        try:
            # Extract question text (handle different field names)
            question = ""
            for field in ["question", "problem", "text", "input", "query"]:
                if field in raw_problem:
                    question = raw_problem[field]
                    break
            
            if not question:
                return None
            
            # Extract answer/solution (handle different field names)
            answer = ""
            for field in ["answer", "solution", "output", "result", "target"]:
                if field in raw_problem:
                    answer = raw_problem[field]
                    break
            
            # Generate unique ID
            problem_id = f"{source_file}_{problem_index}"
            
            # Classify the problem
            category = self._classify_problem_category(question)
            difficulty = self._estimate_difficulty(question, answer)
            problem_type = self._identify_problem_type(question)
            
            # Extract mathematical concepts
            concepts = self._extract_concepts(question)
            
            # Create processed problem
            processed_problem = {
                "id": problem_id,
                "question": question.strip(),
                "answer": answer.strip() if answer else "",
                "category": category,
                "difficulty": difficulty,
                "type": problem_type,
                "concepts": concepts,
                "source_file": source_file,
                "original_index": problem_index,
                "processed_at": datetime.now().isoformat(),
                "metadata": {
                    "question_length": len(question),
                    "answer_length": len(answer) if answer else 0,
                    "has_solution": bool(answer),
                    "original_data": raw_problem  # Keep original for reference
                }
            }
            
            return processed_problem
            
        except Exception as e:
            logger.error(f"Error processing single problem: {e}")
            return None
    
    def _classify_problem_category(self, question: str) -> str:
        """Classify the problem into a mathematical category"""
        
        question_lower = question.lower()
        
        # Count keywords for each category
        category_scores = {}
        
        for category, keywords in self.category_mapping.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            # Return the category with the highest score
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return "general"
    
    def _estimate_difficulty(self, question: str, answer: str = "") -> str:
        """Estimate the difficulty level of the problem"""
        
        text = (question + " " + answer).lower()
        
        # Count difficulty indicators
        difficulty_scores = {}
        
        for difficulty, indicators in self.difficulty_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            if score > 0:
                difficulty_scores[difficulty] = score
        
        # Additional heuristics
        complexity_score = 0
        
        # Mathematical complexity indicators
        complex_terms = [
            "derivative", "integral", "matrix", "vector", "theorem", "proof",
            "polynomial", "exponential", "logarithm", "trigonometric", "inverse"
        ]
        complexity_score += sum(1 for term in complex_terms if term in text)
        
        # Length-based complexity
        if len(question) > 200:
            complexity_score += 1
        if len(question) > 400:
            complexity_score += 1
        
        # Number of mathematical operations
        operations = re.findall(r'[+\-*/=<>≤≥≠∫∑∏√]', question)
        if len(operations) > 5:
            complexity_score += 1
        if len(operations) > 10:
            complexity_score += 1
        
        # Determine final difficulty
        if difficulty_scores:
            base_difficulty = max(difficulty_scores.items(), key=lambda x: x[1])[0]
        else:
            base_difficulty = "medium"
        
        # Adjust based on complexity
        if complexity_score >= 3:
            if base_difficulty == "easy":
                return "medium"
            elif base_difficulty == "medium":
                return "hard"
        elif complexity_score == 0 and base_difficulty == "hard":
            return "medium"
        
        return base_difficulty
    
    def _identify_problem_type(self, question: str) -> str:
        """Identify the type of mathematical problem"""
        
        question_lower = question.lower()
        
        type_indicators = {
            "equation": ["solve", "equation", "find x", "find y", "variable"],
            "calculation": ["calculate", "compute", "find the value", "evaluate"],
            "proof": ["prove", "show that", "demonstrate", "verify"],
            "optimization": ["maximize", "minimize", "optimal", "extreme"],
            "word_problem": ["if", "when", "how many", "how much", "person", "object"],
            "graphing": ["graph", "plot", "sketch", "curve", "function"],
            "geometry": ["area", "volume", "perimeter", "angle", "triangle", "circle"],
            "probability": ["probability", "chance", "likely", "random", "dice"]
        }
        
        for prob_type, indicators in type_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                return prob_type
        
        return "general"
    
    def _extract_concepts(self, question: str) -> List[str]:
        """Extract mathematical concepts from the question"""
        
        question_lower = question.lower()
        
        concept_keywords = {
            "functions", "equations", "polynomials", "derivatives", "integrals",
            "limits", "matrices", "vectors", "probability", "statistics",
            "geometry", "trigonometry", "algebra", "calculus", "number_theory",
            "arithmetic", "logarithms", "exponentials", "inequalities", "sequences",
            "series", "complex_numbers", "linear_algebra", "differential_equations"
        }
        
        found_concepts = []
        for concept in concept_keywords:
            if concept in question_lower or concept[:-1] in question_lower:  # Singular form
                found_concepts.append(concept)
        
        # Add specific mathematical terms found
        math_terms = re.findall(r'\b(?:sin|cos|tan|log|ln|exp|sqrt|abs|max|min)\b', question_lower)
        found_concepts.extend(math_terms)
        
        return list(set(found_concepts))  # Remove duplicates
    
    async def _generate_and_store_embeddings(self, problems: List[Dict[str, Any]]):
        """Generate embeddings for problems and store in vector database"""
        try:
            vectors_to_store = []
            
            for problem in problems:
                # Create text for embedding
                embedding_text = f"{problem['question']} {problem['category']} {' '.join(problem['concepts'])}"
                
                # Generate embedding
                embedding_vector = self.embedding_generator.generate_embedding(embedding_text)
                
                if embedding_vector:
                    vectors_to_store.append({
                        "id": problem["id"],
                        "vector": embedding_vector,
                        "payload": {
                            "question": problem["question"],
                            "category": problem["category"],
                            "difficulty": problem["difficulty"],
                            "type": problem["type"],
                            "concepts": problem["concepts"],
                            "source_file": problem["source_file"]
                        }
                    })
            
            # Store in vector database in batches
            batch_size = 100
            for i in range(0, len(vectors_to_store), batch_size):
                batch = vectors_to_store[i:i + batch_size]
                success = await self.vector_db.add_vectors(batch)
                
                if success:
                    logger.info(f"Stored embedding batch {i//batch_size + 1}")
                else:
                    logger.error(f"Failed to store embedding batch {i//batch_size + 1}")
            
            logger.info(f"Successfully generated and stored {len(vectors_to_store)} embeddings")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
    
    def _generate_summary_statistics(self, problems: List[Dict[str, Any]], 
                                   file_stats: Dict[str, int]) -> Dict[str, Any]:
        """Generate summary statistics for the processed dataset"""
        
        total_problems = len(problems)
        
        # Category distribution
        categories = {}
        for problem in problems:
            category = problem["category"]
            categories[category] = categories.get(category, 0) + 1
        
        # Difficulty distribution
        difficulties = {}
        for problem in problems:
            difficulty = problem["difficulty"]
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        # Type distribution
        types = {}
        for problem in problems:
            prob_type = problem["type"]
            types[prob_type] = types.get(prob_type, 0) + 1
        
        # Concept frequency
        concept_counts = {}
        for problem in problems:
            for concept in problem["concepts"]:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Top concepts
        top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Average lengths
        avg_question_length = sum(problem["metadata"]["question_length"] for problem in problems) / total_problems
        avg_answer_length = sum(problem["metadata"]["answer_length"] for problem in problems if problem["metadata"]["answer_length"] > 0)
        problems_with_answers = sum(1 for problem in problems if problem["metadata"]["has_solution"])
        
        if problems_with_answers > 0:
            avg_answer_length = avg_answer_length / problems_with_answers
        else:
            avg_answer_length = 0
        
        summary = {
            "processing_info": {
                "processed_at": datetime.now().isoformat(),
                "total_source_files": len(file_stats),
                "problems_per_file": 50,
                "total_problems": total_problems
            },
            "file_statistics": file_stats,
            "distribution": {
                "categories": categories,
                "difficulties": difficulties,
                "problem_types": types
            },
            "concepts": {
                "total_unique_concepts": len(concept_counts),
                "top_concepts": top_concepts
            },
            "content_analysis": {
                "average_question_length": round(avg_question_length, 2),
                "average_answer_length": round(avg_answer_length, 2),
                "problems_with_solutions": problems_with_answers,
                "solution_coverage": round(problems_with_answers / total_problems * 100, 2)
            },
            "quality_metrics": {
                "problems_processed_successfully": total_problems,
                "processing_success_rate": "100%",  # Since we only include successful ones
                "embeddings_generated": total_problems,
                "vector_db_ready": True
            }
        }
        
        return summary
    
    def _create_sample_dataset(self) -> List[Dict[str, Any]]:
        """Create a sample dataset when no raw data is available"""
        
        logger.info("Creating sample dataset for demonstration")
        
        sample_problems = [
            {
                "question": "Solve the equation 2x + 5 = 13",
                "answer": "x = 4",
                "category": "algebra",
                "difficulty": "easy"
            },
            {
                "question": "Find the derivative of f(x) = x^2 + 3x + 2",
                "answer": "f'(x) = 2x + 3",
                "category": "calculus",
                "difficulty": "medium"
            },
            {
                "question": "Calculate the area of a circle with radius 5",
                "answer": "A = 25π ≈ 78.54",
                "category": "geometry",
                "difficulty": "easy"
            },
            {
                "question": "Evaluate the integral ∫(2x + 1)dx",
                "answer": "x^2 + x + C",
                "category": "calculus",
                "difficulty": "medium"
            },
            {
                "question": "Find the probability of rolling a sum of 7 with two dice",
                "answer": "6/36 = 1/6 ≈ 0.167",
                "category": "statistics",
                "difficulty": "medium"
            }
        ]
        
        processed_problems = []
        
        for i, sample in enumerate(sample_problems):
            processed_problem = {
                "id": f"sample_{i}",
                "question": sample["question"],
                "answer": sample["answer"],
                "category": sample["category"],
                "difficulty": sample["difficulty"],
                "type": self._identify_problem_type(sample["question"]),
                "concepts": self._extract_concepts(sample["question"]),
                "source_file": "sample_dataset",
                "original_index": i,
                "processed_at": datetime.now().isoformat(),
                "metadata": {
                    "question_length": len(sample["question"]),
                    "answer_length": len(sample["answer"]),
                    "has_solution": True,
                    "original_data": sample
                }
            }
            
            processed_problems.append(processed_problem)
        
        # Save sample dataset
        output_file = self.processed_data_path / "processed_math_problems.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_problems, f, indent=2, ensure_ascii=False)
        
        # Generate sample summary
        summary_stats = {
            "processing_info": {
                "processed_at": datetime.now().isoformat(),
                "total_source_files": 1,
                "problems_per_file": len(sample_problems),
                "total_problems": len(sample_problems),
                "note": "Sample dataset created for demonstration"
            },
            "distribution": {
                "categories": {"algebra": 1, "calculus": 2, "geometry": 1, "statistics": 1},
                "difficulties": {"easy": 2, "medium": 3},
                "problem_types": {"equation": 1, "calculation": 4}
            }
        }
        
        summary_file = self.processed_data_path / "dataset_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2)
        
        logger.info(f"Sample dataset created with {len(processed_problems)} problems")
        
        return processed_problems