"""
JEE Benchmark evaluation script for real JEE dataset
Usage: python scripts/benchmark_jee.py
"""

import asyncio
import json
import time
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.agents.routing_agent import routing_agent
from app.core.logging import setup_logging
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class JEEBenchmark:
    def __init__(self, jee_dataset_path: str = "data/jee_benchmark"):
        self.jee_dataset_path = Path(jee_dataset_path)
        self.results = []
        self.few_shot_examples = None
        
    def load_jee_dataset(self) -> List[Dict[str, Any]]:
        """Load JEE benchmark dataset"""
        
        dataset_file = self.jee_dataset_path / "dataset.json"
        few_shot_file = self.jee_dataset_path / "few_shot_examples.json"
        
        if not dataset_file.exists():
            logger.error(f"Dataset file not found: {dataset_file}")
            return []
        
        # Load main dataset
        with open(dataset_file, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        
        # Load few-shot examples if available
        if few_shot_file.exists():
            with open(few_shot_file, 'r', encoding='utf-8') as f:
                self.few_shot_examples = json.load(f)
        
        logger.info(f"Loaded {len(problems)} problems from JEE dataset")
        return problems
    
    def load_response_comparison_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load existing model responses for comparison"""
        
        responses_dir = self.jee_dataset_path / "responses"
        comparison_data = {}
        
        if not responses_dir.exists():
            logger.warning("Responses directory not found for comparison")
            return {}
        
        # Load different model responses
        response_files = [
            "GPT3.5_normal_responses",
            "GPT3_normal_responses", 
            "GPT4_CoT_responses",
            "GPT4_normal_responses"
        ]
        
        for file_name in response_files:
            file_path = responses_dir / f"{file_name}.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        comparison_data[file_name] = json.load(f)
                    logger.info(f"Loaded {file_name} with {len(comparison_data[file_name])} responses")
                except Exception as e:
                    logger.error(f"Error loading {file_name}: {e}")
        
        return comparison_data
    
    async def evaluate_problem(self, problem: Dict[str, Any], problem_index: int) -> Dict[str, Any]:
        """Evaluate a single JEE problem"""
        
        start_time = time.time()
        
        try:
            # Extract problem details
            question = problem["question"]
            correct_answer = problem["gold"]
            subject = problem.get("subject", "unknown")
            problem_type = problem.get("type", "MCQ")
            description = problem.get("description", "")
            
            # Determine difficulty based on subject and type
            difficulty_map = {
                ("phy", "MCQ"): "medium",
                ("phy", "MCQ(multiple)"): "hard", 
                ("phy", "Integer"): "medium",
                ("phy", "Numeric"): "hard",
                ("chem", "MCQ"): "medium",
                ("chem", "MCQ(multiple)"): "hard",
                ("chem", "Integer"): "medium", 
                ("chem", "Numeric"): "hard",
                ("math", "MCQ"): "medium",
                ("math", "MCQ(multiple)"): "hard",
                ("math", "Integer"): "hard",
                ("math", "Numeric"): "hard"
            }
            
            difficulty = difficulty_map.get((subject, problem_type), "medium")
            
            # Solve the problem using the routing agent
            result = await routing_agent.route_and_solve(
                question=question,
                user_context={
                    "difficulty": difficulty,
                    "subject": subject,
                    "problem_type": problem_type,
                    "description": description
                }
            )
            
            processing_time = time.time() - start_time
            
            if not result["success"]:
                return {
                    "problem_index": problem_index,
                    "problem_id": problem.get("index", problem_index),
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "processing_time": processing_time,
                    "subject": subject,
                    "type": problem_type
                }
            
            # Extract answer from solution
            generated_answer = self._extract_answer_from_solution(result["solution"], problem_type)
            
            # Check correctness
            is_correct = self._check_correctness(generated_answer, correct_answer, problem_type)
            
            evaluation = {
                "problem_index": problem_index,
                "problem_id": problem.get("index", problem_index),
                "success": True,
                "question": question,
                "generated_solution": result["solution"],
                "generated_answer": generated_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "route_taken": result.get("routing_info", {}).get("route_taken", "unknown"),
                "confidence": result.get("routing_info", {}).get("confidence", 0.0),
                "processing_time": processing_time,
                "tokens_used": result.get("tokens_used", 0),
                "subject": subject,
                "type": problem_type,
                "difficulty": difficulty,
                "description": description
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating problem {problem_index}: {e}")
            return {
                "problem_index": problem_index,
                "problem_id": problem.get("index", problem_index),
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "subject": problem.get("subject", "unknown"),
                "type": problem.get("type", "MCQ")
            }
    
    def _extract_answer_from_solution(self, solution: str, problem_type: str) -> str:
        """Extract answer from the generated solution"""
        
        # Look for common answer patterns
        answer_patterns = [
            r'Therefore,?\s*the\s*answer\s*is\s*\(?([A-D]|[ABCD]+|\d+(?:\.\d+)?)\)?',
            r'Answer:\s*\(?([A-D]|[ABCD]+|\d+(?:\.\d+)?)\)?',
            r'Final\s*answer:\s*\(?([A-D]|[ABCD]+|\d+(?:\.\d+)?)\)?',
            r'\\boxed\{([A-D]|[ABCD]+|\d+(?:\.\d+)?)\}',
            r'\(([A-D])\)\s*(?:is\s*)?(?:the\s*)?(?:correct\s*)?answer',
            r'option\s*\(?([A-D])\)?',
            r'choice\s*\(?([A-D])\)?'
        ]
        
        solution_clean = solution.replace('\n', ' ').strip()
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, solution_clean, re.IGNORECASE)
            if matches:
                answer = matches[-1].strip()  # Take the last match
                return answer.upper() if answer.isalpha() else answer
        
        # For numeric problems, try to extract the last number
        if problem_type in ["Integer", "Numeric"]:
            number_pattern = r'(\d+(?:\.\d+)?)\s*(?:\.|$)'
            numbers = re.findall(number_pattern, solution_clean)
            if numbers:
                return numbers[-1]
        
        # For MCQ problems, look for single letters
        if problem_type.startswith("MCQ"):
            letter_pattern = r'\b([A-D])\b'
            letters = re.findall(letter_pattern, solution_clean)
            if letters:
                return letters[-1].upper()
        
        return "NO_ANSWER_FOUND"
    
    def _check_correctness(self, generated_answer: str, correct_answer: str, problem_type: str) -> bool:
        """Check if the generated answer matches the correct answer"""
        
        if generated_answer == "NO_ANSWER_FOUND":
            return False
        
        # Normalize answers
        generated_clean = generated_answer.strip().upper()
        correct_clean = correct_answer.strip().upper()
        
        # Direct match
        if generated_clean == correct_clean:
            return True
        
        # For multiple choice with multiple answers (e.g., "ABD"), check if sets match
        if problem_type == "MCQ(multiple)":
            gen_set = set(generated_clean.replace(" ", ""))
            correct_set = set(correct_clean.replace(" ", ""))
            return gen_set == correct_set
        
        # For numeric answers, check if they're close (within 0.01)
        if problem_type in ["Integer", "Numeric"]:
            try:
                gen_num = float(generated_clean)
                correct_num = float(correct_clean)
                return abs(gen_num - correct_num) < 0.01
            except ValueError:
                return False
        
        return False
    
    async def run_benchmark(self, max_problems: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete JEE benchmark"""
        
        print("🧪 Starting JEE Benchmark Evaluation")
        print("=" * 50)
        
        # Load dataset
        problems = self.load_jee_dataset()
        if not problems:
            logger.error("No problems loaded from dataset")
            return {}
        
        # Load comparison data
        comparison_data = self.load_response_comparison_data()
        
        # Limit problems if specified
        if max_problems:
            problems = problems[:max_problems]
            print(f"📚 Evaluating first {len(problems)} problems")
        else:
            print(f"📚 Evaluating all {len(problems)} problems")
        
        total_start_time = time.time()
        
        # Evaluate each problem
        for i, problem in enumerate(problems):
            print(f"\n🔄 Evaluating problem {i+1}/{len(problems)}: {problem.get('description', 'Unknown')} - {problem.get('subject', '')}/{problem.get('type', '')}")
            
            evaluation = await self.evaluate_problem(problem, i)
            self.results.append(evaluation)
            
            if evaluation["success"]:
                status = "✅ CORRECT" if evaluation.get("is_correct", False) else "❌ INCORRECT"
                print(f"   {status} | Route: {evaluation.get('route_taken', 'unknown')} | Time: {evaluation['processing_time']:.2f}s")
                if evaluation.get("generated_answer"):
                    print(f"   Generated: {evaluation['generated_answer']} | Expected: {evaluation.get('correct_answer', 'N/A')}")
            else:
                print(f"   ❌ FAILED: {evaluation.get('error', 'Unknown error')}")
        
        total_time = time.time() - total_start_time
        
        # Generate benchmark report
        report = self._generate_report(total_time)
        
        # Add comparison with other models if available
        if comparison_data:
            report["model_comparison"] = self._compare_with_other_models(comparison_data)
        
        # Save results
        results_file = Path("data/jee_benchmark_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "benchmark_info": {
                    "timestamp": time.time(),
                    "total_problems": len(problems),
                    "total_time": total_time,
                    "dataset_path": str(self.jee_dataset_path)
                },
                "results": self.results,
                "report": report
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return report
    
    def _generate_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        
        successful_evaluations = [r for r in self.results if r["success"]]
        correct_solutions = [r for r in successful_evaluations if r.get("is_correct", False)]
        
        total_problems = len(self.results)
        success_rate = len(successful_evaluations) / total_problems if total_problems > 0 else 0
        accuracy_rate = len(correct_solutions) / len(successful_evaluations) if successful_evaluations else 0
        
        # Subject-wise analysis
        subject_stats = {}
        for result in successful_evaluations:
            subject = result.get("subject", "unknown")
            if subject not in subject_stats:
                subject_stats[subject] = {"count": 0, "correct": 0}
            subject_stats[subject]["count"] += 1
            if result.get("is_correct", False):
                subject_stats[subject]["correct"] += 1
        
        # Problem type analysis
        type_stats = {}
        for result in successful_evaluations:
            prob_type = result.get("type", "unknown")
            if prob_type not in type_stats:
                type_stats[prob_type] = {"count": 0, "correct": 0}
            type_stats[prob_type]["count"] += 1
            if result.get("is_correct", False):
                type_stats[prob_type]["correct"] += 1
        
        # Route analysis
        route_stats = {}
        for result in successful_evaluations:
            route = result.get("route_taken", "unknown")
            if route not in route_stats:
                route_stats[route] = {"count": 0, "correct": 0}
            route_stats[route]["count"] += 1
            if result.get("is_correct", False):
                route_stats[route]["correct"] += 1
        
        # Performance metrics
        processing_times = [r["processing_time"] for r in successful_evaluations]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        tokens_used = [r.get("tokens_used", 0) for r in successful_evaluations]
        avg_tokens = sum(tokens_used) / len(tokens_used) if tokens_used else 0
        
        report = {
            "overall_metrics": {
                "total_problems": total_problems,
                "successful_evaluations": len(successful_evaluations),
                "correct_solutions": len(correct_solutions),
                "success_rate": round(success_rate * 100, 2),
                "accuracy_rate": round(accuracy_rate * 100, 2),
                "total_time": round(total_time, 2),
                "avg_processing_time": round(avg_processing_time, 2),
                "avg_tokens_used": round(avg_tokens, 2)
            },
            "subject_performance": {
                subject: {
                    "count": stats["count"],
                    "accuracy": round(stats["correct"] / stats["count"] * 100, 2) if stats["count"] > 0 else 0
                }
                for subject, stats in subject_stats.items()
            },
            "type_performance": {
                prob_type: {
                    "count": stats["count"],
                    "accuracy": round(stats["correct"] / stats["count"] * 100, 2) if stats["count"] > 0 else 0
                }
                for prob_type, stats in type_stats.items()
            },
            "route_performance": {
                route: {
                    "count": stats["count"],
                    "accuracy": round(stats["correct"] / stats["count"] * 100, 2) if stats["count"] > 0 else 0
                }
                for route, stats in route_stats.items()
            }
        }
        
        return report
    
    def _compare_with_other_models(self, comparison_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compare performance with other models"""
        
        comparison_results = {}
        
        for model_name, model_responses in comparison_data.items():
            correct_count = 0
            total_count = len(model_responses)
            
            for response in model_responses:
                if response.get("extract", "").upper() == response.get("gold", "").upper():
                    correct_count += 1
            
            accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
            comparison_results[model_name] = {
                "accuracy": round(accuracy, 2),
                "total_problems": total_count,
                "correct_answers": correct_count
            }
        
        # Add our model's performance
        our_accuracy = self._generate_report(0)["overall_metrics"]["accuracy_rate"]
        comparison_results["Math_Routing_Agent"] = {
            "accuracy": our_accuracy,
            "total_problems": len(self.results),
            "correct_answers": len([r for r in self.results if r.get("is_correct", False)])
        }
        
        return comparison_results
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted benchmark report"""
        
        print("\n" + "=" * 60)
        print("📊 JEE BENCHMARK RESULTS")
        print("=" * 60)
        
        metrics = report["overall_metrics"]
        print(f"📈 Overall Performance:")
        print(f"   Total Problems: {metrics['total_problems']}")
        print(f"   Success Rate: {metrics['success_rate']}%")
        print(f"   Accuracy Rate: {metrics['accuracy_rate']}%")
        print(f"   Avg Processing Time: {metrics['avg_processing_time']}s")
        print(f"   Avg Tokens Used: {metrics['avg_tokens_used']}")
        
        print(f"\n📚 Subject Performance:")
        for subject, stats in report["subject_performance"].items():
            print(f"   {subject.upper()}: {stats['accuracy']}% ({stats['count']} problems)")
        
        print(f"\n📝 Problem Type Performance:")
        for prob_type, stats in report["type_performance"].items():
            print(f"   {prob_type}: {stats['accuracy']}% ({stats['count']} problems)")
        
        print(f"\n🛣️  Route Performance:")
        for route, stats in report["route_performance"].items():
            print(f"   {route.replace('_', ' ').title()}: {stats['accuracy']}% ({stats['count']} problems)")
        
        # Print model comparison if available
        if "model_comparison" in report:
            print(f"\n🏆 Model Comparison:")
            sorted_models = sorted(report["model_comparison"].items(), 
                                 key=lambda x: x[1]["accuracy"], reverse=True)
            for model, stats in sorted_models:
                print(f"   {model}: {stats['accuracy']}% ({stats['correct_answers']}/{stats['total_problems']})")

async def main():
    """Main benchmark function"""
    
    benchmark = JEEBenchmark()
    
    try:
        # Run benchmark (limit to first 50 problems for testing, remove limit for full benchmark)
        report = await benchmark.run_benchmark(max_problems=50)
        benchmark.print_report(report)
        
        print("\n🎉 JEE Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())