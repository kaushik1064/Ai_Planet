# Math endpoints
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any
import time
import logging

from app.models.schemas import (
    QuestionRequest, SolutionResponse, SimilarProblemsRequest, 
    SimilarProblemsResponse, ImprovementRequest, ImprovementResponse
)
from app.agents.routing_agent import routing_agent
from app.services.groq_service import groq_service
from app.core.guardrails import guardrails_manager

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/solve", response_model=SolutionResponse)
async def solve_math_problem(
    request: QuestionRequest,
    background_tasks: BackgroundTasks
):
    """Solve a mathematical problem using the routing agent"""
    
    start_time = time.time()
    
    try:
        # Process request through guardrails
        guardrail_result = await guardrails_manager.process_request(
            request.question, 
            {"user_id": request.user_id, "difficulty": request.difficulty}
        )
        
        if not guardrail_result["approved"]:
            raise HTTPException(
                status_code=400,
                detail=guardrail_result["error"]
            )
        
        # Use sanitized input
        sanitized_question = guardrail_result["sanitized_input"]
        
        # Route and solve the problem
        result = await routing_agent.route_and_solve(
            question=sanitized_question,
            user_context={
                "user_id": request.user_id,
                "difficulty": request.difficulty,
                "original_question": request.question
            }
        )
        
        # Process response through output guardrails
        if result["success"]:
            output_result = await guardrails_manager.process_response(result["solution"])
            result["solution"] = output_result["filtered_content"]
            result["warnings"] = result.get("warnings", []) + output_result["warnings"]
        
        # Add processing time
        result["processing_time"] = time.time() - start_time
        
        # Log successful request
        background_tasks.add_task(
            log_solution_request,
            sanitized_question,
            result,
            request.user_id
        )
        
        return SolutionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error solving math problem: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while solving the problem"
        )

@router.post("/improve", response_model=ImprovementResponse)
async def improve_solution(request: ImprovementRequest):
    """Improve a solution based on user feedback"""
    
    try:
        # Process feedback and get improvement
        result = await routing_agent.get_feedback_and_improve(
            question=request.question,
            solution=request.solution,
            user_feedback=request.feedback.dict()
        )
        
        return ImprovementResponse(**result)
        
    except Exception as e:
        logger.error(f"Error improving solution: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while improving the solution"
        )

@router.post("/similar-problems", response_model=SimilarProblemsResponse)
async def generate_similar_problems(request: SimilarProblemsRequest):
    """Generate similar practice problems"""
    
    try:
        result = await groq_service.generate_similar_problems(
            request.question,
            request.count
        )
        
        return SimilarProblemsResponse(**result)
        
    except Exception as e:
        logger.error(f"Error generating similar problems: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating similar problems"
        )

@router.post("/check-answer")
async def check_student_answer(
    question: str,
    student_answer: str,
    correct_answer: str
):
    """Check if a student's answer is correct"""
    
    try:
        result = await groq_service.check_answer_correctness(
            question, student_answer, correct_answer
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error checking answer: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while checking the answer"
        )

@router.post("/simplify")
async def simplify_solution(
    solution: str,
    target_level: str = "high_school"
):
    """Simplify a solution for different education levels"""
    
    try:
        if target_level not in ["middle_school", "high_school", "undergraduate", "basic"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid target level"
            )
        
        result = await groq_service.simplify_explanation(solution, target_level)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error simplifying solution: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while simplifying the solution"
        )

@router.get("/knowledge-base/search")
async def search_knowledge_base(
    query: str,
    limit: int = 5
):
    """Search the knowledge base for similar problems"""
    
    try:
        results = await routing_agent._search_knowledge_base(query)
        
        return {
            "success": True,
            "results": results[:limit],
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while searching the knowledge base"
        )

async def log_solution_request(question: str, result: Dict[str, Any], user_id: str):
    """Background task to log solution requests"""
    try:
        # Log to file or database for analytics
        log_entry = {
            "timestamp": time.time(),
            "question": question,
            "success": result["success"],
            "source": result.get("source"),
            "user_id": user_id,
            "processing_time": result.get("processing_time"),
            "tokens_used": result.get("tokens_used")
        }
        
        logger.info(f"Solution request logged: {log_entry}")
        
    except Exception as e:
        logger.error(f"Error logging solution request: {e}")