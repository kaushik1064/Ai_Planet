# Feedback endpoints
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import logging

from app.models.schemas import (
    FeedbackRequest, FeedbackResponse, FeedbackStatsResponse
)
from app.agents.feedback_agent import feedback_agent, dspy_integration

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """Submit user feedback for a solution"""
    
    try:
        # Collect feedback
        result = await feedback_agent.collect_feedback(request.dict())
        
        # Process learning in background
        if result["success"]:
            background_tasks.add_task(
                process_feedback_learning,
                result["feedback_id"]
            )
        
        return FeedbackResponse(**result)
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while submitting feedback"
        )

@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(days: int = 30):
    """Get feedback statistics for the specified period"""
    
    try:
        if days < 1 or days > 365:
            raise HTTPException(
                status_code=400,
                detail="Days must be between 1 and 365"
            )
        
        stats = feedback_agent.get_feedback_stats(days)
        
        return FeedbackStatsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving feedback statistics"
        )

@router.get("/insights")
async def get_learning_insights():
    """Get learning insights from accumulated feedback"""
    
    try:
        insights = await feedback_agent.generate_learning_insights()
        
        return {
            "success": True,
            "insights": insights
        }
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating insights"
        )

@router.post("/apply-learning")
async def apply_learning():
    """Apply learning from feedback to improve the system"""
    
    try:
        result = await feedback_agent.apply_learning_from_feedback()
        
        return result
        
    except Exception as e:
        logger.error(f"Error applying learning: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while applying learning"
        )

@router.get("/suggestions/{question}")
async def get_improvement_suggestions(question: str):
    """Get improvement suggestions for a specific type of question"""
    
    try:
        suggestions = await feedback_agent.get_improvement_suggestions_for_query(question)
        
        return {
            "success": True,
            "suggestions": suggestions
        }
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while getting improvement suggestions"
        )

@router.post("/analyze-with-dspy")
async def analyze_feedback_with_dspy(
    feedback_text: str,
    question: str,
    solution: str
):
    """Analyze feedback using DSPy for advanced insights"""
    
    try:
        if not dspy_integration.dspy_available:
            raise HTTPException(
                status_code=501,
                detail="DSPy integration not available"
            )
        
        result = await dspy_integration.analyze_feedback_with_dspy(
            feedback_text, question, solution
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing feedback with DSPy: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while analyzing feedback with DSPy"
        )

async def process_feedback_learning(feedback_id: str):
    """Background task to process feedback for learning"""
    try:
        # Apply learning from this feedback
        await feedback_agent.apply_learning_from_feedback()
        logger.info(f"Learning processed for feedback {feedback_id}")
        
    except Exception as e:
        logger.error(f"Error processing feedback learning: {e}")