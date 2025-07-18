# Pydantic models
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="Mathematical question to solve")
    difficulty: Optional[str] = Field("medium", description="Difficulty level: easy, medium, hard")
    user_id: Optional[str] = Field(None, description="User identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        if v not in ['easy', 'medium', 'hard']:
            raise ValueError('Difficulty must be easy, medium, or hard')
        return v

class SolutionResponse(BaseModel):
    success: bool
    solution: Optional[str] = None
    steps: Optional[List[Dict[str, str]]] = None
    source: Optional[str] = None
    similar_problems: Optional[List[Dict[str, Any]]] = None
    web_sources: Optional[List[Dict[str, Any]]] = None
    routing_info: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    tokens_used: Optional[int] = None

class FeedbackType(str, Enum):
    CORRECTNESS = "correctness"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    DIFFICULTY = "difficulty"
    RELEVANCE = "relevance"
    GENERAL = "general"

class FeedbackRequest(BaseModel):
    question: str = Field(..., description="Original question")
    solution: str = Field(..., description="Solution that was provided")
    feedback_type: FeedbackType = Field(FeedbackType.GENERAL, description="Type of feedback")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_text: Optional[str] = Field(None, description="Detailed feedback text")
    user_suggestions: Optional[str] = Field(None, description="User suggestions for improvement")
    user_id: Optional[str] = Field(None, description="User identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class FeedbackResponse(BaseModel):
    success: bool
    feedback_id: Optional[str] = None
    immediate_insights: Optional[Dict[str, Any]] = None
    should_trigger_improvement: Optional[bool] = None
    error: Optional[str] = None

class ImprovementRequest(BaseModel):
    question: str = Field(..., description="Original question")
    solution: str = Field(..., description="Current solution")
    feedback: FeedbackRequest = Field(..., description="User feedback")

class ImprovementResponse(BaseModel):
    success: bool
    improved_solution: Optional[str] = None
    improvement_reason: Optional[List[str]] = None
    feedback_processed: Optional[bool] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]
    uptime: float

class FeedbackStatsResponse(BaseModel):
    period_days: int
    total_feedback: int
    average_rating: float
    positive_feedback: int
    negative_feedback: int
    feedback_by_type: List[Dict[str, Any]]
    daily_trends: List[Dict[str, Any]]

class SimilarProblemsRequest(BaseModel):
    question: str = Field(..., description="Original question to find similar problems for")
    count: int = Field(3, ge=1, le=10, description="Number of similar problems to generate")

class SimilarProblemsResponse(BaseModel):
    success: bool
    problems: Optional[List[Dict[str, str]]] = None
    count: Optional[int] = None
    error: Optional[str] = None