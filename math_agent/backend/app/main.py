# FastAPI main application
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from typing import Dict, Any, Optional

from app.core.config import get_settings
from app.api.routes import math, feedback, health
from app.core.logging import setup_logging
from app.utils.data_processor import MathDatasetProcessor

settings = get_settings()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Math Routing Agent application...")
    
    # Initialize knowledge base if needed
    try:
        processor = MathDatasetProcessor()
        # Check if knowledge base exists, if not create it
        import os
        if not os.path.exists("data/processed/processed_math_problems.json"):
            logger.info("Knowledge base not found, initializing...")
            await processor.process_complete_dataset()
            logger.info("Knowledge base initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing knowledge base: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Math Routing Agent application...")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI-powered mathematical problem-solving agent with human-in-the-loop feedback",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include API routes
app.include_router(
    health.router,
    prefix=settings.API_PREFIX,
    tags=["health"]
)

app.include_router(
    math.router,
    prefix=settings.API_PREFIX,
    tags=["mathematics"]
)

app.include_router(
    feedback.router,
    prefix=settings.API_PREFIX,
    tags=["feedback"]
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Math Routing Agent API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )