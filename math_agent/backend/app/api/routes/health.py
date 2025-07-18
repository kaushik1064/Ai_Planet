from fastapi import APIRouter, HTTPException
from datetime import datetime
import time
import logging

from app.models.schemas import HealthResponse
from app.core.config import get_settings
from app.services.web_search import mcp_manager
from app.core.guardrails import guardrails_manager

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# Track application start time
app_start_time = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for all services"""
    
    try:
        services_status = {}
        
        # Check Groq API
        try:
            from app.services.groq_service import groq_service
            # Simple test to check if Groq is responsive
            test_response = await groq_service.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            services_status["groq_api"] = "healthy"
        except Exception as e:
            logger.warning(f"Groq API health check failed: {e}")
            services_status["groq_api"] = "unhealthy"
        
        # Check Vector Database (Qdrant)
        try:
            from app.agents.routing_agent import routing_agent
            # Test basic connectivity
            routing_agent.qdrant_client.get_collections()
            services_status["vector_db"] = "healthy"
        except Exception as e:
            logger.warning(f"Vector DB health check failed: {e}")
            services_status["vector_db"] = "unhealthy"
        
        # Check MCP Server
        mcp_health = await mcp_manager.check_server_health()
        services_status["mcp_server"] = mcp_health["status"]
        
        # Check Knowledge Base
        try:
            import os
            if os.path.exists("data/processed/processed_math_problems.json"):
                services_status["knowledge_base"] = "healthy"
            else:
                services_status["knowledge_base"] = "missing"
        except Exception as e:
            logger.warning(f"Knowledge base health check failed: {e}")
            services_status["knowledge_base"] = "unhealthy"
        
        # Check Feedback System
        try:
            from app.agents.feedback_agent import feedback_agent
            # Test database connectivity
            feedback_agent.get_feedback_stats(1)
            services_status["feedback_system"] = "healthy"
        except Exception as e:
            logger.warning(f"Feedback system health check failed: {e}")
            services_status["feedback_system"] = "unhealthy"
        
        # Check Guardrails
        try:
            test_validation = await guardrails_manager.process_request("test math question")
            services_status["guardrails"] = "healthy" if test_validation["approved"] else "degraded"
        except Exception as e:
            logger.warning(f"Guardrails health check failed: {e}")
            services_status["guardrails"] = "unhealthy"
        
        # Determine overall status
        unhealthy_services = [k for k, v in services_status.items() if v == "unhealthy"]
        if len(unhealthy_services) > 2:
            overall_status = "unhealthy"
        elif len(unhealthy_services) > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        uptime = time.time() - app_start_time
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            version=settings.VERSION,
            services=services_status,
            uptime=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with more information"""
    
    try:
        detailed_info = {}
        
        # Groq API details
        try:
            from app.services.groq_service import groq_service
            detailed_info["groq"] = {
                "model": settings.GROQ_MODEL,
                "api_configured": bool(settings.GROQ_API_KEY),
                "status": "healthy"
            }
        except Exception as e:
            detailed_info["groq"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Vector DB details
        try:
            from app.agents.routing_agent import routing_agent
            collections = routing_agent.qdrant_client.get_collections()
            collection_info = None
            
            # Get collection info if it exists
            try:
                collection_info = routing_agent.qdrant_client.get_collection(settings.QDRANT_COLLECTION_NAME)
                points_count = collection_info.points_count
            except:
                points_count = 0
            
            detailed_info["vector_db"] = {
                "host": settings.QDRANT_HOST,
                "port": settings.QDRANT_PORT,
                "collection": settings.QDRANT_COLLECTION_NAME,
                "points_count": points_count,
                "status": "healthy"
            }
        except Exception as e:
            detailed_info["vector_db"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # MCP Server details
        mcp_health = await mcp_manager.check_server_health()
        mcp_capabilities = await mcp_manager.get_available_capabilities()
        
        detailed_info["mcp_server"] = {
            "url": settings.MCP_SERVER_URL,
            "health": mcp_health,
            "capabilities": mcp_capabilities
        }
        
        # Web Search APIs
        detailed_info["search_apis"] = {
            "tavily_configured": bool(settings.TAVILY_API_KEY),
            "serper_configured": bool(settings.SERPER_API_KEY)
        }
        
        # Feedback system details
        try:
            from app.agents.feedback_agent import feedback_agent
            recent_stats = feedback_agent.get_feedback_stats(7)
            
            detailed_info["feedback_system"] = {
                "database_path": feedback_agent.db_path,
                "recent_feedback_count": recent_stats["total_feedback"],
                "average_rating": recent_stats["average_rating"],
                "status": "healthy"
            }
        except Exception as e:
            detailed_info["feedback_system"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Guardrails details
        try:
            violation_stats = guardrails_manager.get_violation_stats()
            
            detailed_info["guardrails"] = {
                "total_violations": violation_stats["total_violations"],
                "max_input_length": settings.MAX_INPUT_LENGTH,
                "max_output_length": settings.MAX_OUTPUT_LENGTH,
                "status": "healthy"
            }
        except Exception as e:
            detailed_info["guardrails"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "version": settings.VERSION,
            "uptime": time.time() - app_start_time,
            "detailed_status": detailed_info
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Detailed health check failed"
        )

@router.get("/health/services/{service_name}")
async def service_specific_health(service_name: str):
    """Get health status for a specific service"""
    
    service_checks = {
        "groq": check_groq_health,
        "vector_db": check_vector_db_health,
        "mcp": check_mcp_health,
        "feedback": check_feedback_health,
        "guardrails": check_guardrails_health,
        "knowledge_base": check_knowledge_base_health
    }
    
    if service_name not in service_checks:
        raise HTTPException(
            status_code=404,
            detail=f"Service '{service_name}' not found"
        )
    
    try:
        result = await service_checks[service_name]()
        return result
        
    except Exception as e:
        logger.error(f"Service health check failed for {service_name}: {e}", exc_info=True)
        return {
            "service": service_name,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def check_groq_health():
    """Check Groq API health"""
    try:
        from app.services.groq_service import groq_service
        
        start_time = time.time()
        response = await groq_service.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Health check"}],
            max_tokens=5
        )
        response_time = time.time() - start_time
        
        return {
            "service": "groq",
            "status": "healthy",
            "response_time": response_time,
            "model": settings.GROQ_MODEL,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "service": "groq",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def check_vector_db_health():
    """Check Vector Database health"""
    try:
        from app.agents.routing_agent import routing_agent
        
        start_time = time.time()
        collections = routing_agent.qdrant_client.get_collections()
        response_time = time.time() - start_time
        
        return {
            "service": "vector_db",
            "status": "healthy",
            "response_time": response_time,
            "collections_count": len(collections.collections),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "service": "vector_db",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def check_mcp_health():
    """Check MCP Server health"""
    result = await mcp_manager.check_server_health()
    return {
        "service": "mcp",
        **result
    }

async def check_feedback_health():
    """Check Feedback System health"""
    try:
        from app.agents.feedback_agent import feedback_agent
        
        start_time = time.time()
        stats = feedback_agent.get_feedback_stats(1)
        response_time = time.time() - start_time
        
        return {
            "service": "feedback",
            "status": "healthy",
            "response_time": response_time,
            "database_accessible": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "service": "feedback",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def check_guardrails_health():
    """Check Guardrails health"""
    try:
        start_time = time.time()
        test_result = await guardrails_manager.process_request("test math question")
        response_time = time.time() - start_time
        
        return {
            "service": "guardrails",
            "status": "healthy" if test_result["approved"] else "degraded",
            "response_time": response_time,
            "test_passed": test_result["approved"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "service": "guardrails",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def check_knowledge_base_health():
    """Check Knowledge Base health"""
    try:
        import os
        from pathlib import Path
        
        kb_file = Path("data/processed/processed_math_problems.json")
        summary_file = Path("data/processed/dataset_summary.json")
        
        status = "healthy"
        details = {}
        
        if kb_file.exists():
            details["knowledge_base_exists"] = True
            details["knowledge_base_size"] = kb_file.stat().st_size
        else:
            status = "missing"
            details["knowledge_base_exists"] = False
        
        if summary_file.exists():
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            details["total_problems"] = summary.get("total_problems", 0)
            details["categories"] = len(summary.get("categories", {}))
        
        return {
            "service": "knowledge_base",
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "service": "knowledge_base",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }