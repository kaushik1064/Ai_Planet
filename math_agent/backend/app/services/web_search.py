# Web search using MCP
# Web search using MCP
import asyncio
import httpx
from typing import Dict, Any, List, Optional
import logging
import json
from urllib.parse import quote_plus

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class MCPManager:
    def __init__(self):
        self.server_url = settings.MCP_SERVER_URL
        self.client = httpx.AsyncClient(timeout=30.0)
        self.capabilities = {}
    
    async def check_server_health(self) -> Dict[str, Any]:
        """Check if MCP server is healthy"""
        try:
            response = await self.client.get(f"{self.server_url}/health")
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "timestamp": response.json().get("timestamp"),
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "unreachable",
                "error": str(e)
            }
    
    async def get_available_capabilities(self) -> List[str]:
        """Get available MCP capabilities"""
        try:
            response = await self.client.get(f"{self.server_url}/capabilities")
            if response.status_code == 200:
                data = response.json()
                return data.get("capabilities", [])
            return []
        except Exception as e:
            logger.error(f"Error getting MCP capabilities: {e}")
            return []
    
    async def search_web(self, query: str, search_type: str = "general") -> List[Dict[str, Any]]:
        """Search the web using MCP server"""
        try:
            payload = {
                "query": query,
                "type": search_type,
                "max_results": 5
            }
            
            response = await self.client.post(
                f"{self.server_url}/search",
                json=payload
            )
            
            if response.status_code == 200:
                return response.json().get("results", [])
            else:
                logger.error(f"MCP search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error in MCP web search: {e}")
            return []

class WebSearchService:
    def __init__(self):
        self.tavily_api_key = settings.TAVILY_API_KEY
        self.serper_api_key = settings.SERPER_API_KEY
        self.mcp_manager = MCPManager()
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search_math_resources(self, question: str, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for mathematical resources and examples"""
        
        # Build search query
        search_query = self._build_math_search_query(question, topic)
        
        # Try multiple search providers
        results = []
        
        # Try MCP server first
        try:
            mcp_results = await self.mcp_manager.search_web(search_query, "math")
            results.extend(self._process_search_results(mcp_results, "mcp"))
        except Exception as e:
            logger.warning(f"MCP search failed: {e}")
        
        # Try Tavily if available
        if self.tavily_api_key and len(results) < 3:
            try:
                tavily_results = await self._search_with_tavily(search_query)
                results.extend(self._process_search_results(tavily_results, "tavily"))
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}")
        
        # Try Serper if available
        if self.serper_api_key and len(results) < 3:
            try:
                serper_results = await self._search_with_serper(search_query)
                results.extend(self._process_search_results(serper_results, "serper"))
            except Exception as e:
                logger.warning(f"Serper search failed: {e}")
        
        # Filter and rank results
        filtered_results = self._filter_math_results(results, question)
        
        return filtered_results[:5]  # Return top 5 results
    
    def _build_math_search_query(self, question: str, topic: Optional[str] = None) -> str:
        """Build an optimized search query for mathematical content"""
        
        # Extract key mathematical terms
        math_terms = self._extract_math_terms(question)
        
        # Build query components
        query_parts = []
        
        if topic:
            query_parts.append(f"{topic} mathematics")
        
        # Add the core question terms
        query_parts.extend(math_terms[:3])  # Top 3 mathematical terms
        
        # Add context keywords
        query_parts.extend(["example", "solution", "tutorial"])
        
        return " ".join(query_parts)
    
    def _extract_math_terms(self, question: str) -> List[str]:
        """Extract mathematical terms from the question"""
        
        import re
        
        # Mathematical keywords to prioritize
        math_keywords = [
            "equation", "function", "derivative", "integral", "limit", "matrix",
            "vector", "graph", "solve", "calculate", "theorem", "proof", "formula",
            "algebra", "calculus", "geometry", "trigonometry", "statistics",
            "probability", "linear", "quadratic", "polynomial", "exponential",
            "logarithm", "sine", "cosine", "tangent", "inverse", "domain", "range"
        ]
        
        # Extract words from question
        words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
        
        # Prioritize mathematical terms
        math_terms = [word for word in words if word in math_keywords]
        other_terms = [word for word in words if word not in math_keywords and len(word) > 3]
        
        return math_terms + other_terms[:5]  # Math terms + up to 5 other terms
    
    async def _search_with_tavily(self, query: str) -> List[Dict[str, Any]]:
        """Search using Tavily API"""
        
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "include_domains": [
                "khanacademy.org", "mathworld.wolfram.com", "brilliant.org",
                "mathisfun.com", "wikipedia.org", "stackexchange.com"
            ],
            "max_results": 5
        }
        
        response = await self.client.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        else:
            logger.error(f"Tavily API error: {response.status_code}")
            return []
    
    async def _search_with_serper(self, query: str) -> List[Dict[str, Any]]:
        """Search using Serper API"""
        
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": f"{query} site:khanacademy.org OR site:mathworld.wolfram.com OR site:brilliant.org",
            "num": 5
        }
        
        response = await self.client.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("organic", [])
        else:
            logger.error(f"Serper API error: {response.status_code}")
            return []
    
    def _process_search_results(self, raw_results: List[Dict[str, Any]], 
                              source: str) -> List[Dict[str, Any]]:
        """Process and standardize search results from different sources"""
        
        processed_results = []
        
        for result in raw_results:
            try:
                # Standardize result format
                processed_result = {
                    "title": "",
                    "url": "",
                    "summary": "",
                    "source": source,
                    "relevance_score": 0.0
                }
                
                if source == "tavily":
                    processed_result.update({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "summary": result.get("content", ""),
                        "relevance_score": result.get("score", 0.0)
                    })
                elif source == "serper":
                    processed_result.update({
                        "title": result.get("title", ""),
                        "url": result.get("link", ""),
                        "summary": result.get("snippet", ""),
                        "relevance_score": 0.5  # Default score for Serper
                    })
                elif source == "mcp":
                    processed_result.update({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "summary": result.get("description", ""),
                        "relevance_score": result.get("relevance", 0.0)
                    })
                
                if processed_result["title"] and processed_result["url"]:
                    processed_results.append(processed_result)
                    
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
        
        return processed_results
    
    def _filter_math_results(self, results: List[Dict[str, Any]], 
                           original_question: str) -> List[Dict[str, Any]]:
        """Filter and rank results based on mathematical relevance"""
        
        # Educational domains (higher priority)
        educational_domains = [
            "khanacademy.org", "mathworld.wolfram.com", "brilliant.org",
            "mathisfun.com", "wikipedia.org", "mit.edu", "stanford.edu",
            "math.stackexchange.com", "mathpages.com"
        ]
        
        # Score and filter results
        scored_results = []
        
        for result in results:
            score = result.get("relevance_score", 0.0)
            
            # Boost educational domains
            url = result.get("url", "").lower()
            if any(domain in url for domain in educational_domains):
                score += 0.3
            
            # Boost results with mathematical terms in title
            title = result.get("title", "").lower()
            summary = result.get("summary", "").lower()
            
            math_terms = ["math", "equation", "formula", "solve", "calculate", 
                         "theorem", "proof", "example", "tutorial"]
            
            math_term_count = sum(1 for term in math_terms 
                                if term in title or term in summary)
            score += math_term_count * 0.1
            
            # Check relevance to original question
            question_words = set(original_question.lower().split())
            result_words = set((title + " " + summary).lower().split())
            
            overlap = len(question_words.intersection(result_words))
            if overlap > 0:
                score += overlap * 0.05
            
            result["final_score"] = score
            
            # Only include results with reasonable scores
            if score > 0.2:
                scored_results.append(result)
        
        # Sort by final score
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return scored_results
    
    async def search_specific_topic(self, topic: str, subtopic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for specific mathematical topics"""
        
        query = f"{topic} mathematics"
        if subtopic:
            query += f" {subtopic}"
        
        query += " tutorial examples problems"
        
        return await self.search_math_resources(query, topic)
    
    async def search_problem_examples(self, problem_type: str, difficulty: str = "medium") -> List[Dict[str, Any]]:
        """Search for similar problem examples"""
        
        query = f"{problem_type} {difficulty} math problems examples solutions"
        
        return await self.search_math_resources(query, problem_type)
    
    async def verify_solution_approach(self, approach: str, topic: str) -> List[Dict[str, Any]]:
        """Search for verification of a solution approach"""
        
        query = f"{approach} {topic} mathematics method verification"
        
        return await self.search_math_resources(query, topic)

# Global instances
mcp_manager = MCPManager()
web_search_service = WebSearchService()