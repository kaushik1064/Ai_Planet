# Qdrant vector database
from typing import List, Dict, Any, Optional
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class VectorDatabase:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
    
    async def search_similar(self, query_vector: List[float], limit: int = 5, 
                           score_threshold: float = 0.7, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": limit,
                "score_threshold": score_threshold,
                "with_payload": True
            }
            
            if filters:
                search_params["query_filter"] = self._build_filter(filters)
            
            results = self.client.search(**search_params)
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary"""
        conditions = []
        
        for field, value in filters.items():
            if isinstance(value, list):
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(any=value))
                )
            else:
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
        
        return Filter(must=conditions)
    
    async def add_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """Add vectors to the collection"""
        try:
            points = [
                PointStruct(
                    id=vector["id"],
                    vector=vector["vector"],
                    payload=vector["payload"]
                )
                for vector in vectors
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}