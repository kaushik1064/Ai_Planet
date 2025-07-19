from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from typing import List, Dict, Any, Optional
import asyncio

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, cloud_url: str, api_key: str, collection_name: str):
        self.cloud_url = cloud_url
        self.api_key = api_key
        self.collection_name = collection_name
        self.client = self._initialize_client()
    
    def _initialize_client(self) -> QdrantClient:
        """Initialize and verify Qdrant Cloud connection"""
        try:
            client = QdrantClient(
                url=self.cloud_url,
                api_key=self.api_key,
                timeout=30,
                prefer_grpc=True  # Recommended for better performance
            )
            
            # Verify connection
            client.get_collections()
            logger.info("Successfully connected to Qdrant Cloud")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise ConnectionError(f"Could not connect to Qdrant Cloud: {e}")

    async def ensure_collection_exists(self, vector_size: int):
        """Ensure target collection exists with proper configuration"""
        try:
            from qdrant_client.http.exceptions import UnexpectedResponse
            
            try:
                collection_info = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.get_collection(self.collection_name)
                )
                logger.info(f"Collection '{self.collection_name}' already exists")
                
                # Verify vector size matches
                if collection_info.config.params.vectors.size != vector_size:
                    raise ValueError(
                        f"Existing collection has vector size {collection_info.config.params.vectors.size} "
                        f"but expected {vector_size}"
                    )
                    
            except UnexpectedResponse as e:
                if e.status_code == 404:
                    # Collection doesn't exist, create it
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=models.VectorParams(
                                size=vector_size,
                                distance=models.Distance.COSINE
                            ),
                            optimizers_config=models.OptimizersConfigDiff(
                                indexing_threshold=0  # Ensure immediate indexing
                            )
                        )
                    )
                    logger.info(f"Created new collection '{self.collection_name}'")
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    async def add_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """Add batch of vectors to the collection"""
        try:
            points = [
                models.PointStruct(
                    id=vector["id"],
                    vector=vector["vector"],
                    payload=vector.get("payload", {})
                )
                for vector in vectors
            ]
            
            # Execute upsert with retry logic
            def _upsert_with_retry():
                return self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
            
            await asyncio.get_event_loop().run_in_executor(None, _upsert_with_retry)
            return True
            
        except Exception as e:
            logger.error(f"Error adding vectors: {str(e)[:200]}")  # Truncate long error messages
            return False

    async def search_similar(self, query_vector: List[float], limit: int = 5, **kwargs):
        """Search for similar vectors"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                    **kwargs
                )
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise