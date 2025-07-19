# Generate embeddings
# Generate embeddings
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
import torch

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class EmbeddingGenerator:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Embedding model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            # Fallback to a simpler model
            try:
                logger.info("Falling back to all-MiniLM-L6-v2 model")
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                self.model_name = 'all-MiniLM-L6-v2'
                logger.info("Fallback model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                raise Exception("Could not load any embedding model")
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        if not self.model:
            logger.error("Embedding model not initialized")
            return None
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode(processed_text, convert_to_tensor=False)
            
            # Convert to list if numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts in batches"""
        if not self.model:
            logger.error("Embedding model not initialized")
            return [None] * len(texts)
        
        try:
            # Preprocess all texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            embeddings = []
            
            # Process in batches
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                
                try:
                    batch_embeddings = self.model.encode(batch, convert_to_tensor=False)
                    
                    # Convert each embedding to list
                    for embedding in batch_embeddings:
                        if isinstance(embedding, np.ndarray):
                            embeddings.append(embedding.tolist())
                        else:
                            embeddings.append(embedding)
                            
                except Exception as batch_error:
                    logger.error(f"Error in batch processing: {batch_error}")
                    # Add None for failed batch
                    embeddings.extend([None] * len(batch))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embeddings"""
        if not text:
            return ""
        
        # Basic cleaning
        processed = text.strip()
        
        # Remove excessive whitespace
        import re
        processed = re.sub(r'\s+', ' ', processed)
        
        # Limit length (most models have token limits)
        max_length = 512  # Conservative limit for most sentence transformers
        if len(processed) > max_length:
            processed = processed[:max_length]
        
        return processed
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from the model"""
        if not self.model:
            return 0
        
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.model.encode("test", convert_to_tensor=False)
            if isinstance(test_embedding, np.ndarray):
                return test_embedding.shape[0]
            elif isinstance(test_embedding, list):
                return len(test_embedding)
            else:
                return 0
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            return 0
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: List[float], 
                         candidate_embeddings: List[List[float]], 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar embeddings to a query embedding"""
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.calculate_similarity(query_embedding, candidate)
                similarities.append({
                    "index": i,
                    "similarity": similarity
                })
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar embeddings: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "is_loaded": self.model is not None,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown') if self.model else 'unknown'
        }