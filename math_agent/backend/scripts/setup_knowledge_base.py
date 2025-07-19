#!/usr/bin/env python3
"""
Math Agent Knowledge Base Setup Script for Qdrant Cloud
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.data_processor import MathDatasetProcessor
from app.core.config import get_settings
from app.core.logging import setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

async def verify_qdrant_connection(processor: MathDatasetProcessor) -> bool:
    """Verify connection to Qdrant Cloud"""
    try:
        collections = await processor.vector_db.client.get_collections()
        logger.info(f"Connected to Qdrant Cloud. Existing collections: {[col.name for col in collections.collections]}")
        return True
    except Exception as e:
        logger.error(f"Qdrant connection verification failed: {e}")
        return False

async def initialize_processor() -> MathDatasetProcessor:
    """Initialize processor with proper async collection setup"""
    processor = MathDatasetProcessor()
    await processor._initialize_collection()  # Now properly awaited
    return processor

async def process_dataset(processor: MathDatasetProcessor) -> List[Dict[str, Any]]:
    """Process the complete dataset"""
    try:
        logger.info("Starting dataset processing pipeline")
        problems = await processor.process_complete_dataset()
        
        if not problems:
            logger.warning("No problems processed - falling back to sample dataset")
            problems = processor._create_sample_dataset()
        
        return problems
        
    except Exception as e:
        logger.error(f"Dataset processing failed: {e}")
        raise

def print_summary(problems: List[Dict[str, Any]], settings) -> None:
    """Print processing summary"""
    from collections import defaultdict
    
    category_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)
    
    for problem in problems:
        category_counts[problem.get('category', 'unknown')] += 1
        difficulty_counts[problem.get('difficulty', 'unknown')] += 1
    
    print("\n" + "=" * 60)
    print("📊 Processing Summary")
    print("=" * 60)
    print(f"🔢 Total Problems Processed: {len(problems)}")
    print(f"📂 Collection Name: {settings.QDRANT_COLLECTION_NAME}")
    print(f"🌐 Qdrant Cloud URL: {settings.QDRANT_CLOUD_URL}")
    
    print("\n📚 Problem Categories:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {category.title()}: {count}")
    
    print("\n⚖️ Difficulty Distribution:")
    for difficulty, count in sorted(difficulty_counts.items()):
        print(f"  • {difficulty.title()}: {count}")
    
    print("\n" + "=" * 60)
    print("🚀 Knowledge Base Setup Complete!")
    print("=" * 60)

async def main():
    """Main async setup function"""
    try:
        print("\n" + "=" * 60)
        print("🚀 Math Agent Knowledge Base Setup - Qdrant Cloud Edition")
        print("=" * 60)
        
        settings = get_settings()
        
        # Initialize processor with proper async handling
        print("\n🔧 Initializing processor...")
        processor = await initialize_processor()
        
        # Verify Qdrant connection
        print("\n🔌 Verifying Qdrant Cloud connection...")
        if not await verify_qdrant_connection(processor):
            print("❌ Failed to connect to Qdrant Cloud")
            print(f"Please verify your configuration in .env:")
            print(f"QDRANT_CLOUD_URL={settings.QDRANT_CLOUD_URL}")
            print("QDRANT_CLOUD_API_KEY=[set in .env]")
            sys.exit(1)
        
        print("✅ Successfully connected to Qdrant Cloud")
        
        # Process dataset
        print("\n🔄 Processing math problems...")
        problems = await process_dataset(processor)
        
        # Print summary and save results
        print_summary(problems, settings)
        output_file = processor.processed_data_path / "processed_problems.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved processed data to: {output_file}")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        print(f"\n❌ Fatal error during setup: {e}")
        sys.exit(1)

def run():
    """Synchronous entry point"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    run()