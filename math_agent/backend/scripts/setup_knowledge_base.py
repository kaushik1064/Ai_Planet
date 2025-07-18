# Initialize knowledge base
"""
Script to set up the knowledge base from DeepMind math dataset JSON files
Uses first 50 problems from each file for efficient processing
Usage: python scripts/setup_knowledge_base.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.data_processor import MathDatasetProcessor
from app.core.logging import setup_logging
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

async def main():
    """Main setup function"""
    
    print("🚀 Starting Math Agent Knowledge Base Setup")
    print("=" * 60)
    print("🎯 Strategy: First 50 problems from each JSON file")
    print("💡 This provides diverse coverage while keeping processing efficient")
    print("🧠 LLM will handle the mathematical reasoning for remaining complexity")
    print("=" * 60)
    
    # Check if raw data directory exists
    raw_data_path = Path("data/raw")
    if not raw_data_path.exists():
        print(f"❌ Raw data directory not found: {raw_data_path}")
        print("Please ensure you have placed the 238 JSON files from DeepMind math dataset in data/raw/")
        sys.exit(1)
    
    # Count JSON files
    json_files = list(raw_data_path.glob("*.json"))
    print(f"📁 Found {len(json_files)} JSON files in {raw_data_path}")
    
    if len(json_files) == 0:
        print("❌ No JSON files found in raw data directory")
        print("Please place the DeepMind math dataset JSON files in data/raw/")
        sys.exit(1)
    
    expected_problems = len(json_files) * 50
    print(f"📊 Expected to process ~{expected_problems} problems total (50 per file)")
    print(f"⚡ This creates a focused but representative knowledge base")
    
    # Initialize processor
    processor = MathDatasetProcessor()
    
    try:
        # Process the complete dataset
        print("\n🔄 Processing dataset with first 50 problems per file...")
        print("📈 Generating embeddings for vector similarity search...")
        print("💾 Creating knowledge base for intelligent routing...")
        
        processed_problems = processor.process_complete_dataset()
        
        print(f"\n✅ Successfully processed {len(processed_problems)} problems!")
        print(f"📊 Knowledge base saved to: data/processed/processed_math_problems.json")
        print(f"📈 Summary statistics saved to: data/processed/dataset_summary.json")
        print(f"🔍 Vector embeddings stored in Qdrant collection: {processor.qdrant_client._collection_name}")
        
        # Display summary
        print(f"\n📋 Processing Summary:")
        print(f"   • Total files processed: {len(json_files)}")
        print(f"   • Problems per file: 50 (first 50 from each)")
        print(f"   • Total problems in KB: {len(processed_problems)}")
        print(f"   • Vector database: Ready for similarity search")
        print(f"   • LLM integration: Groq API for mathematical reasoning")
        
        print("\n" + "=" * 60)
        print("🎉 Knowledge base setup completed successfully!")
        print("🚀 System ready for intelligent mathematical problem solving")
        print("💡 The focused approach ensures quality over quantity")
        print("🧠 LLM capabilities handle the full mathematical complexity")
        print("\nYou can now start the Math Agent application.")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        print(f"\n❌ Setup failed: {e}")
        print("🔧 Check logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())