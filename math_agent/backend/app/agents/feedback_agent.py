# Human-in-the-loop feedback
# Human-in-the-loop feedback
import asyncio
import sqlite3
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

from app.core.config import get_settings
from app.services.groq_service import groq_service

logger = logging.getLogger(__name__)
settings = get_settings()

class FeedbackAgent:
    def __init__(self):
        self.db_path = "data/feedback.db"
        self.learning_insights_cache = {}
        self.improvement_patterns = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize the feedback database"""
        try:
            # Create data directory if it doesn't exist
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    feedback_text TEXT,
                    user_suggestions TEXT,
                    user_id TEXT,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create learning insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_insights (
                    id TEXT PRIMARY KEY,
                    insight_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    impact_score REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    applied BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create improvement patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS improvement_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                    effectiveness REAL DEFAULT 0.0
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing feedback database: {e}")
    
    async def collect_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect and store user feedback"""
        
        try:
            feedback_id = str(uuid.uuid4())
            
            # Store feedback in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO feedback (
                    id, question, solution, feedback_type, rating, 
                    feedback_text, user_suggestions, user_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback_id,
                feedback_data.get("question", ""),
                feedback_data.get("solution", ""),
                feedback_data.get("feedback_type", "general"),
                feedback_data.get("rating", 3),
                feedback_data.get("feedback_text", ""),
                feedback_data.get("user_suggestions", ""),
                feedback_data.get("user_id", ""),
                json.dumps(feedback_data.get("metadata", {}))
            ))
            
            conn.commit()
            conn.close()
            
            # Generate immediate insights
            immediate_insights = await self._generate_immediate_insights(feedback_data)
            
            # Check if improvement should be triggered
            should_improve = self._should_trigger_improvement(feedback_data)
            
            return {
                "success": True,
                "feedback_id": feedback_id,
                "immediate_insights": immediate_insights,
                "should_trigger_improvement": should_improve
            }
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_immediate_insights(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate immediate insights from feedback"""
        
        insights = {
            "feedback_summary": self._summarize_feedback(feedback_data),
            "improvement_suggestions": [],
            "pattern_detection": {}
        }
        
        rating = feedback_data.get("rating", 3)
        feedback_text = feedback_data.get("feedback_text", "")
        
        # Rating-based insights
        if rating <= 2:
            insights["improvement_suggestions"].append("Critical: Solution needs major revision")
        elif rating == 3:
            insights["improvement_suggestions"].append("Moderate: Solution has room for improvement")
        elif rating >= 4:
            insights["improvement_suggestions"].append("Good: Minor refinements may be beneficial")
        
        # Text-based insights
        if feedback_text:
            text_insights = await self._analyze_feedback_text(feedback_text)
            insights["improvement_suggestions"].extend(text_insights)
        
        # Pattern detection
        patterns = await self._detect_feedback_patterns(feedback_data)
        insights["pattern_detection"] = patterns
        
        return insights
    
    def _summarize_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Create a brief summary of the feedback"""
        
        rating = feedback_data.get("rating", 3)
        feedback_type = feedback_data.get("feedback_type", "general")
        
        summary = f"Rating: {rating}/5, Type: {feedback_type}"
        
        if feedback_data.get("feedback_text"):
            text_length = len(feedback_data["feedback_text"])
            summary += f", Detailed feedback provided ({text_length} chars)"
        
        return summary
    
    async def _analyze_feedback_text(self, feedback_text: str) -> List[str]:
        """Analyze feedback text for specific improvement suggestions"""
        
        try:
            analysis_prompt = f"""
            Analyze this user feedback about a mathematical solution and provide specific improvement suggestions:
            
            Feedback: {feedback_text}
            
            Identify specific issues and provide actionable suggestions for improvement.
            Focus on: correctness, clarity, completeness, methodology, presentation.
            
            Format as a simple list of suggestions.
            """
            
            response = await groq_service.client.chat.completions.create(
                model=groq_service.model,
                messages=[
                    {"role": "system", "content": "You are an expert in educational feedback analysis."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            suggestions_text = response.choices[0].message.content
            
            # Parse suggestions from response
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove bullet points and numbering
                    clean_line = line.lstrip('•-*123456789. ')
                    if clean_line:
                        suggestions.append(clean_line)
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Error analyzing feedback text: {e}")
            return ["Review solution based on user feedback"]
    
    async def _detect_feedback_patterns(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns in feedback for learning"""
        
        patterns = {
            "common_issues": [],
            "recurring_themes": [],
            "improvement_areas": []
        }
        
        try:
            # Get recent feedback for pattern analysis
            recent_feedback = self._get_recent_feedback(days=30)
            
            if len(recent_feedback) >= 5:  # Need minimum feedback for pattern detection
                # Analyze common rating patterns
                ratings = [f.get("rating", 3) for f in recent_feedback]
                avg_rating = sum(ratings) / len(ratings)
                
                if avg_rating < 3.5:
                    patterns["common_issues"].append("Consistently lower ratings indicate systemic issues")
                
                # Analyze feedback text patterns
                feedback_texts = [f.get("feedback_text", "") for f in recent_feedback if f.get("feedback_text")]
                
                if feedback_texts:
                    common_words = self._extract_common_feedback_terms(feedback_texts)
                    patterns["recurring_themes"] = common_words
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    def _extract_common_feedback_terms(self, feedback_texts: List[str]) -> List[str]:
        """Extract common terms from feedback texts"""
        
        import re
        from collections import Counter
        
        # Combine all feedback texts
        combined_text = " ".join(feedback_texts).lower()
        
        # Extract meaningful words (excluding common stop words)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
        
        words = re.findall(r'\b[a-z]{3,}\b', combined_text)
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequency
        word_counts = Counter(filtered_words)
        
        # Return top 5 most common words
        return [word for word, count in word_counts.most_common(5) if count > 1]
    
    def _should_trigger_improvement(self, feedback_data: Dict[str, Any]) -> bool:
        """Determine if feedback should trigger automatic improvement"""
        
        rating = feedback_data.get("rating", 3)
        feedback_text = feedback_data.get("feedback_text", "")
        
        # Trigger on low ratings
        if rating <= 2:
            return True
        
        # Trigger on specific negative keywords
        negative_keywords = ["wrong", "incorrect", "error", "mistake", "unclear", "confusing"]
        if any(keyword in feedback_text.lower() for keyword in negative_keywords):
            return True
        
        return False
    
    def _get_recent_feedback(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent feedback from database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT question, solution, feedback_type, rating, feedback_text, 
                       user_suggestions, metadata, timestamp
                FROM feedback 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            """.format(days))
            
            rows = cursor.fetchall()
            conn.close()
            
            feedback_list = []
            for row in rows:
                feedback_list.append({
                    "question": row[0],
                    "solution": row[1],
                    "feedback_type": row[2],
                    "rating": row[3],
                    "feedback_text": row[4],
                    "user_suggestions": row[5],
                    "metadata": json.loads(row[6]) if row[6] else {},
                    "timestamp": row[7]
                })
            
            return feedback_list
            
        except Exception as e:
            logger.error(f"Error getting recent feedback: {e}")
            return []
    
    def get_feedback_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback statistics for the specified period"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total feedback count
            cursor.execute("""
                SELECT COUNT(*) FROM feedback 
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days))
            total_feedback = cursor.fetchone()[0]
            
            # Average rating
            cursor.execute("""
                SELECT AVG(rating) FROM feedback 
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days))
            avg_rating = cursor.fetchone()[0] or 0.0
            
            # Positive vs negative feedback
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as negative
                FROM feedback 
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days))
            pos_neg = cursor.fetchone()
            positive_feedback = pos_neg[0] or 0
            negative_feedback = pos_neg[1] or 0
            
            # Feedback by type
            cursor.execute("""
                SELECT feedback_type, COUNT(*), AVG(rating)
                FROM feedback 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY feedback_type
            """.format(days))
            feedback_by_type = [
                {"type": row[0], "count": row[1], "avg_rating": row[2]}
                for row in cursor.fetchall()
            ]
            
            # Daily trends (last 7 days)
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*), AVG(rating)
                FROM feedback 
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            """)
            daily_trends = [
                {"date": row[0], "count": row[1], "avg_rating": row[2]}
                for row in cursor.fetchall()
            ]
            
            conn.close()
            
            return {
                "period_days": days,
                "total_feedback": total_feedback,
                "average_rating": round(avg_rating, 2),
                "positive_feedback": positive_feedback,
                "negative_feedback": negative_feedback,
                "feedback_by_type": feedback_by_type,
                "daily_trends": daily_trends
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {
                "period_days": days,
                "total_feedback": 0,
                "average_rating": 0.0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "feedback_by_type": [],
                "daily_trends": []
            }
    
    async def generate_learning_insights(self) -> Dict[str, Any]:
        """Generate learning insights from accumulated feedback"""
        
        try:
            recent_feedback = self._get_recent_feedback(days=30)
            
            if not recent_feedback:
                return {
                    "insights": ["Insufficient feedback data for meaningful insights"],
                    "recommendations": ["Collect more user feedback to enable learning"]
                }
            
            insights = []
            recommendations = []
            
            # Performance insights
            ratings = [f["rating"] for f in recent_feedback]
            avg_rating = sum(ratings) / len(ratings)
            
            if avg_rating < 3.0:
                insights.append(f"Average rating is low ({avg_rating:.1f}/5) - significant improvement needed")
                recommendations.append("Focus on solution quality and clarity")
            elif avg_rating >= 4.0:
                insights.append(f"Average rating is good ({avg_rating:.1f}/5) - maintain current quality")
                recommendations.append("Continue current approach with minor optimizations")
            
            # Common issues analysis
            feedback_texts = [f["feedback_text"] for f in recent_feedback if f["feedback_text"]]
            if feedback_texts:
                common_issues = await self._analyze_common_issues(feedback_texts)
                insights.extend(common_issues)
            
            # Type-specific insights
            type_stats = {}
            for feedback in recent_feedback:
                fb_type = feedback["feedback_type"]
                if fb_type not in type_stats:
                    type_stats[fb_type] = []
                type_stats[fb_type].append(feedback["rating"])
            
            for fb_type, ratings in type_stats.items():
                avg_type_rating = sum(ratings) / len(ratings)
                if avg_type_rating < 3.5:
                    insights.append(f"{fb_type.title()} feedback shows issues (avg: {avg_type_rating:.1f}/5)")
                    recommendations.append(f"Improve {fb_type} aspects of solutions")
            
            return {
                "insights": insights,
                "recommendations": recommendations,
                "data_points": len(recent_feedback),
                "analysis_period": "30 days"
            }
            
        except Exception as e:
            logger.error(f"Error generating learning insights: {e}")
            return {
                "insights": ["Error analyzing feedback data"],
                "recommendations": ["Check system logs for detailed error information"]
            }
    
    async def _analyze_common_issues(self, feedback_texts: List[str]) -> List[str]:
        """Analyze common issues from feedback texts"""
        
        try:
            combined_feedback = "\n".join(feedback_texts)
            
            analysis_prompt = f"""
            Analyze these user feedback comments and identify the top 3 most common issues:
            
            {combined_feedback}
            
            Provide insights in this format:
            1. [Most common issue]
            2. [Second most common issue]  
            3. [Third most common issue]
            
            Focus on actionable insights about solution quality, clarity, accuracy, etc.
            """
            
            response = await groq_service.client.chat.completions.create(
                model=groq_service.model,
                messages=[
                    {"role": "system", "content": "You are an expert in educational feedback analysis."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract numbered insights
            insights = []
            import re
            matches = re.findall(r'\d+\.\s*(.+)', analysis_text)
            for match in matches[:3]:  # Top 3 issues
                insights.append(f"Common issue: {match.strip()}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing common issues: {e}")
            return ["Unable to analyze common issues from feedback"]
    
    async def apply_learning_from_feedback(self) -> Dict[str, Any]:
        """Apply learning from feedback to improve the system"""
        
        try:
            # Get unprocessed feedback
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, question, solution, feedback_type, rating, feedback_text
                FROM feedback 
                WHERE processed = FALSE
                ORDER BY timestamp DESC
                LIMIT 20
            """)
            
            unprocessed = cursor.fetchall()
            
            if not unprocessed:
                return {
                    "success": True,
                    "message": "No new feedback to process",
                    "processed_count": 0
                }
            
            learning_outcomes = []
            
            for feedback_row in unprocessed:
                feedback_id, question, solution, fb_type, rating, feedback_text = feedback_row
                
                # Generate learning outcome
                if rating <= 2 and feedback_text:
                    outcome = await self._generate_learning_outcome(question, solution, feedback_text)
                    if outcome:
                        learning_outcomes.append(outcome)
                
                # Mark as processed
                cursor.execute("UPDATE feedback SET processed = TRUE WHERE id = ?", (feedback_id,))
            
            # Store learning insights
            for outcome in learning_outcomes:
                insight_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO learning_insights (id, insight_type, content, confidence)
                    VALUES (?, ?, ?, ?)
                """, (insight_id, outcome["type"], outcome["content"], outcome["confidence"]))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "processed_count": len(unprocessed),
                "learning_outcomes": len(learning_outcomes),
                "insights_generated": learning_outcomes
            }
            
        except Exception as e:
            logger.error(f"Error applying learning from feedback: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_learning_outcome(self, question: str, solution: str, 
                                       feedback_text: str) -> Optional[Dict[str, Any]]:
        """Generate a learning outcome from negative feedback"""
        
        try:
            learning_prompt = f"""
            Question: {question}
            Solution: {solution}
            User Feedback: {feedback_text}
            
            Based on this negative feedback, generate a specific learning insight that could improve future solutions.
            
            Format:
            Type: [issue_type]
            Insight: [specific learning point]
            Action: [how to apply this learning]
            """
            
            response = await groq_service.client.chat.completions.create(
                model=groq_service.model,
                messages=[
                    {"role": "system", "content": "You are an AI learning system that improves from feedback."},
                    {"role": "user", "content": learning_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            learning_text = response.choices[0].message.content
            
            # Parse the learning outcome
            import re
            type_match = re.search(r'Type:\s*(.+)', learning_text)
            insight_match = re.search(r'Insight:\s*(.+)', learning_text)
            action_match = re.search(r'Action:\s*(.+)', learning_text)
            
            if insight_match:
                return {
                    "type": type_match.group(1).strip() if type_match else "general",
                    "content": insight_match.group(1).strip(),
                    "action": action_match.group(1).strip() if action_match else "",
                    "confidence": 0.7
                }
            
        except Exception as e:
            logger.error(f"Error generating learning outcome: {e}")
        
        return None
    
    async def get_improvement_suggestions_for_query(self, question: str) -> List[str]:
        """Get improvement suggestions for a specific type of question"""
        
        try:
            # Find similar questions in feedback history
            similar_feedback = []
            recent_feedback = self._get_recent_feedback(days=90)
            
            question_lower = question.lower()
            for feedback in recent_feedback:
                if any(word in feedback["question"].lower() for word in question_lower.split()[:3]):
                    if feedback["rating"] <= 3 and feedback["feedback_text"]:
                        similar_feedback.append(feedback)
            
            if not similar_feedback:
                return ["No specific improvement suggestions available for this type of question"]
            
            # Analyze patterns in similar feedback
            suggestions = []
            common_issues = [f["feedback_text"] for f in similar_feedback]
            
            if common_issues:
                analysis = await self._analyze_common_issues(common_issues)
                suggestions.extend([f"Consider: {issue}" for issue in analysis])
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Error getting improvement suggestions: {e}")
            return ["Error retrieving improvement suggestions"]

# DSPy Integration (optional advanced feature)
class DSPyIntegration:
    def __init__(self):
        self.dspy_available = False
        try:
            import dspy
            self.dspy_available = True
            logger.info("DSPy integration available")
        except ImportError:
            logger.info("DSPy not available - advanced feedback analysis disabled")
    
    async def analyze_feedback_with_dspy(self, feedback_text: str, question: str, 
                                       solution: str) -> Dict[str, Any]:
        """Advanced feedback analysis using DSPy (if available)"""
        
        if not self.dspy_available:
            return {
                "success": False,
                "error": "DSPy integration not available"
            }
        
        try:
            # This would use DSPy for advanced feedback analysis
            # Implementation would depend on specific DSPy setup
            return {
                "success": True,
                "analysis": "Advanced DSPy analysis would go here",
                "suggestions": ["DSPy-powered improvement suggestions"]
            }
        except Exception as e:
            logger.error(f"DSPy analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Global instances
feedback_agent = FeedbackAgent()
dspy_integration = DSPyIntegration()