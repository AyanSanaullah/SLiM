"""
Training Database Manager
Manages local SQLite database for storing training metrics and similarity scores
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

logger = logging.getLogger(__name__)

class TrainingDatabase:
    """
    Local SQLite database for storing training metrics and similarity scores
    """
    
    def __init__(self, db_path: str = "data/training_metrics.db"):
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Training sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS training_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        status TEXT DEFAULT 'active',
                        total_cycles INTEGER DEFAULT 0,
                        metadata TEXT
                    )
                """)
                
                # Training cycles table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS training_cycles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        cycle_number INTEGER NOT NULL,
                        prompt TEXT NOT NULL,
                        expected_answer TEXT NOT NULL,
                        model_response TEXT NOT NULL,
                        similarity_score REAL NOT NULL,
                        quality_label TEXT NOT NULL,
                        model_confidence REAL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
                    )
                """)
                
                # Similarity scores aggregation table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS similarity_aggregations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        avg_similarity REAL NOT NULL,
                        max_similarity REAL NOT NULL,
                        min_similarity REAL NOT NULL,
                        high_quality_count INTEGER NOT NULL,
                        medium_quality_count INTEGER NOT NULL,
                        low_quality_count INTEGER NOT NULL,
                        total_cycles INTEGER NOT NULL,
                        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON training_sessions(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_cycles_session_id ON training_cycles(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_cycles_similarity ON training_cycles(similarity_score)")
                
                conn.commit()
                logger.info(f"Training database initialized at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def start_training_session(self, user_id: str, model_type: str, metadata: Dict[str, Any] = None) -> str:
        """
        Start a new training session
        
        Args:
            user_id: User identifier
            model_type: Type of model being trained (basic, advanced, etc.)
            metadata: Additional session metadata
            
        Returns:
            session_id: Unique session identifier
        """
        try:
            session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO training_sessions (user_id, session_id, model_type, metadata)
                    VALUES (?, ?, ?, ?)
                """, (user_id, session_id, model_type, json.dumps(metadata or {})))
                conn.commit()
                
            logger.info(f"Started training session {session_id} for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting training session: {e}")
            raise
    
    def record_training_cycle(self, session_id: str, cycle_data: Dict[str, Any]):
        """
        Record a training cycle with similarity score
        
        Args:
            session_id: Training session identifier
            cycle_data: Dictionary containing cycle information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current cycle number
                cursor.execute("""
                    SELECT COALESCE(MAX(cycle_number), 0) + 1 
                    FROM training_cycles 
                    WHERE session_id = ?
                """, (session_id,))
                cycle_number = cursor.fetchone()[0]
                
                # Insert cycle data
                cursor.execute("""
                    INSERT INTO training_cycles (
                        session_id, cycle_number, prompt, expected_answer, 
                        model_response, similarity_score, quality_label, 
                        model_confidence, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    cycle_number,
                    cycle_data.get('prompt', ''),
                    cycle_data.get('expected_answer', ''),
                    cycle_data.get('model_response', ''),
                    cycle_data.get('similarity_score', 0.0),
                    cycle_data.get('quality_label', 'UNKNOWN'),
                    cycle_data.get('model_confidence', 0.0),
                    json.dumps(cycle_data.get('metadata', {}))
                ))
                
                # Update session cycle count
                cursor.execute("""
                    UPDATE training_sessions 
                    SET total_cycles = total_cycles + 1 
                    WHERE session_id = ?
                """, (session_id,))
                
                conn.commit()
                
            logger.info(f"Recorded cycle {cycle_number} for session {session_id} - Similarity: {cycle_data.get('similarity_score', 0):.3f}")
            
        except Exception as e:
            logger.error(f"Error recording training cycle: {e}")
            raise
    
    def complete_training_session(self, session_id: str):
        """
        Complete a training session and calculate aggregations
        
        Args:
            session_id: Training session identifier
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate aggregations
                cursor.execute("""
                    SELECT 
                        AVG(similarity_score) as avg_similarity,
                        MAX(similarity_score) as max_similarity,
                        MIN(similarity_score) as min_similarity,
                        COUNT(*) as total_cycles,
                        SUM(CASE WHEN quality_label = 'HIGH' THEN 1 ELSE 0 END) as high_quality,
                        SUM(CASE WHEN quality_label = 'MEDIUM' THEN 1 ELSE 0 END) as medium_quality,
                        SUM(CASE WHEN quality_label = 'LOW' THEN 1 ELSE 0 END) as low_quality
                    FROM training_cycles 
                    WHERE session_id = ?
                """, (session_id,))
                
                result = cursor.fetchone()
                if result and result[3] > 0:  # total_cycles > 0
                    # Insert aggregation (add session_id to the result tuple)
                    cursor.execute("""
                        INSERT INTO similarity_aggregations (
                            session_id, avg_similarity, max_similarity, min_similarity,
                            high_quality_count, medium_quality_count, low_quality_count, total_cycles
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (session_id,) + result)
                
                # Mark session as completed
                cursor.execute("""
                    UPDATE training_sessions 
                    SET completed_at = CURRENT_TIMESTAMP, status = 'completed'
                    WHERE session_id = ?
                """, (session_id,))
                
                conn.commit()
                
            logger.info(f"Completed training session {session_id}")
            
        except Exception as e:
            logger.error(f"Error completing training session: {e}")
            raise
    
    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get metrics for a training session
        
        Args:
            session_id: Training session identifier
            
        Returns:
            Dictionary with session metrics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get session info
                cursor.execute("""
                    SELECT user_id, model_type, started_at, completed_at, status, total_cycles
                    FROM training_sessions 
                    WHERE session_id = ?
                """, (session_id,))
                
                session_info = cursor.fetchone()
                if not session_info:
                    return {}
                
                # Get aggregations
                cursor.execute("""
                    SELECT avg_similarity, max_similarity, min_similarity,
                           high_quality_count, medium_quality_count, low_quality_count, total_cycles
                    FROM similarity_aggregations 
                    WHERE session_id = ?
                    ORDER BY calculated_at DESC LIMIT 1
                """, (session_id,))
                
                agg_info = cursor.fetchone()
                
                return {
                    'session_id': session_id,
                    'user_id': session_info[0],
                    'model_type': session_info[1],
                    'started_at': session_info[2],
                    'completed_at': session_info[3],
                    'status': session_info[4],
                    'total_cycles': session_info[5],
                    'metrics': {
                        'avg_similarity': agg_info[0] if agg_info else 0,
                        'max_similarity': agg_info[1] if agg_info else 0,
                        'min_similarity': agg_info[2] if agg_info else 0,
                        'high_quality_count': agg_info[3] if agg_info else 0,
                        'medium_quality_count': agg_info[4] if agg_info else 0,
                        'low_quality_count': agg_info[5] if agg_info else 0,
                        'total_cycles': agg_info[6] if agg_info else 0
                    } if agg_info else {}
                }
                
        except Exception as e:
            logger.error(f"Error getting session metrics: {e}")
            return {}
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all training sessions for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id, model_type, started_at, completed_at, status, total_cycles
                    FROM training_sessions 
                    WHERE user_id = ?
                    ORDER BY started_at DESC
                """, (user_id,))
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        'session_id': row[0],
                        'model_type': row[1],
                        'started_at': row[2],
                        'completed_at': row[3],
                        'status': row[4],
                        'total_cycles': row[5]
                    })
                
                return sessions
                
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
