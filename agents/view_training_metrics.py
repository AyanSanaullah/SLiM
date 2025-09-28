#!/usr/bin/env python3
"""
Training Metrics Viewer
Script to view and analyze training metrics from the local database
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adk_agents.training_database import TrainingDatabase

def print_separator(title: str = ""):
    """Print a separator line with optional title"""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("="*60)

def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp_str

def display_session_summary(db: TrainingDatabase):
    """Display summary of all training sessions"""
    print_separator("TRAINING SESSIONS SUMMARY")
    
    # Get all sessions
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, session_id, model_type, started_at, completed_at, 
                   status, total_cycles
            FROM training_sessions 
            ORDER BY started_at DESC
        """)
        
        sessions = cursor.fetchall()
        
        if not sessions:
            print("No training sessions found.")
            return
        
        print(f"{'User ID':<20} {'Session ID':<25} {'Model Type':<15} {'Status':<10} {'Cycles':<8} {'Started':<20}")
        print("-" * 110)
        
        for session in sessions:
            user_id, session_id, model_type, started_at, completed_at, status, total_cycles = session
            started_formatted = format_timestamp(started_at) if started_at else "N/A"
            
            print(f"{user_id:<20} {session_id:<25} {model_type:<15} {status:<10} {total_cycles:<8} {started_formatted:<20}")

def display_session_details(db: TrainingDatabase, session_id: str):
    """Display detailed information for a specific session"""
    print_separator(f"SESSION DETAILS: {session_id}")
    
    # Get session metrics
    metrics = db.get_session_metrics(session_id)
    
    if not metrics:
        print(f"Session {session_id} not found.")
        return
    
    print(f"User ID: {metrics['user_id']}")
    print(f"Model Type: {metrics['model_type']}")
    print(f"Status: {metrics['status']}")
    print(f"Started: {format_timestamp(metrics['started_at'])}")
    print(f"Completed: {format_timestamp(metrics['completed_at']) if metrics['completed_at'] else 'In Progress'}")
    print(f"Total Cycles: {metrics['total_cycles']}")
    
    if metrics.get('metrics'):
        m = metrics['metrics']
        print(f"\nMetrics:")
        print(f"  Average Similarity: {m.get('avg_similarity', 0):.3f}")
        print(f"  Max Similarity: {m.get('max_similarity', 0):.3f}")
        print(f"  Min Similarity: {m.get('min_similarity', 0):.3f}")
        print(f"  High Quality Count: {m.get('high_quality_count', 0)}")
        print(f"  Medium Quality Count: {m.get('medium_quality_count', 0)}")
        print(f"  Low Quality Count: {m.get('low_quality_count', 0)}")
    
    # Get individual cycles
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT cycle_number, prompt, model_response, similarity_score, 
                   quality_label, model_confidence, timestamp
            FROM training_cycles 
            WHERE session_id = ?
            ORDER BY cycle_number
        """, (session_id,))
        
        cycles = cursor.fetchall()
        
        if cycles:
            print(f"\nTraining Cycles ({len(cycles)} total):")
            print(f"{'#':<4} {'Similarity':<12} {'Quality':<8} {'Confidence':<12} {'Prompt':<40}")
            print("-" * 80)
            
            for cycle in cycles[:10]:  # Show first 10 cycles
                cycle_num, prompt, response, similarity, quality, confidence, timestamp = cycle
                prompt_short = prompt[:37] + "..." if len(prompt) > 40 else prompt
                print(f"{cycle_num:<4} {similarity:<12.3f} {quality:<8} {confidence or 0:<12.3f} {prompt_short:<40}")
            
            if len(cycles) > 10:
                print(f"... and {len(cycles) - 10} more cycles")

def display_user_sessions(db: TrainingDatabase, user_id: str):
    """Display all sessions for a specific user"""
    print_separator(f"SESSIONS FOR USER: {user_id}")
    
    sessions = db.get_user_sessions(user_id)
    
    if not sessions:
        print(f"No sessions found for user {user_id}")
        return
    
    print(f"{'Session ID':<30} {'Model Type':<15} {'Status':<10} {'Cycles':<8} {'Started':<20}")
    print("-" * 90)
    
    for session in sessions:
        started_formatted = format_timestamp(session['started_at']) if session['started_at'] else "N/A"
        print(f"{session['session_id']:<30} {session['model_type']:<15} {session['status']:<10} {session['total_cycles']:<8} {started_formatted:<20}")

def display_similarity_stats(db: TrainingDatabase):
    """Display overall similarity statistics"""
    print_separator("SIMILARITY STATISTICS")
    
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_cycles,
                AVG(similarity_score) as avg_similarity,
                MAX(similarity_score) as max_similarity,
                MIN(similarity_score) as min_similarity,
                COUNT(CASE WHEN quality_label = 'HIGH' THEN 1 END) as high_quality,
                COUNT(CASE WHEN quality_label = 'MEDIUM' THEN 1 END) as medium_quality,
                COUNT(CASE WHEN quality_label = 'LOW' THEN 1 END) as low_quality
            FROM training_cycles
        """)
        
        stats = cursor.fetchone()
        
        if stats and stats[0] > 0:
            total, avg, max_sim, min_sim, high, medium, low = stats
            
            print(f"Total Training Cycles: {total}")
            print(f"Average Similarity: {avg:.3f}")
            print(f"Max Similarity: {max_sim:.3f}")
            print(f"Min Similarity: {min_sim:.3f}")
            print(f"\nQuality Distribution:")
            print(f"  High Quality (≥80%): {high} ({high/total*100:.1f}%)")
            print(f"  Medium Quality (50-79%): {medium} ({medium/total*100:.1f}%)")
            print(f"  Low Quality (<50%): {low} ({low/total*100:.1f}%)")
        else:
            print("No training cycles found.")

def main():
    """Main function"""
    db = TrainingDatabase()
    
    if len(sys.argv) < 2:
        print("Training Metrics Viewer")
        print("Usage:")
        print("  python view_training_metrics.py summary              - Show all sessions summary")
        print("  python view_training_metrics.py session <session_id> - Show session details")
        print("  python view_training_metrics.py user <user_id>       - Show user sessions")
        print("  python view_training_metrics.py stats                - Show similarity statistics")
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command == "summary":
            display_session_summary(db)
        
        elif command == "session":
            if len(sys.argv) < 3:
                print("Please provide session_id")
                return
            session_id = sys.argv[2]
            display_session_details(db, session_id)
        
        elif command == "user":
            if len(sys.argv) < 3:
                print("Please provide user_id")
                return
            user_id = sys.argv[2]
            display_user_sessions(db, user_id)
        
        elif command == "stats":
            display_similarity_stats(db)
        
        else:
            print(f"Unknown command: {command}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
