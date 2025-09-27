#!/usr/bin/env python3
"""
Simple script to list existing agents in the project
"""

import requests
import json

def list_agents():
    """List all active agents"""
    base_url = "http://localhost:8080"
    
    try:
        # Check if server is running
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Server is not running. Run: python3 app.py")
            return
        
        print("✅ Server is working!")
        
        # List agents
        response = requests.get(f"{base_url}/api/v1/agents")
        
        if response.status_code == 200:
            data = response.json()
            total = data.get('total_users', 0)
            
            print(f"\n📋 Total active agents: {total}")
            
            if total == 0:
                print("   No agents found.")
                print("   To create an agent, use:")
                print("   python3 quick_example.py")
            else:
                print("\n🤖 Agents found:")
                print("-" * 50)
                
                users = data.get('users', {})
                for user_id, info in users.items():
                    status = info.get('status', 'unknown')
                    created_at = info.get('created_at', 'unknown')
                    base_model = info.get('base_model', 'unknown')
                    
                    # Format date
                    if created_at != 'unknown':
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            created_at = dt.strftime('%d/%m/%Y %H:%M')
                        except:
                            pass
                    
                    print(f"👤 User ID: {user_id}")
                    print(f"   Status: {status}")
                    print(f"   Base Model: {base_model}")
                    print(f"   Created at: {created_at}")
                    
                    if 'model_ready_at' in info:
                        print(f"   Model ready at: {info['model_ready_at']}")
                    
                    if 'training_data_size' in info:
                        print(f"   Data size: {info['training_data_size']} characters")
                    
                    print("-" * 50)
                
                print(f"\n💡 To ask questions to an agent:")
                print(f"   curl -X POST {base_url}/api/v1/agents/USER_ID/inference \\")
                print(f"     -H 'Content-Type: application/json' \\")
                print(f"     -d '{{\"prompt\": \"Your question here\"}}'")
        
        else:
            error_data = response.json()
            print(f"❌ Error listing agents: {error_data.get('error', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server.")
        print("   Make sure the server is running:")
        print("   python3 app.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🔍 Listing existing agents...")
    print("=" * 40)
    list_agents()
