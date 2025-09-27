#!/usr/bin/env python3
"""
Help script that shows all available commands in the project
"""

import sys
import requests

def show_help():
    """Shows help menu with all commands"""
    
    print("🚀 SHELLHACKS AGENTS - AVAILABLE COMMANDS")
    print("=" * 60)
    
    # Check service status
    agents_status = check_service("http://localhost:8080/health", "Agents Service")
    string_status = check_service("http://localhost:8000/health", "String Comparison")
    
    print(f"\n📊 SERVICE STATUS:")
    print(f"   Agents Service (8080): {'🟢 Online' if agents_status else '🔴 Offline'}")
    print(f"   String Comparison (8000): {'🟢 Online' if string_status else '🔴 Offline'}")
    
    print(f"\n🔧 INITIALIZATION COMMANDS:")
    print(f"   Start Agents:             python3 app.py")
    print(f"   Start String Compare:     cd ../string-comparison && python3 backend.py")
    print(f"   Start Both:               ./start_services.sh")
    
    print(f"\n🤖 CREATE AGENTS:")
    print(f"   Basic Agent:              curl -X POST localhost:8080/api/v1/agents -H 'Content-Type: application/json' -d '{{\"user_id\": \"expert\", \"training_data\": \"I am an expert in...\"}}'")
    print(f"   Advanced Agent:           curl -X POST localhost:8080/api/v1/agents/advanced -H 'Content-Type: application/json' -d '{{\"user_id\": \"advanced\", \"json_dataset\": [...]}}'")
    
    print(f"\n📋 MANAGEMENT:")
    print(f"   List Agents:              curl localhost:8080/api/v1/agents")
    print(f"   Agent Status:             curl localhost:8080/api/v1/agents/USER_ID/status")
    print(f"   Agent Pipeline:           curl localhost:8080/api/v1/agents/USER_ID/pipeline")
    print(f"   Delete Agent:             curl -X DELETE localhost:8080/api/v1/agents/USER_ID")
    
    print(f"\n💬 INFERENCE:")
    print(f"   Ask Question:             curl -X POST localhost:8080/api/v1/agents/USER_ID/inference -H 'Content-Type: application/json' -d '{{\"prompt\": \"Your question\"}}'")
    print(f"   Evaluate Response:        curl -X POST localhost:8080/api/v1/agents/USER_ID/evaluate -H 'Content-Type: application/json' -d '{{\"test_prompt\": \"...\", \"expected_answer\": \"...\"}}'")
    
    print(f"\n🧪 AUTOMATED TESTS:")
    print(f"   Advanced System Test:     python3 test_advanced_system.py")
    print(f"   Basic System Test:        python3 test_agent_api.py")
    print(f"   List Agents:              python3 list_agents.py")
    
    print(f"\n🤖 MULTI-AGENT TRAINING:")
    print(f"   Train 5 Specialized Agents: python3 train_multiple_agents.py")
    print(f"   Quick Start Multi-Agents:   python3 quick_start_multiple_agents.py")
    
    print(f"\n🧬 REINFORCED LEARNING & EVOLUTION:")
    print(f"   Start Evolution (Easy):     python3 start_evolution.py")
    print(f"   Full Evolution System:      python3 evolutionary_agent_optimizer.py")
    print(f"   Reinforced Learning:        python3 reinforced_learning_system.py")
    print(f"   Real-time Monitor:          python3 real_time_monitor.py")
    
    print(f"\n🔍 MONITORING:")
    print(f"   Health Check Agents:      curl localhost:8080/health")
    print(f"   Health Check String:      curl localhost:8000/health")
    print(f"   View Configuration:       curl localhost:8080/api/v1/config")
    
    print(f"\n🛠️ DEBUG AND CLEANUP:")
    print(f"   Stop Services:            killall Python")
    print(f"   View Processes:           ps aux | grep python")
    print(f"   Clean Data:               rm -rf data/user_data/* models/user_models/* logs/user_logs/*")
    
    print(f"\n📚 DOCUMENTATION:")
    print(f"   How to Use:               cat HOW_TO_USE.md")
    print(f"   Quick Commands:           cat QUICK_COMMANDS.md")
    print(f"   Complete Commands:        cat COMPLETE_COMMANDS.md")
    print(f"   Requests Guide:           cat REQUESTS_GUIDE.md")
    
    print(f"\n📊 TRAINING DATASETS:")
    print(f"   Python Expert:            ls training_datasets/python_expert_dataset.json")
    print(f"   ML Expert:                ls training_datasets/ml_expert_dataset.json")
    print(f"   DevOps Expert:            ls training_datasets/devops_expert_dataset.json")
    print(f"   Data Science Expert:      ls training_datasets/data_science_expert_dataset.json")
    print(f"   Cybersecurity Expert:     ls training_datasets/cybersecurity_expert_dataset.json")
    
    print(f"\n🎯 QUICK EXAMPLES:")
    print(f"   1. Start services:")
    print(f"      python3 app.py")
    print(f"      cd ../string-comparison && python3 backend.py")
    print(f"   ")
    print(f"   2. Create agent:")
    print(f"      curl -X POST localhost:8080/api/v1/agents \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"user_id\": \"expert\", \"training_data\": \"I am a Python expert...\"}}'")
    print(f"   ")
    print(f"   3. Ask question:")
    print(f"      curl -X POST localhost:8080/api/v1/agents/expert/inference \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"prompt\": \"How to create a REST API?\"}}'")
    
    print(f"\n💡 TIPS:")
    print(f"   - Always check service status before creating agents")
    print(f"   - Wait for 'ready' status before asking questions")
    print(f"   - Use advanced agents for better performance")
    print(f"   - Check logs for debugging issues")
    
    print(f"\n🔗 USEFUL LINKS:")
    print(f"   - Documentation: README.md")
    print(f"   - API Examples: test_agent_api.py")
    print(f"   - Advanced Tests: test_advanced_system.py")

def check_service(url, name):
    """Check if a service is running"""
    try:
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except:
        return False

def show_service_status():
    """Show detailed service status"""
    print("🔍 CHECKING SERVICE STATUS...")
    print("=" * 40)
    
    services = [
        ("Agents Service", "http://localhost:8080/health"),
        ("String Comparison", "http://localhost:8000/health")
    ]
    
    for name, url in services:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ {name}: Online")
                try:
                    data = response.json()
                    if 'version' in data:
                        print(f"   Version: {data['version']}")
                    if 'status' in data:
                        print(f"   Status: {data['status']}")
                except:
                    pass
            else:
                print(f"❌ {name}: Offline (Status: {response.status_code})")
        except requests.exceptions.ConnectionError:
            print(f"❌ {name}: Offline (Connection refused)")
        except Exception as e:
            print(f"❌ {name}: Offline (Error: {e}")
        print()

def show_agent_examples():
    """Show agent creation examples"""
    print("🤖 AGENT CREATION EXAMPLES")
    print("=" * 40)
    
    print("1. Basic Agent (Simple text training):")
    print("""
curl -X POST localhost:8080/api/v1/agents \\
  -H 'Content-Type: application/json' \\
  -d '{
    "user_id": "python_expert",
    "training_data": "I am a Python expert with experience in Flask, Django, FastAPI. I can help with web development, data analysis and machine learning.",
    "base_model": "distilbert-base-uncased"
  }'
""")
    
    print("2. Advanced Agent (JSON dataset + QLoRA):")
    print("""
curl -X POST localhost:8080/api/v1/agents/advanced \\
  -H 'Content-Type: application/json' \\
  -d '{
    "user_id": "advanced_expert",
    "json_dataset": [
      {"prompt": "How to create REST API?", "answer": "Use Flask with @app.route() decorators"},
      {"prompt": "What is Django?", "answer": "Django is a Python web framework"},
      {"prompt": "How to use pandas?", "answer": "Pandas is for data analysis and manipulation"}
    ],
    "base_model": "distilbert-base-uncased"
  }'
""")

def show_inference_examples():
    """Show inference examples"""
    print("💬 INFERENCE EXAMPLES")
    print("=" * 40)
    
    print("1. Ask a question:")
    print("""
curl -X POST localhost:8080/api/v1/agents/python_expert/inference \\
  -H 'Content-Type: application/json' \\
  -d '{"prompt": "How to create a REST API with Flask?"}'
""")
    
    print("2. Evaluate response:")
    print("""
curl -X POST localhost:8080/api/v1/agents/python_expert/evaluate \\
  -H 'Content-Type: application/json' \\
  -d '{
    "test_prompt": "How to create Flask API?",
    "expected_answer": "Use Flask with @app.route decorators"
  }'
""")

def show_troubleshooting():
    """Show troubleshooting tips"""
    print("🚨 TROUBLESHOOTING")
    print("=" * 40)
    
    print("Common Issues:")
    print("1. Connection refused:")
    print("   - Check if server is running: python3 app.py")
    print("   - Check port availability: lsof -i :8080")
    print()
    print("2. Model not ready:")
    print("   - Wait for training to complete")
    print("   - Check status: curl localhost:8080/api/v1/agents/USER_ID/status")
    print()
    print("3. User not found:")
    print("   - Create agent first")
    print("   - List agents: curl localhost:8080/api/v1/agents")
    print()
    print("4. String comparison not working:")
    print("   - Start service: cd ../string-comparison && python3 backend.py")
    print("   - Check health: curl localhost:8000/health")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "status":
            show_service_status()
        elif command == "examples":
            show_agent_examples()
        elif command == "inference":
            show_inference_examples()
        elif command == "troubleshoot":
            show_troubleshooting()
        elif command == "all":
            show_help()
            print("\n" + "="*60)
            show_service_status()
        else:
            print("❌ Unknown command. Available commands:")
            print("   status       - Show service status")
            print("   examples     - Show agent examples")
            print("   inference    - Show inference examples")
            print("   troubleshoot - Show troubleshooting tips")
            print("   all          - Show complete help")
    else:
        show_help()

if __name__ == "__main__":
    main()