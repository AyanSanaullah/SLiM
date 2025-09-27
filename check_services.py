#!/usr/bin/env python3
"""
Service Status Checker
Quick script to check if all services are running properly
"""

import requests
import json

def check_service_status():
    """Check the status of all services"""
    print("🔍 Checking Service Status")
    print("=" * 30)
    
    # Check backend
    try:
        response = requests.get('http://localhost:5001/test', timeout=3)
        if response.status_code == 200:
            print("✅ Backend Server: Running on port 5001")
        else:
            print(f"❌ Backend Server: Error {response.status_code}")
    except Exception as e:
        print(f"❌ Backend Server: Not responding ({e})")
    
    # Check GPU monitoring
    try:
        response = requests.get('http://localhost:5001/gpu/current', timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                gpu_name = data.get('data', {}).get('name', 'Unknown')
                print(f"✅ GPU Monitoring: Active ({gpu_name})")
            else:
                print("❌ GPU Monitoring: Error in response")
        else:
            print(f"❌ GPU Monitoring: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ GPU Monitoring: Not responding ({e})")
    
    # Check GPU graph data
    try:
        response = requests.get('http://localhost:5001/gpu/graph', timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('data'):
                graph_data = data['data']
                data_points = len(graph_data.get('gpu_usage', []))
                print(f"✅ GPU Graph Data: {data_points} data points")
            else:
                print("❌ GPU Graph Data: No data available")
        else:
            print(f"❌ GPU Graph Data: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ GPU Graph Data: Not responding ({e})")
    
    # Check frontend (if accessible)
    try:
        response = requests.get('http://localhost:3000', timeout=3)
        if response.status_code == 200:
            print("✅ Frontend: Running on port 3000")
        else:
            print(f"⚠️ Frontend: HTTP {response.status_code}")
    except Exception as e:
        print(f"⚠️ Frontend: Not accessible ({e})")
    
    print("\n🌐 Access Points:")
    print("   Dashboard: http://localhost:3000/dashboard")
    print("   Backend API: http://localhost:5001")
    print("   GPU Data: http://localhost:5001/gpu/current")

if __name__ == "__main__":
    check_service_status()
