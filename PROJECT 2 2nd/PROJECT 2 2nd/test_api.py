"""
Simple test script for the Data Analyst Agent API
Run this script to verify the system is working correctly
"""

import requests
import json
import sys
import time

API_BASE = "http://localhost:8000"

def test_health():
    """Test if the server is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server health check passed")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to server: {e}")
        print("Make sure the server is running with: python main.py")
        return False

def test_analysis():
    """Test the analysis endpoint with a simple Wikipedia URL"""
    test_data = {
        "url": "https://en.wikipedia.org/wiki/List_of_countries_by_population",
        "tasks": [
            "How many countries are listed in the table?",
            "What is the population of China?"
        ]
    }
    
    try:
        print("🔍 Testing data analysis...")
        response = requests.post(
            f"{API_BASE}/analyze", 
            json=test_data, 
            timeout=60
        )
        
        if response.status_code == 200:
            results = response.json()
            print("✅ Analysis completed successfully!")
            print("Results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result}")
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return False

def main():
    print("🤖 Data Analyst Agent - API Test")
    print("=" * 40)
    
    # Test server health
    if not test_health():
        sys.exit(1)
    
    print()
    
    # Test analysis functionality
    if not test_analysis():
        sys.exit(1)
    
    print()
    print("🎉 All tests passed! The system is working correctly.")
    print(f"Access the web interface at: {API_BASE}/frontend.html")

if __name__ == "__main__":
    main()
