import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file in the root directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..','.env'))

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file")
    print("Please check that your .env file in the root directory contains: GEMINI_API_KEY=your_api_key_here")
    exit(1)

# Use the standard Gemini API instead of the Live API
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

def chat_with_gemini(prompt="give an error message"):
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    }
    
    # Prepare the request payload
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }
    
    try:
        print("Sending request to Gemini API...")
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("Response received:")
            print("-" * 50)
            
            with open("../db/LLMData.txt", "w", encoding="utf-8") as f:
                # format: {prompt,}
                f.write(payload["contents"][0]["parts"][0]["text"] + "\n")
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            if "text" in part:
                                # Remove newlines and extra whitespace to make it one line
                                text_content = part["text"].replace('\n', ' ').replace('\r', '').strip()
                                f.write(text_content)
                                print(part["text"])
                                
                    else:
                        print("No text content found in response")
                else:
                    print("No candidates found in response")
                    print("Full response:", json.dumps(data, indent=2))
                f.write("\n")
            with open("../db/LLMCurrData.txt", "a", encoding="utf-8") as f:
                f.write(payload["contents"][0]["parts"][0]["text"] + "\n")
                f.write(data["candidates"][0]["content"]["parts"][0]["text"] + "\n")
                f.write("\n")
        else:
            print(f"Error: HTTP {response.status_code}")
            print("Response:", response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Response:", response.text)
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    import sys
    
    # Check if prompt was provided as command line argument
    if len(sys.argv) > 1:
        # Join all arguments to form the prompt
        prompt = " ".join(sys.argv[1:])
        print(f"Using prompt: {prompt}")
    else:
        # Ask user for input
        prompt = input("Enter your prompt: ")
        if not prompt.strip():
            prompt = "give an error message"
            print(f"No prompt provided, using default: {prompt}")
    
    chat_with_gemini(prompt)
