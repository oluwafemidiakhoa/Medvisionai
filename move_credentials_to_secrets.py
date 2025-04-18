
import os
import json
import streamlit as st

def move_service_account_to_secrets():
    """Move service account credentials from gen.json to Replit Secrets"""
    
    print("Moving service account credentials to Replit Secrets...")
    
    # Check if gen.json exists
    if os.path.exists("gen.json"):
        try:
            # Read the service account file
            with open("gen.json", "r") as f:
                service_account_info = json.load(f)
            
            # Convert to string for storage in secrets
            service_account_json = json.dumps(service_account_info)
            
            print("Successfully read service account info from gen.json")
            print("Please go to the Secrets tab in Replit and:")
            print("1. Click on '+ New secret'")
            print("2. Enter 'SERVICE_ACCOUNT_JSON' as the key")
            print("3. Paste the following as the value:")
            print("\n" + service_account_json + "\n")
            
            print("After adding the secret, you can safely remove gen.json by running:")
            print("git rm --cached gen.json")
            
        except Exception as e:
            print(f"Error reading service account file: {e}")
    else:
        print("gen.json file not found.")
        
        # Try to read from .env as fallback
        if os.path.exists(".env"):
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("SERVICE_ACCOUNT_JSON="):
                            service_account_json = line[len("SERVICE_ACCOUNT_JSON="):].strip()
                            print("Found service account in .env file")
                            print("Please go to the Secrets tab in Replit and:")
                            print("1. Click on '+ New secret'")
                            print("2. Enter 'SERVICE_ACCOUNT_JSON' as the key")
                            print("3. Paste the following as the value:")
                            print("\n" + service_account_json + "\n")
            except Exception as e:
                print(f"Error reading .env file: {e}")

if __name__ == "__main__":
    move_service_account_to_secrets()
