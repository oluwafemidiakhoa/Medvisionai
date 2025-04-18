
import os
import re
import json

def update_sheets_integration():
    """Update the sheets_integration.py file to use Replit Secrets instead of gen.json"""
    
    if os.path.exists("sheets_integration.py"):
        try:
            with open("sheets_integration.py", "r") as f:
                content = f.read()
            
            # Look for patterns that load credentials from gen.json
            if "from_service_account_file" in content and "gen.json" in content:
                # Replace with code that uses environment variables (Replit Secrets)
                updated_content = re.sub(
                    r"credentials = Credentials\.from_service_account_file\(['\"]gen\.json['\"]",
                    "credentials = Credentials.from_service_account_info(json.loads(os.environ.get('SERVICE_ACCOUNT_JSON', '{}')))",
                    content
                )
                
                # Make sure to import json
                if "import json" not in updated_content:
                    updated_content = "import json\n" + updated_content
                
                # Write the updated file
                with open("sheets_integration.py", "w") as f:
                    f.write(updated_content)
                
                print("✅ Updated sheets_integration.py to use Replit Secrets")
            else:
                print("sheets_integration.py already appears to be using environment variables")
        
        except Exception as e:
            print(f"Error updating sheets_integration.py: {e}")
    else:
        print("sheets_integration.py not found")
    
    print("\nReminder: Be sure to add your credentials to Replit Secrets:")
    print("1. Go to the 'Secrets' tab in the Tools panel")
    print("2. Add a new secret with key 'SERVICE_ACCOUNT_JSON'")
    print("3. Paste the entire service account JSON as the value")

if __name__ == "__main__":
    update_sheets_integration()
