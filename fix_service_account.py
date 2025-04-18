
import os
import json
import streamlit as st

def fix_service_account():
    st.title("🛠️ Fix Google Service Account")
    
    st.markdown("""
    ## Service Account JSON Fixer
    
    This utility will help you correctly input your Google Service Account JSON.
    
    The previous error occurred because the JSON was truncated or malformed.
    """)
    
    # Instructions
    st.info("Please paste your **complete** Google Service Account JSON below")
    
    # Get the current service account JSON from .env if available
    current_json = ""
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("SERVICE_ACCOUNT_JSON="):
                        current_json = line[len("SERVICE_ACCOUNT_JSON="):].strip()
                        break
        except Exception as e:
            st.warning(f"Could not read current service account: {e}")
    
    # Display in text area
    service_account_json = st.text_area(
        "Service Account JSON",
        value=current_json if current_json else "",
        height=400,
        placeholder='''{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "abcdef1234567890",
  "private_key": "-----BEGIN PRIVATE KEY-----\\nYOUR_PRIVATE_KEY_HERE\\n-----END PRIVATE KEY-----\\n",
  "client_email": "your-service-account@your-project-id.iam.gserviceaccount.com",
  "client_id": "123456789012345678901",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project-id.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}'''
    )
    
    if st.button("Validate and Save", type="primary"):
        if not service_account_json:
            st.error("Please paste your service account JSON")
            return
            
        try:
            # Try to validate and clean JSON format
            try:
                # First, try direct JSON load
                parsed_json = json.loads(service_account_json)
                
                # Check for required fields
                required_fields = ["type", "project_id", "private_key", "client_email"]
                missing_fields = [field for field in required_fields if field not in parsed_json]
                
                if missing_fields:
                    st.error(f"Missing required fields: {', '.join(missing_fields)}")
                    return
                
                # Re-serialize to ensure clean JSON
                clean_json = json.dumps(parsed_json)
                
                # Store the validated and cleaned JSON directly in environment
                os.environ["SERVICE_ACCOUNT_JSON"] = clean_json
                
                # Write to .env file
                with open(".env", "w") as f:
                    f.write(f'SERVICE_ACCOUNT_JSON={clean_json}')
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {e}")
                st.info("Let's try to clean up the JSON format...")
                
                # Try to clean up common issues
                try:
                    # Replace any escaped quotes
                    cleaned = service_account_json.replace('\\"', '"')
                    # Replace double backslashes in the private key
                    cleaned = cleaned.replace('\\\\n', '\\n')
                    # Try loading again
                    parsed_json = json.loads(cleaned)
                    
                    # Check for required fields
                    required_fields = ["type", "project_id", "private_key", "client_email"]
                    missing_fields = [field for field in required_fields if field not in parsed_json]
                    
                    if missing_fields:
                        st.error(f"Missing required fields after cleanup: {', '.join(missing_fields)}")
                        return
                    
                    # Re-serialize to ensure clean JSON
                    clean_json = json.dumps(parsed_json)
                    
                    # Store the validated and cleaned JSON directly in environment
                    os.environ["SERVICE_ACCOUNT_JSON"] = clean_json
                    
                    # Write to .env file
                    with open(".env", "w") as f:
                        f.write(f'SERVICE_ACCOUNT_JSON={clean_json}')
                    
                    st.success("Successfully cleaned and validated the JSON!")
                    return
                except Exception as inner_e:
                    st.error(f"Still unable to parse JSON after cleanup attempts: {inner_e}")
                    st.info("Please make sure you're pasting the complete, valid JSON from your service account key file.")
                    return
                
            st.success("✅ Service account JSON saved successfully!")
            
            # Display a summary of the configuration 
            st.info(f"Project ID: {parsed_json['project_id']}\nClient Email: {parsed_json['client_email']}")
            
            if st.button("Return to Main App"):
                st.rerun()
                
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {e}")
            st.info("Make sure to paste the complete JSON without any line breaks or modifications")
        except Exception as e:
            st.error(f"Error saving service account: {e}")

if __name__ == "__main__":
    fix_service_account()
