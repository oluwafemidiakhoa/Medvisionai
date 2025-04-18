import os
import json
import streamlit as st
import webbrowser
from google.oauth2.service_account import Credentials
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_service_account():
    """Diagnostic tool to check service account JSON formatting issues."""
    st.title("🔍 Service Account JSON Diagnostics")
    
    st.markdown("""
    ## Service Account JSON Diagnostic Tool
    
    This utility will help diagnose issues with your Google Service Account JSON format.
    It will show details about the format and any parsing errors.
    """)
    
    # Get the current service account JSON from environment or .env file
    current_json = os.environ.get("SERVICE_ACCOUNT_JSON", "")
    
    if not current_json and os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("SERVICE_ACCOUNT_JSON="):
                        current_json = line[len("SERVICE_ACCOUNT_JSON="):].strip()
                        break
        except Exception as e:
            st.warning(f"Could not read from .env file: {e}")
    
    # Display basic information
    st.header("Basic Information")
    if current_json:
        st.success("✅ SERVICE_ACCOUNT_JSON environment variable is set")
        st.info(f"Length of JSON string: {len(current_json)} characters")
        
        # Display the first and last few characters
        st.code(f"Beginning: {current_json[:50]}...\nEnding: ...{current_json[-50:]}")
    else:
        st.error("❌ SERVICE_ACCOUNT_JSON environment variable is not set")
    
    # Parsing test
    st.header("JSON Parsing Test")
    
    if current_json:
        try:
            # Try to parse the JSON
            parsed = json.loads(current_json)
            st.success("✅ Successfully parsed JSON")
            
            # Check required fields
            required_fields = ["type", "project_id", "private_key", "client_email"]
            missing_fields = [field for field in required_fields if field not in parsed]
            
            if missing_fields:
                st.error(f"❌ Missing required fields: {', '.join(missing_fields)}")
            else:
                st.success("✅ All required fields present")
                
                # Show some details
                st.json({
                    "type": parsed.get("type"),
                    "project_id": parsed.get("project_id"),
                    "client_email": parsed.get("client_email"),
                    "private_key_length": len(parsed.get("private_key", "")) if "private_key" in parsed else 0
                })
        except json.JSONDecodeError as e:
            st.error(f"❌ Failed to parse JSON: {e}")
            st.markdown(f"Error position: character {e.pos}, line {e.lineno}, column {e.colno}")
            
            # Try some cleaning approaches
            st.header("Attempted JSON Fixes")
            
            # 1. Fix double-encoding
            try:
                st.subheader("1. Testing for double-encoding")
                # Sometimes environment variables can get double-encoded
                if current_json.startswith('"') and current_json.endswith('"'):
                    internal_json = current_json[1:-1].replace('\\"', '"')
                    parsed = json.loads(internal_json)
                    st.success("✅ Successfully parsed after fixing double-encoding")
                    st.json({
                        "type": parsed.get("type"),
                        "project_id": parsed.get("project_id")
                    })
                else:
                    st.warning("Not double-encoded.")
            except Exception as e1:
                st.error(f"❌ Fix attempt failed: {e1}")
            
            # 2. Fix escaped characters
            try:
                st.subheader("2. Testing for escaped characters")
                cleaned = current_json.replace('\\"', '"').replace('\\\\n', '\\n')
                parsed = json.loads(cleaned)
                st.success("✅ Successfully parsed after fixing escaped characters")
                st.json({
                    "type": parsed.get("type"),
                    "project_id": parsed.get("project_id")
                })
            except Exception as e2:
                st.error(f"❌ Fix attempt failed: {e2}")
            
            # 3. Show problematic section
            st.subheader("3. Problematic section")
            if hasattr(e, 'pos') and e.pos > 10:
                start = max(0, e.pos - 15)
                end = min(len(current_json), e.pos + 15)
                problematic = current_json[start:end]
                st.code(f"...{problematic}...")
                st.markdown(f"Problem around: **{current_json[e.pos:e.pos+1]}**")
    
    # Manual JSON input for testing
    st.header("Test Alternative JSON")
    test_json = st.text_area(
        "Paste service account JSON to test",
        height=200,
        placeholder="Paste your JSON here to test parsing"
    )
    
    if test_json and st.button("Test This JSON"):
        try:
            parsed = json.loads(test_json)
            st.success("✅ Successfully parsed test JSON")
            
            # Show some details
            st.json({
                "type": parsed.get("type"),
                "project_id": parsed.get("project_id"),
                "client_email": parsed.get("client_email"),
                "private_key_length": len(parsed.get("private_key", "")) if "private_key" in parsed else 0
            })
            
            # Option to use this JSON
            if st.button("Use This JSON"):
                # Clean JSON string by re-serializing
                clean_json = json.dumps(parsed)
                
                # Update environment
                os.environ["SERVICE_ACCOUNT_JSON"] = clean_json
                
                # Update .env file
                with open(".env", "w") as f:
                    f.write(f'SERVICE_ACCOUNT_JSON={clean_json}')
                
                st.success("✅ Service account JSON updated!")
                st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"❌ Failed to parse test JSON: {e}")

def diagnose_sheets_api():
    st.title("🔍 Google Sheets API Diagnostic Tool")

    # Check if gen.json exists
    if os.path.exists("gen.json"):
        st.success("✅ Service account file (gen.json) found!")

        try:
            # Load the service account info
            with open("gen.json", "r") as f:
                service_account_info = json.load(f)

            # Extract key information
            project_id = service_account_info.get("project_id")
            client_email = service_account_info.get("client_email")

            if project_id and client_email:
                st.write(f"Project ID: `{project_id}`")
                st.write(f"Service Account Email: `{client_email}`")

                # Generate API enablement links
                sheets_api_url = f"https://console.developers.google.com/apis/api/sheets.googleapis.com/overview?project={project_id}"
                drive_api_url = f"https://console.developers.google.com/apis/api/drive.googleapis.com/overview?project={project_id}"

                st.markdown("### Required Steps")

                st.markdown("""
                To fix the Google Sheets integration, you need to enable these APIs in your Google Cloud Console:

                1. **Google Sheets API**
                2. **Google Drive API** (required for creating/accessing sheets)

                Click the buttons below to open the API enablement pages:
                """)

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("🔗 Enable Google Sheets API", use_container_width=True):
                        st.markdown(f"[Open Sheets API Page]({sheets_api_url})")

                with col2:
                    if st.button("🔗 Enable Google Drive API", use_container_width=True):
                        st.markdown(f"[Open Drive API Page]({drive_api_url})")

                st.info("After enabling the APIs, wait a few minutes for the changes to propagate before retrying the Sheets integration.")

                # Verify API permissions
                st.markdown("### Test Service Account Permissions")
                if st.button("Test Service Account Permissions"):
                    st.write("Attempting to authenticate with service account...")

                    try:
                        # Create credentials from the service account
                        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
                        credentials = Credentials.from_service_account_info(
                            service_account_info,
                            scopes=scopes)

                        st.success("✅ Successfully created credentials from service account!")
                        st.info("Note: This only verifies that the service account file is valid. It does not verify that the APIs are enabled.")
                    except Exception as e:
                        st.error(f"❌ Error creating credentials: {str(e)}")
            else:
                st.error("❌ Service account file is missing required fields (project_id or client_email).")
        except json.JSONDecodeError:
            st.error("❌ Service account file exists but is not valid JSON.")
        except Exception as e:
            st.error(f"❌ Error processing service account file: {str(e)}")
    else:
        st.error("❌ Service account file (gen.json) not found!")

    st.markdown("---")

    st.markdown("""
    ### After Enabling APIs

    Once you've enabled the APIs:

    1. Wait 5-10 minutes for changes to propagate
    2. Go back to the main app
    3. Try the Sheets integration again

    If it still doesn't work, check these common issues:

    - Ensure your service account has the necessary permissions
    - Verify that your Google Cloud project billing is set up (some APIs require billing)
    - Check that your service account hasn't been disabled
    """)

if __name__ == "__main__":
    diagnose_service_account()
    diagnose_sheets_api()