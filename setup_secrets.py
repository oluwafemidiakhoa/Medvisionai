
import os
import json
import streamlit as st
import base64

def setup_secrets():
    st.title("🔒 RadVision AI Setup")
    
    st.header("Google Sheets Integration Setup")
    
    # Check if service account is already configured
    current_service_account = os.environ.get("SERVICE_ACCOUNT_JSON")
    if current_service_account:
        st.success("✅ Google Service Account is already configured!")
        
        # Option to view or update
        if st.checkbox("View/Update Service Account Configuration"):
            # Show the first few and last few characters of the key
            key_preview = current_service_account[:20] + "..." + current_service_account[-20:]
            st.code(key_preview, language="json")
            
            new_service_account = st.text_area(
                "Enter new Service Account JSON (leave empty to keep current)",
                height=200,
                placeholder="Paste the full JSON content of your service account key file here"
            )
            
            if new_service_account and st.button("Update Service Account"):
                try:
                    # Validate JSON format
                    json.loads(new_service_account)
                    
                    # Store in secrets
                    with open(".env", "w") as f:
                        f.write(f'SERVICE_ACCOUNT_JSON={json.dumps(json.loads(new_service_account))}')
                    os.environ["SERVICE_ACCOUNT_JSON"] = new_service_account
                    st.success("✅ Service Account updated!")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Please check the content and try again.")
    else:
        st.warning("⚠️ Google Service Account is not configured")
        
        service_account_json = st.text_area(
            "Enter Google Service Account JSON",
            height=200,
            placeholder="Paste the full JSON content of your service account key file here"
        )
        
        if service_account_json and st.button("Save Service Account"):
            try:
                # Validate JSON format
                parsed_json = json.loads(service_account_json)
                
                # Store the raw JSON string directly in environment
                os.environ["SERVICE_ACCOUNT_JSON"] = service_account_json
                
                # Store in .env file - properly escape the JSON
                with open(".env", "w") as f:
                    json_str = json.dumps(parsed_json)
                    f.write(f'SERVICE_ACCOUNT_JSON={json_str}')
                
                st.success("✅ Google Service Account configured successfully!")
                st.rerun()
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check the content and try again.")
    
    st.header("UMLS API Key Setup")
    
    # Check if UMLS API key is already configured
    current_umls_key = os.environ.get("UMLS_APIKEY")
    if current_umls_key:
        st.success("✅ UMLS API Key is already configured!")
        
        # Option to update
        if st.checkbox("Update UMLS API Key"):
            new_umls_key = st.text_input(
                "Enter new UMLS API Key (leave empty to keep current)",
                type="password"
            )
            
            if new_umls_key and st.button("Update UMLS API Key"):
                # Store in .env file
                with open(".env", "a") as f:
                    f.write(f"\nUMLS_APIKEY='{new_umls_key}'")
                
                # Set in current environment
                os.environ["UMLS_APIKEY"] = new_umls_key
                
                st.success("✅ UMLS API Key updated!")
                st.rerun()
    else:
        st.warning("⚠️ UMLS API Key is not configured")
        
        umls_key = st.text_input(
            "Enter UMLS API Key",
            type="password",
            help="Get this from https://uts.nlm.nih.gov/uts/signup-login"
        )
        
        if umls_key and st.button("Save UMLS API Key"):
            # Store in .env file
            with open(".env", "a") as f:
                f.write(f"\nUMLS_APIKEY='{umls_key}'")
            
            # Set in current environment
            os.environ["UMLS_APIKEY"] = umls_key
            
            st.success("✅ UMLS API Key configured successfully!")
            st.rerun()
    
    st.header("Gemini API Key Setup")
    
    # Check if Gemini API key is already configured
    current_gemini_key = os.environ.get("GEMINI_API_KEY")
    if current_gemini_key:
        st.success("✅ Gemini API Key is already configured!")
        
        # Option to update
        if st.checkbox("Update Gemini API Key"):
            new_gemini_key = st.text_input(
                "Enter new Gemini API Key (leave empty to keep current)",
                type="password"
            )
            
            if new_gemini_key and st.button("Update Gemini API Key"):
                # Store in .env file
                with open(".env", "a") as f:
                    f.write(f"\nGEMINI_API_KEY='{new_gemini_key}'")
                
                # Set in current environment
                os.environ["GEMINI_API_KEY"] = new_gemini_key
                
                st.success("✅ Gemini API Key updated!")
                st.rerun()
    else:
        st.warning("⚠️ Gemini API Key is not configured")
        
        gemini_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            help="Get this from https://ai.google.dev/"
        )
        
        if gemini_key and st.button("Save Gemini API Key"):
            # Store in .env file
            with open(".env", "a") as f:
                f.write(f"\nGEMINI_API_KEY='{gemini_key}'")
            
            # Set in current environment
            os.environ["GEMINI_API_KEY"] = gemini_key
            
            st.success("✅ Gemini API Key configured successfully!")
            st.rerun()
    
    st.markdown("---")
    st.info("All configurations are stored securely in environment variables.")
    
    # Return to main app
    if st.button("Return to Main App"):
        st.session_state.setup_complete = True
        st.rerun()

if __name__ == "__main__":
    setup_secrets()
