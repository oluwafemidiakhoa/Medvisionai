
import os
import json
import streamlit as st

def check_secrets():
    """Check if the necessary Replit Secrets are set up correctly"""
    
    st.title("🔑 Replit Secrets Configuration Check")
    
    # Check if SERVICE_ACCOUNT_JSON is set
    service_account_json = os.environ.get("SERVICE_ACCOUNT_JSON")
    
    if service_account_json:
        st.success("✅ SERVICE_ACCOUNT_JSON is configured in Replit Secrets")
        
        # Check if it's valid JSON
        try:
            service_account_info = json.loads(service_account_json)
            
            # Check for required fields
            required_fields = ["type", "project_id", "private_key", "client_email"]
            missing_fields = [field for field in required_fields if field not in service_account_info]
            
            if missing_fields:
                st.error(f"❌ SERVICE_ACCOUNT_JSON is missing required fields: {', '.join(missing_fields)}")
            else:
                st.info(f"Project ID: {service_account_info.get('project_id')}")
                st.info(f"Client Email: {service_account_info.get('client_email')}")
                st.success("✅ SERVICE_ACCOUNT_JSON appears to be properly formatted")
        except json.JSONDecodeError:
            st.error("❌ SERVICE_ACCOUNT_JSON is not valid JSON")
    else:
        st.error("❌ SERVICE_ACCOUNT_JSON is not set in Replit Secrets")
        st.markdown("""
        ### How to Set Up SERVICE_ACCOUNT_JSON
        
        1. Go to the **Tools** panel and click on **Secrets**
        2. Click on **+ New secret**
        3. For the key, enter `SERVICE_ACCOUNT_JSON`
        4. For the value, paste your entire Google Service Account JSON
        5. Click **Add Secret**
        """)
    
    # Check other secrets
    other_secrets = ["GEMINI_API_KEY", "UMLS_APIKEY", "HF_API_TOKEN"]
    
    st.subheader("Other API Keys")
    
    for key in other_secrets:
        if os.environ.get(key):
            st.success(f"✅ {key} is configured")
        else:
            st.warning(f"⚠️ {key} is not configured")
    
    st.markdown("---")
    
    st.markdown("""
    ### Next Steps
    
    1. Ensure all required secrets are properly configured
    2. Run the application to verify everything works correctly
    3. For more information, see the [Replit Secrets documentation](https://docs.replit.com/programming-ide/storing-sensitive-information-environment-variables)
    """)

if __name__ == "__main__":
    check_secrets()
