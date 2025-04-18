
import streamlit as st
import webbrowser
import json
import os

def main():
    st.title("🔧 Google Sheets Integration Fix")
    
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
                """)

                if st.button("1. Enable Google Sheets API"):
                    st.markdown(f"Opening: {sheets_api_url}")
                    webbrowser.open_new_tab(sheets_api_url)
                
                if st.button("2. Enable Google Drive API"):
                    st.markdown(f"Opening: {drive_api_url}")
                    webbrowser.open_new_tab(drive_api_url)
                
                st.markdown("### Share your spreadsheet")
                st.markdown(f"""
                Make sure you've shared your Google Sheet with the service account email:
                ```
                {client_email}
                ```
                """)
                
                st.markdown("### Test Connection")
                if st.button("Test Google Sheets Connection"):
                    st.markdown("Running connection test...")
                    
                    try:
                        import gspread
                        from google.oauth2 import service_account
                        
                        # Create credentials
                        credentials = service_account.Credentials.from_service_account_info(
                            service_account_info,
                            scopes=['https://www.googleapis.com/auth/spreadsheets', 
                                   'https://www.googleapis.com/auth/drive']
                        )
                        
                        # Connect to gspread
                        client = gspread.authorize(credentials)
                        
                        # Try to create a test sheet
                        sheet = client.create("RadVision_Test_Sheet")
                        sheet.sheet1.update('A1', 'Connection Test Successful!')
                        
                        st.success("✅ Connection successful! Test spreadsheet created.")
                        st.info(f"Test sheet URL: {sheet.url}")
                    except Exception as e:
                        st.error(f"❌ Connection test failed: {str(e)}")
                        st.info("Make sure you've enabled the APIs and installed the gspread package.")
            else:
                st.error("❌ Invalid service account file. Missing project_id or client_email.")
        except Exception as e:
            st.error(f"❌ Failed to parse service account file: {str(e)}")
    else:
        st.error("❌ Service account file (gen.json) not found!")
        st.info("Please upload your service account JSON file and name it 'gen.json'")

if __name__ == "__main__":
    main()
