"""
sheets_integration.py - Google Sheets Integration for RadVision AI
=================================================================

Provides functionality to:
1. Log analysis results to a Google Sheet for tracking
2. Retrieve and display past analyses 
3. Export/import findings for research or clinical review
"""

import os
import json
import logging
import pandas as pd
import streamlit as st
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# Define SHEETS_AVAILABLE globally before any import attempts
SHEETS_AVAILABLE = False

# Try importing gspread
try:
    import gspread
    from google.oauth2.service_account import Credentials
    SHEETS_AVAILABLE = True
    logger.info("Google Sheets integration available.")
except ImportError:
    logger.warning("Google Sheets integration unavailable - missing dependencies.")
    # Create placeholder for gspread to avoid NameError
    class DummyClient:
        pass
    class gspread:
        Client = DummyClient

# Constants
WORKSHEET_NAME = "RadVision_Analyses"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_sheet_client():
    """
    Create and return a Google Sheets client using service account credentials.

    Returns:
        gspread.Client or None: A configured Google Sheets client, or None if credentials weren't found.
    """
    # First try loading from the gen.json file directly
    try:
        if os.path.exists("gen.json"):
            # Log file exists and attempt to load
            logger.info("Found gen.json file, attempting to use it for authentication")

            # Load credentials directly from the file
            credentials = Credentials.from_service_account_file(
                "gen.json",
                scopes=SCOPES
            )

            # Create and return the gspread client
            client = gspread.authorize(credentials)
            logger.info("Google Sheets client initialized successfully from gen.json file")
            return client
        else:
            logger.error("gen.json file not found in project directory")

    except Exception as file_error:
        logger.error(f"Error loading service account from gen.json: {file_error}")

    # Fall back to using environment variable if direct file loading fails
    service_account_json = os.environ.get("SERVICE_ACCOUNT_JSON")

    if not service_account_json:
        logger.error("SERVICE_ACCOUNT_JSON environment variable not found and gen.json not available")
        return None

    # Try multiple approaches to parse the JSON
    try:
        # First attempt: direct JSON parsing
        service_account_info = json.loads(service_account_json)

        # Create credentials from the service account info
        credentials = Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPES
        )

        # Create and return the gspread client
        client = gspread.authorize(credentials)
        logger.info("Google Sheets client initialized successfully from environment variable")
        return client
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in SERVICE_ACCOUNT_JSON: {e}")

        # Second attempt: clean up the JSON and try again
        try:
            # Remove any quotes at beginning and end if present
            cleaned_json = service_account_json.strip()
            if (cleaned_json.startswith('"') and cleaned_json.endswith('"')) or \
               (cleaned_json.startswith("'") and cleaned_json.endswith("'")):
                cleaned_json = cleaned_json[1:-1]

            # Replace escaped quotes
            cleaned_json = cleaned_json.replace('\\"', '"')
            # Fix newlines in private key
            cleaned_json = cleaned_json.replace('\\\\n', '\\n')

            # Try to parse the cleaned JSON
            service_account_info = json.loads(cleaned_json)

            # Create credentials from the service account info
            credentials = Credentials.from_service_account_info(
                service_account_info,
                scopes=SCOPES
            )

            # Create and return the gspread client
            client = gspread.authorize(credentials)
            logger.info("Google Sheets client initialized successfully after cleaning JSON")
            return client
        except json.JSONDecodeError as e2:
            logger.error(f"Invalid JSON in SERVICE_ACCOUNT_JSON after cleaning: {e2}")
            return None
    except Exception as e:
        logger.error(f"Error creating Google Sheets client: {e}")
        return None

def get_or_create_sheet() -> Optional[gspread.Spreadsheet]:
    """
    Get or create the tracking spreadsheet.

    Returns:
        gspread.Spreadsheet or None if unavailable
    """
    client = get_sheet_client()
    if not client:
        return None

    try:
        # Try to open existing sheet
        try:
            sheet = client.open(WORKSHEET_NAME)
            logger.info(f"Opened existing spreadsheet: {WORKSHEET_NAME}")
        except gspread.SpreadsheetNotFound:
            # Create a new spreadsheet if not found
            try:
                sheet = client.create(WORKSHEET_NAME)
                logger.info(f"Created new spreadsheet: {WORKSHEET_NAME}")

                # Set up worksheet with headers
                worksheet = sheet.get_worksheet(0)
                headers = [
                    "Timestamp", "Session ID", "Image Name", "Analysis Type",
                    "Key Findings", "Confidence Score", "UMLS Concepts"
                ]
                worksheet.append_row(headers)
            except Exception as api_error:
                if "drive.googleapis.com" in str(api_error) and "403" in str(api_error):
                    logger.error(f"Google Drive API is not enabled for this project: {api_error}")
                    # Re-raise with more specific error message
                    raise Exception("Google Drive API is not enabled for this service account. Please enable it in the Google Cloud Console.") from api_error
                raise

        return sheet
    except Exception as e:
        logger.error(f"Error getting or creating spreadsheet: {e}")
        return None

def log_analysis_to_sheet(analysis_data: Dict[str, Any]) -> bool:
    """
    Log an analysis to the Google Sheet.

    Args:
        analysis_data: Dictionary containing analysis information

    Returns:
        bool: True if successful, False otherwise
    """
    sheet = get_or_create_sheet()
    if not sheet:
        return False

    try:
        # Get the first worksheet
        worksheet = sheet.get_worksheet(0)
        if not worksheet:
            worksheet = sheet.add_worksheet("Analyses", 1000, 20)

        # Prepare row data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract UMLS concepts as comma-separated CUIs
        umls_concepts = analysis_data.get("umls_concepts", [])
        if umls_concepts:
            umls_str = ", ".join([f"{c.name} ({c.ui})" for c in umls_concepts[:5]])
            if len(umls_concepts) > 5:
                umls_str += f" and {len(umls_concepts)-5} more"
        else:
            umls_str = "None identified"

        # Format key findings to truncate if too long
        key_findings = analysis_data.get("key_findings", "")
        if isinstance(key_findings, str) and len(key_findings) > 1000:
            key_findings = key_findings[:997] + "..."
        elif not isinstance(key_findings, str):
            key_findings = str(key_findings) if key_findings is not None else "No findings available"

        row_data = [
            timestamp,
            analysis_data.get("session_id", "unknown"),
            analysis_data.get("image_name", "unnamed"),
            analysis_data.get("analysis_type", "General Analysis"),
            key_findings,
            analysis_data.get("confidence", "Not estimated"),
            umls_str
        ]

        # Add the row
        worksheet.append_row(row_data)
        logger.info(f"Successfully logged analysis for session {analysis_data.get('session_id')}")
        return True

    except Exception as e:
        logger.error(f"Error logging analysis to sheet: {e}")
        return False

def get_analyses_history() -> Optional[pd.DataFrame]:
    """
    Retrieve analysis history from the Google Sheet.

    Returns:
        pandas.DataFrame or None if unavailable
    """
    sheet = get_or_create_sheet()
    if not sheet:
        return None

    try:
        # Get the first worksheet
        worksheet = sheet.get_worksheet(0)
        if not worksheet:
            logger.warning("Worksheet not found.")
            return None

        # Get all values
        records = worksheet.get_all_records()
        if not records:
            logger.info("No analysis history found.")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(records)

        # Sort by timestamp (most recent first)
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values('Timestamp', ascending=False)

        return df

    except Exception as e:
        logger.error(f"Error retrieving analysis history: {e}")
        return None

def delete_analysis(session_id: str, timestamp: str) -> bool:
    """
    Delete a specific analysis entry.

    Args:
        session_id: Session ID of the analysis to delete
        timestamp: Timestamp of the analysis to delete

    Returns:
        bool: True if successful, False otherwise
    """
    if not session_id or not timestamp:
        return False

    sheet = get_or_create_sheet()
    if not sheet:
        return False

    try:
        # Get the first worksheet
        worksheet = sheet.get_worksheet(0)
        if not worksheet:
            return False

        # Find the row with matching session_id and timestamp
        all_values = worksheet.get_all_values()
        headers = all_values[0]

        # Get column indices
        try:
            timestamp_idx = headers.index("Timestamp")
            session_id_idx = headers.index("Session ID")
        except ValueError:
            logger.error("Required columns not found in spreadsheet.")
            return False

        # Look for the matching row
        row_idx = None
        for i, row in enumerate(all_values[1:], start=2):  # Start from 2 (1-indexed + header)
            if (row[timestamp_idx] == timestamp and 
                row[session_id_idx] == session_id):
                row_idx = i
                break

        if row_idx:
            worksheet.delete_rows(row_idx)
            logger.info(f"Deleted analysis for session {session_id} at {timestamp}")
            return True
        else:
            logger.warning(f"No matching analysis found for session {session_id} at {timestamp}")
            return False

    except Exception as e:
        logger.error(f"Error deleting analysis: {e}")
        return False

def render_sheet_history() -> None:
    """
    Render the analysis history in Streamlit.
    """
    if not SHEETS_AVAILABLE:
        st.warning("Google Sheets integration unavailable. Install required packages.")
        st.code("pip install gspread google-auth pandas", language="bash")
        return

    # Check for service account
    if not os.environ.get("SERVICE_ACCOUNT_JSON"):
        st.warning("Google Sheets integration requires SERVICE_ACCOUNT_JSON to be set.")
        return

    # Get history data
    with st.spinner("Fetching analysis history..."):
        df = get_analyses_history()

    if df is None:
        st.error("Failed to fetch analysis history. Check logs for details.")
        return

    if df.empty:
        st.info("No analysis history found. Analyze some images to start recording history.")
        return

    # Display the history
    st.success(f"Found {len(df)} past analyses")

    # Enhance dataframe display
    # Adjust timestamp format for better display
    if 'Timestamp' in df.columns:
        df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')

    # Add deletion functionality
    if st.checkbox("Enable Delete Mode", key="enable_delete_history"):
        # Add a selection column
        df['Delete'] = False
        selected_df = st.data_editor(
            df,
            column_config={
                "Delete": st.column_config.CheckboxColumn(
                    "Select to Delete",
                    help="Select rows to delete",
                    default=False,
                ),
                "Key Findings": st.column_config.TextColumn(
                    "Key Findings",
                    width="large",
                    help="Main analysis points"
                ),
                "UMLS Concepts": st.column_config.TextColumn(
                    "UMLS Concepts",
                    width="medium",
                    help="Linked medical concepts"
                )
            },
            hide_index=True,
            use_container_width=True
        )

        # Delete selected rows
        if st.button("Delete Selected Entries", type="primary", use_container_width=True):
            selected_rows = selected_df[selected_df['Delete']].copy()
            if not selected_rows.empty:
                with st.spinner(f"Deleting {len(selected_rows)} entries..."):
                    success = True
                    for _, row in selected_rows.iterrows():
                        timestamp = row['Timestamp']
                        session_id = row['Session ID']
                        if not delete_analysis(session_id, timestamp):
                            success = False

                    if success:
                        st.success("Selected entries deleted successfully.")
                        st.session_state.refresh_case_history = True
                        st.rerun()
                    else:
                        st.error("Some entries could not be deleted. Please try again.")
    else:
        # Standard view without delete option
        st.dataframe(
            df,
            column_config={
                "Key Findings": st.column_config.TextColumn(
                    "Key Findings",
                    width="large",
                    help="Main analysis points"
                ),
                "UMLS Concepts": st.column_config.TextColumn(
                    "UMLS Concepts",
                    width="medium", 
                    help="Linked medical concepts"
                )
            },
            hide_index=True,
            use_container_width=True
        )

    # Statistics on case history
    if len(df) > 1:
        st.subheader("Analysis Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Most common condition
            if 'Analysis Type' in df.columns:
                condition_counts = df['Analysis Type'].value_counts()
                if not condition_counts.empty:
                    st.metric("Most Common Condition", condition_counts.index[0], 
                             f"{condition_counts.iloc[0]} cases")

        with col2:
            # Average confidence
            if 'Confidence Score' in df.columns:
                # Try to extract numeric confidence values
                try:
                    df['Numeric Confidence'] = df['Confidence Score'].str.extract(r'(\d+)').astype(float)
                    avg_confidence = df['Numeric Confidence'].mean()
                    if not pd.isna(avg_confidence):
                        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                except:
                    pass

        with col3:
            # Cases per day
            if 'Timestamp' in df.columns:
                df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
                days = (df['Date'].max() - df['Date'].min()).days
                if days > 0:
                    rate = len(df) / max(1, days)
                    st.metric("Cases Per Day", f"{rate:.1f}")

        # Timeline chart
        if 'Timestamp' in df.columns:
            try:
                date_counts = df.resample('D', on='Timestamp').size()
                if len(date_counts) > 1:
                    st.subheader("Analysis Timeline")
                    st.bar_chart(date_counts)
            except:
                pass