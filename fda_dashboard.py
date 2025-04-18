
"""
fda_dashboard.py - FDA Compliance Dashboard for RadVision AI
========================================================

Provides a comprehensive dashboard to track FDA compliance requirements,
view documentation status, and manage compliance tasks.
"""

import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logger
import logging
logger = logging.getLogger(__name__)

# Load FDA requirements from files
def load_fda_requirements() -> Dict[str, Any]:
    """Load FDA requirements from Markdown files and other sources"""
    requirements = {
        "510k_checklist": [],
        "documentation": {},
        "validation": {},
        "risk_assessment": {}
    }
    
    # Load 510k checklist if available
    try:
        if os.path.exists("fda_docs/510k_checklist.md"):
            with open("fda_docs/510k_checklist.md", "r") as f:
                content = f.read()
                
                # Parse checklist items
                import re
                checklist_items = re.findall(r'- \[ \] (.*?)$', content, re.MULTILINE)
                
                # Add to requirements
                requirements["510k_checklist"] = [
                    {"item": item, "status": "Not Started"} for item in checklist_items
                ]
    except Exception as e:
        logger.error(f"Error loading FDA requirements: {e}")
    
    # Load FDA requirements markdown
    try:
        if os.path.exists("fda_requirements.md"):
            with open("fda_requirements.md", "r") as f:
                requirements["general_requirements"] = f.read()
    except Exception as e:
        logger.error(f"Error loading FDA requirements markdown: {e}")
    
    return requirements

# Load compliance status
def load_compliance_status() -> Dict[str, Any]:
    """Load saved compliance status or create default"""
    try:
        if os.path.exists("fda_compliance_status.json"):
            with open("fda_compliance_status.json", "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading compliance status: {e}")
    
    # Default status
    return {
        "overall_status": "Not FDA Ready",
        "last_updated": datetime.now().isoformat(),
        "checklist_items": {},
        "documentation_status": {
            "User Manual": "Not Started",
            "Software Requirements": "Not Started",
            "Risk Analysis": "Not Started",
            "Verification & Validation": "Not Started",
            "Clinical Validation": "Not Started"
        },
        "validation_status": {
            "Unit Tests": "Not Started",
            "Integration Tests": "Not Started",
            "Performance Testing": "Not Started",
            "Clinical Testing": "Not Started"
        }
    }

# Save compliance status
def save_compliance_status(status: Dict[str, Any]) -> bool:
    """Save compliance status to JSON file"""
    try:
        status["last_updated"] = datetime.now().isoformat()
        
        with open("fda_compliance_status.json", "w") as f:
            json.dump(status, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving compliance status: {e}")
        return False

def render_fda_dashboard():
    """Render the FDA compliance dashboard"""
    st.title("🏥 FDA Compliance Dashboard")
    
    # Add a back button at the top
    if st.button("← Back to Main Interface", key="fda_dashboard_back"):
        st.session_state.active_view = "main"
        st.rerun()
    
    # Load requirements and status
    requirements = load_fda_requirements()
    compliance_status = load_compliance_status()
    
    # Overview section
    st.header("FDA Compliance Overview")
    
    # Overall status with color
    status_color = {
        "Not FDA Ready": "🔴",
        "In Progress": "🟠",
        "Nearly Ready": "🟡",
        "FDA Ready": "🟢"
    }
    
    current_status = compliance_status.get("overall_status", "Not FDA Ready")
    status_icon = status_color.get(current_status, "🔴")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"### {status_icon} {current_status}")
    
    with col2:
        # Status selector
        new_status = st.selectbox(
            "Update Overall Status:",
            options=list(status_color.keys()),
            index=list(status_color.keys()).index(current_status)
        )
        
        if new_status != current_status:
            compliance_status["overall_status"] = new_status
            save_compliance_status(compliance_status)
            st.success(f"Status updated to {new_status}")
            st.rerun()
    
    # Display last updated time
    last_updated = compliance_status.get("last_updated", "Never")
    if last_updated != "Never":
        try:
            last_updated_dt = datetime.fromisoformat(last_updated)
            last_updated = last_updated_dt.strftime("%Y-%m-%d %H:%M")
        except:
            pass
    
    st.caption(f"Last Updated: {last_updated}")
    
    # Display disclaimer
    st.info("""
    **Disclaimer:** This dashboard is for tracking FDA compliance requirements. 
    RadVision AI is currently NOT FDA cleared or approved and is labeled for research and educational use only.
    """)
    
    # Main content tabs
    tabs = st.tabs([
        "📋 510(k) Checklist", 
        "📑 Documentation", 
        "🧪 Validation", 
        "⚠️ Risk Assessment",
        "📊 Compliance Summary"
    ])
    
    # Tab 1: 510(k) Checklist
    with tabs[0]:
        st.subheader("510(k) Submission Checklist")
        st.caption("Track required items for a potential 510(k) submission")
        
        checklist_items = requirements.get("510k_checklist", [])
        
        if not checklist_items:
            st.warning("No checklist items found. Please check fda_docs/510k_checklist.md")
        else:
            # Convert to DataFrame for better display
            checklist_df = pd.DataFrame(checklist_items)
            
            # Get saved statuses
            saved_statuses = compliance_status.get("checklist_items", {})
            
            # Add status column with default values
            checklist_df["status"] = checklist_df["item"].apply(
                lambda x: saved_statuses.get(x, "Not Started")
            )
            
            # Create editable dataframe
            edited_df = st.data_editor(
                checklist_df,
                column_config={
                    "item": st.column_config.TextColumn(
                        "Checklist Item",
                        width="large"
                    ),
                    "status": st.column_config.SelectboxColumn(
                        "Status",
                        width="medium",
                        options=["Not Started", "In Progress", "Complete", "N/A"],
                        required=True
                    )
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Save changes if any
            if not edited_df.equals(checklist_df):
                # Update status dict
                new_statuses = {}
                for _, row in edited_df.iterrows():
                    new_statuses[row["item"]] = row["status"]
                
                compliance_status["checklist_items"] = new_statuses
                if save_compliance_status(compliance_status):
                    st.success("Checklist updated")
            
            # Show progress stats
            if not edited_df.empty:
                status_counts = edited_df["status"].value_counts()
                
                total = len(edited_df)
                complete = status_counts.get("Complete", 0)
                na_items = status_counts.get("N/A", 0)
                applicable = total - na_items
                
                if applicable > 0:
                    progress = (complete / applicable) * 100
                else:
                    progress = 0
                
                st.progress(progress / 100, text=f"Progress: {progress:.1f}% ({complete}/{applicable} applicable items)")
    
    # Tab 2: Documentation
    with tabs[1]:
        st.subheader("FDA Documentation Status")
        st.caption("Track required documentation for FDA submission")
        
        # Documentation items
        doc_items = [
            "User Manual",
            "Software Requirements Specification",
            "Software Design Specification",
            "Hazard Analysis & Risk Assessment",
            "Verification & Validation Plan",
            "Verification & Validation Results",
            "Clinical Validation Protocol",
            "Clinical Validation Results",
            "Labeling & Instructions for Use",
            "510(k) Summary"
        ]
        
        # Get saved statuses
        doc_statuses = compliance_status.get("documentation_status", {})
        
        # Create DataFrame
        doc_df = pd.DataFrame({
            "document": doc_items,
            "status": [doc_statuses.get(item, "Not Started") for item in doc_items],
            "notes": ["" for _ in doc_items]
        })
        
        # Edit documentation status
        edited_doc_df = st.data_editor(
            doc_df,
            column_config={
                "document": st.column_config.TextColumn(
                    "Document",
                    width="large"
                ),
                "status": st.column_config.SelectboxColumn(
                    "Status",
                    width="medium",
                    options=["Not Started", "In Progress", "Draft Complete", "Final", "N/A"],
                    required=True
                ),
                "notes": st.column_config.TextColumn(
                    "Notes",
                    width="large"
                )
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Save changes if any
        if not edited_doc_df.equals(doc_df):
            # Update status dict
            new_doc_statuses = {}
            for _, row in edited_doc_df.iterrows():
                new_doc_statuses[row["document"]] = row["status"]
            
            compliance_status["documentation_status"] = new_doc_statuses
            if save_compliance_status(compliance_status):
                st.success("Documentation status updated")
        
        # Show progress stats
        if not edited_doc_df.empty:
            doc_status_counts = edited_doc_df["status"].value_counts()
            
            total_docs = len(edited_doc_df)
            final_docs = doc_status_counts.get("Final", 0)
            draft_docs = doc_status_counts.get("Draft Complete", 0)
            na_docs = doc_status_counts.get("N/A", 0)
            applicable_docs = total_docs - na_docs
            
            if applicable_docs > 0:
                doc_progress = ((final_docs + (draft_docs * 0.5)) / applicable_docs) * 100
            else:
                doc_progress = 0
            
            st.progress(doc_progress / 100, text=f"Documentation Progress: {doc_progress:.1f}%")
    
    # Tab 3: Validation
    with tabs[2]:
        st.subheader("Software Validation & Verification")
        st.caption("Track validation activities required for FDA submission")
        
        # Validation activities
        validation_items = [
            "Unit Tests",
            "Integration Tests",
            "System Tests",
            "Performance Tests",
            "User Acceptance Tests",
            "Clinical Algorithm Validation",
            "Human Factors Validation",
            "Edge Case Testing",
            "Cybersecurity Testing",
            "Interoperability Testing"
        ]
        
        # Get saved statuses
        validation_statuses = compliance_status.get("validation_status", {})
        
        # Create DataFrame
        validation_df = pd.DataFrame({
            "test_type": validation_items,
            "status": [validation_statuses.get(item, "Not Started") for item in validation_items],
            "pass_rate": [0 for _ in validation_items],
            "notes": ["" for _ in validation_items]
        })
        
        # Edit validation status
        edited_validation_df = st.data_editor(
            validation_df,
            column_config={
                "test_type": st.column_config.TextColumn(
                    "Validation Activity",
                    width="large"
                ),
                "status": st.column_config.SelectboxColumn(
                    "Status",
                    width="medium",
                    options=["Not Started", "In Progress", "Complete", "Failed", "N/A"],
                    required=True
                ),
                "pass_rate": st.column_config.NumberColumn(
                    "Pass Rate (%)",
                    min_value=0,
                    max_value=100,
                    step=0.1,
                    format="%.1f%%"
                ),
                "notes": st.column_config.TextColumn(
                    "Notes",
                    width="large"
                )
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Save changes if any
        if not edited_validation_df.equals(validation_df):
            # Update status dict
            new_validation_statuses = {}
            for _, row in edited_validation_df.iterrows():
                new_validation_statuses[row["test_type"]] = row["status"]
            
            compliance_status["validation_status"] = new_validation_statuses
            
            # Save pass rates
            if "validation_pass_rates" not in compliance_status:
                compliance_status["validation_pass_rates"] = {}
            
            for _, row in edited_validation_df.iterrows():
                compliance_status["validation_pass_rates"][row["test_type"]] = row["pass_rate"]
            
            if save_compliance_status(compliance_status):
                st.success("Validation status updated")
        
        # Show validation stats
        if not edited_validation_df.empty:
            validation_status_counts = edited_validation_df["status"].value_counts()
            
            total_validations = len(edited_validation_df)
            complete_validations = validation_status_counts.get("Complete", 0)
            na_validations = validation_status_counts.get("N/A", 0)
            applicable_validations = total_validations - na_validations
            
            if applicable_validations > 0:
                validation_progress = (complete_validations / applicable_validations) * 100
            else:
                validation_progress = 0
            
            st.progress(validation_progress / 100, text=f"Validation Progress: {validation_progress:.1f}%")
            
            # Chart of pass rates
            complete_tests = edited_validation_df[
                (edited_validation_df["status"] == "Complete") & 
                (edited_validation_df["pass_rate"] > 0)
            ]
            
            if not complete_tests.empty:
                st.subheader("Validation Pass Rates")
                chart_data = pd.DataFrame({
                    "Test": complete_tests["test_type"],
                    "Pass Rate (%)": complete_tests["pass_rate"]
                })
                st.bar_chart(chart_data.set_index("Test"))
    
    # Tab 4: Risk Assessment
    with tabs[3]:
        st.subheader("Risk Assessment & Mitigation")
        st.caption("Track identified risks and mitigation strategies")
        
        # Load or initialize risks
        if "risks" not in compliance_status:
            compliance_status["risks"] = []
        
        risks = compliance_status.get("risks", [])
        
        # Convert to DataFrame
        risk_df = pd.DataFrame(risks) if risks else pd.DataFrame({
            "risk_category": [],
            "description": [],
            "severity": [],
            "probability": [],
            "mitigation": [],
            "status": []
        })
        
        # Add new risk form
        with st.expander("Add New Risk", expanded=len(risks) == 0):
            with st.form("add_risk_form"):
                risk_category = st.selectbox(
                    "Risk Category",
                    options=[
                        "Clinical Performance", 
                        "Software Functionality",
                        "Cybersecurity",
                        "Data Privacy",
                        "User Error",
                        "Integration/Interoperability",
                        "Other"
                    ]
                )
                
                risk_description = st.text_area(
                    "Risk Description",
                    placeholder="Describe the potential risk..."
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    severity = st.selectbox(
                        "Severity",
                        options=["Low", "Medium", "High", "Critical"]
                    )
                
                with col2:
                    probability = st.selectbox(
                        "Probability",
                        options=["Low", "Medium", "High"]
                    )
                
                mitigation = st.text_area(
                    "Mitigation Strategy",
                    placeholder="Describe how this risk will be mitigated..."
                )
                
                risk_status = st.selectbox(
                    "Status",
                    options=["Identified", "Mitigation Planned", "Mitigated", "Accepted"]
                )
                
                submit_risk = st.form_submit_button("Add Risk")
                
                if submit_risk:
                    if not risk_description or not mitigation:
                        st.error("Please provide both a risk description and mitigation strategy")
                    else:
                        new_risk = {
                            "risk_category": risk_category,
                            "description": risk_description,
                            "severity": severity,
                            "probability": probability,
                            "mitigation": mitigation,
                            "status": risk_status
                        }
                        
                        compliance_status["risks"].append(new_risk)
                        save_compliance_status(compliance_status)
                        st.success("Risk added successfully")
                        st.rerun()
        
        # Display risks
        if len(risk_df) == 0:
            st.info("No risks have been identified yet. Use the form above to add risks.")
        else:
            # Edit risks
            edited_risk_df = st.data_editor(
                risk_df,
                column_config={
                    "risk_category": st.column_config.SelectboxColumn(
                        "Category",
                        width="medium",
                        options=[
                            "Clinical Performance", 
                            "Software Functionality",
                            "Cybersecurity",
                            "Data Privacy",
                            "User Error",
                            "Integration/Interoperability",
                            "Other"
                        ]
                    ),
                    "description": st.column_config.TextColumn(
                        "Description",
                        width="large"
                    ),
                    "severity": st.column_config.SelectboxColumn(
                        "Severity",
                        width="small",
                        options=["Low", "Medium", "High", "Critical"]
                    ),
                    "probability": st.column_config.SelectboxColumn(
                        "Probability",
                        width="small",
                        options=["Low", "Medium", "High"]
                    ),
                    "mitigation": st.column_config.TextColumn(
                        "Mitigation",
                        width="large"
                    ),
                    "status": st.column_config.SelectboxColumn(
                        "Status",
                        width="medium",
                        options=["Identified", "Mitigation Planned", "Mitigated", "Accepted"]
                    )
                },
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic"
            )
            
            # Save changes if any
            if not edited_risk_df.equals(risk_df) and len(edited_risk_df) > 0:
                compliance_status["risks"] = edited_risk_df.to_dict('records')
                if save_compliance_status(compliance_status):
                    st.success("Risk assessment updated")
            
            # Risk statistics
            if not edited_risk_df.empty:
                st.subheader("Risk Statistics")
                
                # Count by severity
                severity_counts = edited_risk_df["severity"].value_counts()
                
                # Count by status
                status_counts = edited_risk_df["status"].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.caption("Risks by Severity")
                    severity_chart = pd.DataFrame({
                        "Severity": severity_counts.index,
                        "Count": severity_counts.values
                    })
                    st.bar_chart(severity_chart.set_index("Severity"))
                
                with col2:
                    st.caption("Risks by Status")
                    status_chart = pd.DataFrame({
                        "Status": status_counts.index,
                        "Count": status_counts.values
                    })
                    st.bar_chart(status_chart.set_index("Status"))
                
                # Risk matrix
                st.subheader("Risk Matrix")
                
                # Create risk matrix data
                matrix_data = {
                    "Low": {"Low": 0, "Medium": 0, "High": 0},
                    "Medium": {"Low": 0, "Medium": 0, "High": 0},
                    "High": {"Low": 0, "Medium": 0, "High": 0},
                    "Critical": {"Low": 0, "Medium": 0, "High": 0}
                }
                
                for _, row in edited_risk_df.iterrows():
                    severity = row["severity"]
                    probability = row["probability"]
                    if severity in matrix_data and probability in matrix_data[severity]:
                        matrix_data[severity][probability] += 1
                
                # Convert to DataFrame for display
                matrix_df = pd.DataFrame(matrix_data)
                
                # Display as heatmap
                import plotly.figure_factory as ff
                import numpy as np
                
                # Create matrix values
                z = np.array([
                    [matrix_data["Low"]["Low"], matrix_data["Medium"]["Low"], matrix_data["High"]["Low"], matrix_data["Critical"]["Low"]],
                    [matrix_data["Low"]["Medium"], matrix_data["Medium"]["Medium"], matrix_data["High"]["Medium"], matrix_data["Critical"]["Medium"]],
                    [matrix_data["Low"]["High"], matrix_data["Medium"]["High"], matrix_data["High"]["High"], matrix_data["Critical"]["High"]]
                ])
                
                x = ["Low", "Medium", "High", "Critical"]
                y = ["Low", "Medium", "High"]
                
                # Create heatmap
                fig = ff.create_annotated_heatmap(
                    z=z,
                    x=x,
                    y=y,
                    annotation_text=z.astype(str),
                    colorscale=[
                        [0, "green"],
                        [0.25, "lightgreen"],
                        [0.5, "yellow"],
                        [0.75, "orange"],
                        [1, "red"]
                    ],
                    hoverinfo="text",
                    showscale=True
                )
                
                fig.update_layout(
                    title="Risk Matrix (Severity vs. Probability)",
                    xaxis_title="Severity",
                    yaxis_title="Probability",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Compliance Summary
    with tabs[4]:
        st.subheader("FDA Compliance Overview")
        
        # Display general requirements
        if "general_requirements" in requirements:
            with st.expander("FDA Requirements Overview", expanded=True):
                st.markdown(requirements["general_requirements"])
        
        # Calculate overall compliance scores
        overall_scores = {}
        
        # Checklist score
        checklist_items = compliance_status.get("checklist_items", {})
        if checklist_items:
            complete_items = sum(1 for status in checklist_items.values() if status == "Complete")
            total_items = len(checklist_items)
            na_items = sum(1 for status in checklist_items.values() if status == "N/A")
            applicable_items = total_items - na_items
            
            if applicable_items > 0:
                checklist_score = (complete_items / applicable_items) * 100
            else:
                checklist_score = 0
            
            overall_scores["510(k) Checklist"] = checklist_score
        
        # Documentation score
        doc_statuses = compliance_status.get("documentation_status", {})
        if doc_statuses:
            doc_scores = {
                "Final": 1.0,
                "Draft Complete": 0.7,
                "In Progress": 0.3,
                "Not Started": 0,
                "N/A": None  # Exclude from calculation
            }
            
            applicable_docs = [status for status in doc_statuses.values() if status != "N/A"]
            if applicable_docs:
                doc_score = sum(doc_scores.get(status, 0) for status in applicable_docs) / len(applicable_docs) * 100
            else:
                doc_score = 0
            
            overall_scores["Documentation"] = doc_score
        
        # Validation score
        validation_statuses = compliance_status.get("validation_status", {})
        if validation_statuses:
            validation_scores = {
                "Complete": 1.0,
                "In Progress": 0.5,
                "Not Started": 0,
                "Failed": 0.25,  # Some progress but not complete
                "N/A": None  # Exclude from calculation
            }
            
            applicable_validations = [status for status in validation_statuses.values() if status != "N/A"]
            if applicable_validations:
                validation_score = sum(validation_scores.get(status, 0) for status in applicable_validations) / len(applicable_validations) * 100
            else:
                validation_score = 0
            
            overall_scores["Validation"] = validation_score
        
        # Risk assessment score
        risks = compliance_status.get("risks", [])
        if risks:
            risk_scores = {
                "Mitigated": 1.0,
                "Accepted": 0.9,
                "Mitigation Planned": 0.5,
                "Identified": 0.2
            }
            
            risk_score = sum(risk_scores.get(risk.get("status", "Identified"), 0) for risk in risks) / len(risks) * 100
            overall_scores["Risk Assessment"] = risk_score
        
        # Display scores as gauge charts
        if overall_scores:
            st.subheader("Compliance Readiness by Category")
            
            import plotly.express as px
            
            # Create a row of gauges
            cols = st.columns(len(overall_scores))
            
            for i, (category, score) in enumerate(overall_scores.items()):
                with cols[i]:
                    # Create a gauge chart using go instead of px
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(go.Indicator(
                        value=score,
                        mode="gauge+number",
                        title={"text": category},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 30], "color": "red"},
                                {"range": [30, 70], "color": "orange"},
                                {"range": [70, 90], "color": "yellow"},
                                {"range": [90, 100], "color": "green"}
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": 90
                            }
                        }
                    ))
                    
                    fig.update_layout(height=250)
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Calculate overall readiness
            overall_readiness = sum(overall_scores.values()) / len(overall_scores)
            
            st.subheader("Overall FDA Readiness")
            
            # Create overall gauge using go instead of px
            import plotly.graph_objects as go
            
            fig = go.Figure(go.Indicator(
                value=overall_readiness,
                mode="gauge+number",
                title={"text": "Overall Readiness Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 30], "color": "red"},
                        {"range": [30, 70], "color": "orange"},
                        {"range": [70, 90], "color": "yellow"},
                        {"range": [90, 100], "color": "green"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommended status based on score
            recommended_status = "Not FDA Ready"
            if overall_readiness >= 90:
                recommended_status = "FDA Ready"
            elif overall_readiness >= 70:
                recommended_status = "Nearly Ready"
            elif overall_readiness >= 30:
                recommended_status = "In Progress"
            
            if recommended_status != compliance_status.get("overall_status"):
                st.info(f"Recommended status based on scores: **{recommended_status}**")
                
                if st.button(f"Update Status to '{recommended_status}'"):
                    compliance_status["overall_status"] = recommended_status
                    save_compliance_status(compliance_status)
                    st.success(f"Status updated to {recommended_status}")
                    st.rerun()
