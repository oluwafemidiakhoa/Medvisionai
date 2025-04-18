
import streamlit as st
import time
import pandas as pd
from report_utils import generate_smart_recommendations

def render_smart_recommendations_ui():
    """Render UI for testing and evaluating Smart Recommendations feature."""
    
    st.title("Smart Recommendations Testing")
    st.markdown("""
    This tool allows you to test and evaluate the Smart Recommendations feature 
    of RadVision AI, which generates appropriate clinical recommendations based on findings.
    """)
    
    # Input for findings text
    findings_text = st.text_area(
        "Enter radiological findings text:",
        height=150,
        placeholder="Example: There is a consolidation in the right lower lobe consistent with pneumonia."
    )
    
    # Buttons for preset findings
    st.markdown("### Preset Findings")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Pneumonia"):
            findings_text = "There is a consolidation in the right lower lobe consistent with pneumonia."
            st.session_state.findings_text = findings_text
    with col2:
        if st.button("Tuberculosis"):
            findings_text = "Cavitary lesion in the right upper lobe with surrounding fibrosis, consistent with tuberculosis."
            st.session_state.findings_text = findings_text
    with col3:
        if st.button("Cardiomegaly"):
            findings_text = "Enlarged cardiac silhouette consistent with cardiomegaly. No pulmonary edema."
            st.session_state.findings_text = findings_text
            
    # Use session state if available
    if hasattr(st.session_state, 'findings_text'):
        findings_text = st.session_state.findings_text
    
    # Generate recommendations button
    if st.button("Generate Recommendations") and findings_text:
        with st.spinner("Generating smart recommendations..."):
            start_time = time.time()
            recommendations = generate_smart_recommendations(findings_text)
            execution_time = time.time() - start_time
            
            # Display recommendations
            st.success(f"Generated {len(recommendations)} recommendations in {execution_time:.4f} seconds")
            
            st.subheader("Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
            
            # Rate the recommendations
            st.subheader("Rate these recommendations:")
            accuracy = st.slider("Accuracy", 1, 5, 3, 1)
            relevance = st.slider("Clinical Relevance", 1, 5, 3, 1)
            completeness = st.slider("Completeness", 1, 5, 3, 1)
            
            if st.button("Submit Rating"):
                # In a real implementation, you would save this rating
                st.success("Rating submitted! Thank you for your feedback.")
    
    # Show recommendations history
    with st.expander("Recommendations History", expanded=False):
        st.info("This would show history of previously generated recommendations")
        
        # Example history table (replace with actual data in a real implementation)
        history_data = {
            "Date": ["2025-04-17", "2025-04-16", "2025-04-15"],
            "Findings": ["Pneumonia", "Tuberculosis", "Normal study"],
            "Recommendations": ["4", "3", "2"],
            "Avg. Rating": ["4.5", "3.7", "4.2"]
        }
        
        st.table(pd.DataFrame(history_data))

if __name__ == "__main__":
    render_smart_recommendations_ui()
